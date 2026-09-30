import torch
import torch.nn.functional as F


class Controller:
    def __init__(
        self,
        labels,
        alpha=0.5,
        rho=0.05,
        lam=0.5,
        beta=0.9,
        is_clip=10.0,
        eps=0.1,
        seed=42,
        same_label=False,
        correction=True,
        top_k=20,
        search_batch_size=128,
        sender=True,
        receiver=True,
    ):
        self.sender, self.receiver = sender, receiver
        self.labels = labels
        self.n = len(labels)
        self.alpha, self.rho, self.lam, self.beta = alpha, rho, lam, beta
        self.is_clip, self.eps = is_clip, eps
        self.same_label, self.correction = same_label, correction
        self.w = torch.ones(self.n, device=labels.device)
        self.available = torch.ones_like(labels, dtype=torch.bool)
        self.last_seen = torch.zeros_like(self.w)
        self.g_bar = None
        self.ready = False
        self.epoch = -1
        self.top_k = top_k
        self.search_batch_size = search_batch_size
        self.embedding_cache = None
        self.embedding_valid = torch.zeros(
            self.n, dtype=torch.bool, device=labels.device
        )
        self.embedding_step = torch.full(
            (self.n,), -1, dtype=torch.long, device=labels.device
        )
        self.rows = torch.full(
            (self.n, top_k), -1, dtype=torch.long, device=labels.device
        )
        self.neff = torch.zeros_like(self.w)
        self.row_valid = torch.zeros_like(self.available)
        self.indeg = torch.zeros_like(self.w)
        self.psi = torch.ones_like(self.w)
        self.nu = torch.ones_like(self.w)
        self.pending = None
        self.generator = torch.Generator(device=labels.device).manual_seed(seed)

    @torch.no_grad()
    def cache_embeddings(self, ids, features, step):
        z = F.normalize(features.detach().float(), dim=1)
        if self.embedding_cache is None:
            self.embedding_cache = torch.zeros(self.n, z.shape[1], device=z.device)
        self.embedding_cache[ids] = z
        self.embedding_valid[ids] = True
        self.embedding_step[ids] = step

    @torch.no_grad()
    def search(self, ids):
        candidates = torch.where(self.available & self.embedding_valid)[0]
        k = min(self.top_k, len(candidates))
        if not k:
            return ids.new_empty((len(ids), 0)), self.w.new_empty((len(ids), 0))
        neighbors, similarities = [], []
        bank = self.embedding_cache[candidates]
        for begin in range(0, len(ids), self.search_batch_size):
            scores = (
                self.embedding_cache[ids[begin : begin + self.search_batch_size]]
                @ bank.T
            )
            values, positions = scores.topk(k, dim=1)
            neighbors.append(candidates[positions])
            similarities.append(values)
        return torch.cat(neighbors), torch.cat(similarities)

    @torch.no_grad()
    def update_geometry(self, ids, nb, sims):
        previous = self.rows[ids]
        previous = previous[previous >= 0]
        self.indeg.index_add_(
            0, previous, -torch.ones_like(previous, dtype=self.w.dtype)
        )
        self.rows[ids] = -1
        if nb.shape[1]:
            self.rows[ids, : nb.shape[1]] = nb
            self.indeg.index_add_(
                0, nb.flatten(), torch.ones_like(nb.flatten(), dtype=self.w.dtype)
            )
        s = sims.clamp_min(0)
        self.neff[ids] = s.sum(1).square() / s.square().sum(1).clamp_min(1e-12)
        self.row_valid[ids] = True
        ln = self.neff[self.row_valid].clamp_min(1e-3).log()
        self.psi[self.row_valid] = torch.exp(
            -0.5 * ((ln - ln.mean()) / ln.std(correction=0).clamp_min(1e-8)).square()
        )
        self.psi[self.row_valid & (self.neff == 0)] = 0
        self.nu = (self.indeg.median() / self.indeg.clamp_min(1)).clamp(0.1, 3.0)
        if self.indeg.median() == 0:
            self.nu.fill_(1.0)

    def new_epoch(self):
        if self.pending is not None:
            raise RuntimeError("Missing feedback from previous batch")
        self.epoch += 1
        self.ready = self.epoch > 0
        if self.ready and not bool(self.embedding_valid.all()):
            raise RuntimeError("Complete full-data warm-up before enabling live search")
        self.available.fill_(True)

    def sample(self, m, step, spe):
        if self.pending is not None:
            raise RuntimeError("Must observe previous batch before sampling")
        stale = ((step - self.last_seen) / spe).clamp(0, 2)
        eff = self.w * (1 + self.lam * stale) if self.ready else torch.ones_like(self.w)
        eff = eff * self.available
        p = eff / eff.sum()
        ids = torch.multinomial(eff, m, replacement=False, generator=self.generator)
        n_candidates = int(self.available.sum())
        raw = 1 / (n_candidates * p[ids]).clamp_min(1e-12)
        iw = (
            raw.clamp(max=self.is_clip)
            if self.correction and self.ready
            else torch.ones_like(raw)
        )
        self.available[ids] = False
        self.pending = ids.clone()
        return (
            ids,
            iw,
            {
                "probabilities": p[ids],
                "raw_is": raw,
                "n_candidates": n_candidates,
                "is_clip_rate": float((raw > self.is_clip).float().mean())
                if self.correction and self.ready
                else 0.0,
                "stale_max": float(stale.max()),
                "stale": stale[ids],
            },
        )

    @torch.no_grad()
    def observe(self, ids, logits, targets, step):
        if self.pending is None or not torch.equal(ids, self.pending):
            raise RuntimeError("Feedback IDs differ from sampled batch")
        p = logits.detach().softmax(1)
        q = torch.full_like(p, self.eps / p.shape[1])
        q.scatter_(1, targets[:, None], 1 - self.eps + self.eps / p.shape[1])
        g = (p - q).norm(dim=1)
        self.g_bar = (
            g.mean()
            if self.g_bar is None
            else self.beta * self.g_bar + (1 - self.beta) * g.mean()
        )
        gh = (g / self.g_bar.clamp_min(1e-12)).clamp(1e-3, 1e3)
        self.w.mul_(1 - self.rho).add_(self.rho)
        touched = 0
        affected_count = torch.zeros_like(ids)
        if self.ready:
            nb, sims = self.search(ids)
            self.update_geometry(ids, nb, sims)
            s = sims.clamp_min(0)
            sender = self.psi[ids] if self.sender else torch.ones_like(gh)
            receiver = self.nu[nb] if self.receiver else torch.ones_like(s)
            lg = (self.alpha * gh.log() * sender)[:, None] * s * receiver
            valid = torch.ones_like(nb, dtype=torch.bool)
            match = self.labels[nb] == targets[:, None]
            if self.same_label:
                valid &= match
            sums, counts = torch.zeros_like(self.w), torch.zeros_like(self.w)
            sums.index_add_(0, nb[valid], lg[valid])
            counts.index_add_(0, nb[valid], torch.ones_like(lg[valid]))
            hit = counts > 0
            self.w[hit] *= (sums[hit] / counts[hit]).exp()
            touched = int(hit.sum())
            affected_count = (valid & (lg != 0)).sum(1)
        self.w.clamp_(0.05, 1000.0)
        self.last_seen[ids] = float(step)
        self.pending = None
        return {
            "G": g,
            "g_hat": gh,
            "affected_neighbors": affected_count,
            "G_bar": float(self.g_bar),
            "n_touched": touched,
            "search_candidates": int((self.available & self.embedding_valid).sum())
            if self.ready
            else 0,
            "cache_age_max": int(
                (step - self.embedding_step[self.embedding_valid]).max()
            )
            if self.embedding_valid.any()
            else 0,
            "info_ratio": float((gh > 1).float().mean()),
        }

    def load_state_dict(self, state):
        mapping = {
            "weights": "w",
            "available": "available",
            "last_seen": "last_seen",
            "G_bar": "g_bar",
            "embeddings": "embedding_cache",
            "embedding_valid": "embedding_valid",
            "embedding_step": "embedding_step",
            "rows": "rows",
            "row_valid": "row_valid",
            "neff": "neff",
            "indegree": "indeg",
            "psi": "psi",
            "nu": "nu",
        }
        for key, attr in mapping.items():
            value = state[key]
            setattr(
                self,
                attr,
                value.to(self.labels.device)
                if isinstance(value, torch.Tensor)
                else value,
            )
        self.generator.set_state(state["generator_state"].cpu())
        self.epoch = state["epoch"]
        self.ready = self.epoch > 0
        self.pending = None

    def state_dict(self):
        return {
            "weights": self.w,
            "available": self.available,
            "last_seen": self.last_seen,
            "G_bar": self.g_bar,
            "generator_state": self.generator.get_state(),
            "embeddings": self.embedding_cache,
            "embedding_valid": self.embedding_valid,
            "embedding_step": self.embedding_step,
            "rows": self.rows,
            "row_valid": self.row_valid,
            "neff": self.neff,
            "indegree": self.indeg,
            "psi": self.psi,
            "nu": self.nu,
            "epoch": self.epoch,
        }
