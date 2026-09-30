import gzip
import json
import multiprocessing as mp


def _writer(queue, step_path, sample_path):
    with (
        open(step_path, "x") as steps,
        gzip.open(sample_path, "xt", compresslevel=3) as samples,
    ):
        while True:
            message = queue.get()
            if message is None:
                break
            kind, payload = message
            if kind == "step":
                steps.write(json.dumps(payload, separators=(",", ":")) + "\n")
            elif kind == "samples":
                samples.write(json.dumps(payload, separators=(",", ":")) + "\n")
            elif kind == "flush":
                steps.flush()
                samples.flush()
            else:
                raise ValueError(kind)


class AsyncStepLogger:
    def __init__(self, directory, queue_size=256):
        context = mp.get_context("spawn")
        self.queue = context.Queue(maxsize=queue_size)
        self.process = context.Process(
            target=_writer,
            args=(
                self.queue,
                directory / "step_metrics.jsonl",
                directory / "step_samples.jsonl.gz",
            ),
        )
        self.process.start()

    def write(self, step, samples):
        self.queue.put(("step", step))
        self.queue.put(("samples", samples))

    def flush(self):
        self.queue.put(("flush", None))

    def close(self):
        self.queue.put(None)
        self.process.join()
        self.queue.close()
        if self.process.exitcode:
            raise RuntimeError(f"Logger exited with code {self.process.exitcode}")
