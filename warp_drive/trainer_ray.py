from typing import List

from ray.tune.trial import Trial
from trainer_lightning import PerfStatsCallback
from ray.tune import Callback
import torch


class RayPerfStatsCallback(Callback, PerfStatsCallback):
    """
    Performance stats that will be included in rollout metrics.
    """

    def __init__(self, batch_size=None, num_iters=None, log_freq=1000):
        assert batch_size is not None
        assert num_iters is not None
        super().__init__(batch_size=batch_size, num_iters=num_iters, log_freq=log_freq)

    def on_step_begin(self, iteration: int, trials: List["Trial"], **info):
        self.iters = self.steps = iteration
        self.start_event_batch.record()

    def on_step_end(self, iteration: int, trials: List["Trial"], **info):
        self.end_event_batch.record()
        torch.cuda.synchronize()

        self.training_time += (
                self.start_event_batch.elapsed_time(self.end_event_batch) / 1000
        )

        if (self.iters + 1) % self.log_freq == 0 or self.iters == self.num_iters:
            self.pretty_print(self.get_perf_stats())
