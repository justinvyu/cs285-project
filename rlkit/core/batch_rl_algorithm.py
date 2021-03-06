from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.misc import eval_util
from rlkit.samplers.data_collector.path_collector import PathCollector
from rlkit.core.rl_algorithm import BaseRLAlgorithm

class BatchRLAlgorithm(BaseRLAlgorithm):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            use_env_count_bonus=False,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self._use_env_count_bonus = use_env_count_bonus

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )

            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                timer.start_timer('exploration sampling', unique=False)
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                timer.stop_timer('exploration sampling')

                timer.start_timer('replay buffer data storing', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                timer.stop_timer('replay buffer data storing')

                timer.start_timer('training', unique=False)
                for j in range(self.num_trains_per_train_loop):
                    timer.start_timer('replay buffer sampling (training)', unique=False)
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    timer.stop_timer('replay buffer sampling (training)')
                    # ==== START ====
                    if self._use_env_count_bonus:
                        timer.start_timer('count bonus (training)', unique=False)
                        assert hasattr(self.expl_env, "get_count_bonuses")
                        obs = train_data["observations"]
                        count_bonuses = self.expl_env.get_count_bonuses(obs)
                        train_data["rewards"] += count_bonuses.reshape(train_data["rewards"].shape)
                        timer.stop_timer('count bonus (training)')
                    # ==== END ====
                    self.trainer.train(train_data)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False
