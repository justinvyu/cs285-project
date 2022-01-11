from abc import ABCMeta
import numpy as np
import gym
import collections
import pickle
import time
from rlkit.core.timer import timer

def exp_normalize(x, temperature=1):
    x = x / temperature
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

class GoalGenerator(metaclass=ABCMeta):
    def __init__(
            self,
            env,
            replay_buffer,
            verbose=False,
            batch_size=np.inf,
            observation_key="observation",
            **kwargs
    ):
        self._env = env
        self._replay_buffer = replay_buffer
        # Size of our batch to sample a goal from
        # If batch_size = inf, then sample the goal from the entire replay buffer
        self._batch_size = batch_size
        self._verbose = verbose

        self._goal_history_per_epoch = []
        self._cur_epoch_goal_history = []

    def get_goal(self):
        # Sample a batch from replay buffer and return that obs as the goal
        raise NotImplementedError

    def train(self, algorithm, epoch):
        self._goal_history_per_epoch.append(np.array(self._cur_epoch_goal_history))
        self._cur_epoch_goal_history = []

    def get_snapshot(self):
        return dict(
            goal_history_per_epoch=self._goal_history_per_epoch,
        )

    def get_diagnostics(self):
        return {}

class RandomGoalGenerator(GoalGenerator):
    def get_goal(self):
        if self._replay_buffer.num_steps_can_sample() == 0:
            return None

        goal = self._replay_buffer.random_batch(1)["observations"].squeeze()
        self._cur_epoch_goal_history.append(goal)
        return goal

class RewardGoalGenerator(GoalGenerator):
    def get_goal(self):
        if self._replay_buffer.num_steps_can_sample() == 0:
            return None

        if self._batch_size < np.inf:
            batch = self._replay_buffer.random_batch(self._batch_size)
            obs = batch["observations"]
            rew = batch["rewards"]
        else:
            obs = self._replay_buffer._obs['observation'][:self._replay_buffer._size]
            rew = self._replay_buffer._rewards[:self._replay_buffer._size]
        goal = obs[np.argmax(rew)]
        self._cur_epoch_goal_history.append(goal)
        return goal

class UCBGoalGenerator(GoalGenerator):
    # Oracle count goal generator that samples based on maximum novelty
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self._env, "get_count_bonuses"), (
            "Environment needs to implement `get_count_bonuses` for this goal generation scheme")

    def get_goal(self):
        if self._replay_buffer.num_steps_can_sample() == 0:
            # Return None if there's nothing in the replay buffer yet
            return None

        if self._batch_size < np.inf:
            batch = self._replay_buffer.random_batch(self._batch_size)["observations"]
        else:
            batch = self._replay_buffer._obs['observation'][:self._replay_buffer._size]

        count_bonuses = self._env.get_count_bonuses(batch)

        # Choose the goal with the maximum count bonus
        goal_idx = np.argmax(count_bonuses)
        goal = batch[goal_idx]

        self._cur_epoch_goal_history.append(goal)
        return goal

class OracleSkewFitGoalGenerator(GoalGenerator):
    # Oracle skew fit goal generator that samples based on state density
    def __init__(self, *args, alpha=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        assert hasattr(self._env, "get_bin_counts"), (
            "Environment needs to implement `get_bin_counts` for this goal generation scheme")

    def get_goal(self):
        if self._replay_buffer.num_steps_can_sample() == 0:
            # Return None if there's nothing in the replay buffer yet
            return None

        if self._batch_size < np.inf:
            batch = self._replay_buffer.random_batch(self._batch_size)["observations"]
        else:
            batch = self._replay_buffer._obs['observation'][:self._replay_buffer._size]

        # Use skewed goal distribution (correpsonds to alpha = -1 in SkewFit)
        bin_counts = self._env.get_bin_counts(batch)
        skewed_goal_densities = np.power(bin_counts, self.alpha)
        # Normalize to a proper probability distribution
        skewed_goal_probs = skewed_goal_densities / np.sum(skewed_goal_densities)

        # Sample a goal from the skewed distribution
        goal_idx = np.random.choice(np.arange(batch.shape[0]), p=skewed_goal_probs)
        goal = batch[goal_idx]
        self._cur_epoch_goal_history.append(goal)
        return goal

class RNDGoalGenerator(GoalGenerator):
    def get_goal(self):
        pass

class ValueGoalGenerator(GoalGenerator):
    def get_goal(self):
        pass

from rlkit.torch.empowerment.gaussian_channel import GaussianChannelModel
from rlkit.torch.empowerment.dataset import get_random_batch, get_shuffled_minibatches
from rlkit.torch.pytorch_util import get_global_device

class EmpowermentGoalGenerator(GoalGenerator):
    def __init__(
            self,
            *args,
            N=8,
            model_pkl_path=None,
            extract_fn=(lambda x: x),
            cycle_dataset=True,
            num_train_steps_per_iter=1000,
            batch_size=128,
            training_starts_epoch=5,
            train_every_n_epochs=1,
            train_online=True,
            sample_goal=True,  # Whether or not to take the argmax vs sample
            # Whether or not to generate a Skewfit batch to account for data distribution imbalance
            skewfit_candidate_sampling=False,
            skewfit_alpha=-1,
            skewfit_sample_size=500,
            # Factor to scale up normalized empowerment to draw a bigger distinction
            # between low/high empowerment points with very close values
            # Larger value = a harder max
            softmax_temperature=1,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if isinstance(self._env.observation_space, gym.spaces.Dict):
            obs_dim = self._env.observation_space["observation"].low.size
        else:
            obs_dim = self._env.observation_space.low.size
        ac_dim = self._env.action_space.low.size

        if model_pkl_path is not None:
            with open(model_pkl_path, "rb") as f:
                self.gc_model = extract_fn(pickle.load(f))
        else:
            self.gc_model = GaussianChannelModel(obs_dim, ac_dim, N, **kwargs)

        self.gc_model.to(get_global_device())

        self._cycle_dataset = True
        self._num_train_steps_per_iter = num_train_steps_per_iter
        self._batch_size = batch_size
        self._training_starts_epoch = training_starts_epoch
        self._train_every_n_epochs = train_every_n_epochs
        self._training_started = False
        self._loss_history = collections.deque([], maxlen=int(1e6))
        self._train_online = train_online
        self._sample_goal = sample_goal
        self._skewfit_candidate_sampling = skewfit_candidate_sampling
        self._skewfit_alpha = skewfit_alpha
        self._skewfit_sample_size = skewfit_sample_size

        self._softmax_temperature = softmax_temperature

        # Diagnostics
        self._gc_model_loss = 0
        self._total_grad_steps = 0
        # self._query_time = 0
        self._iter_training_time = 0

    def train(self, algorithm, epoch):
        super().train(algorithm, epoch) # This will update the history of generated goals

        start = time.time()
        # If we don't train online, then we assume a loaded, frozen model that was trained offline
        # Wait until enough epochs have passed, as well as training every n epochs
        if (self._train_online
                and epoch >= self._training_starts_epoch
                and (epoch - self._training_starts_epoch) % self._train_every_n_epochs == 0):

            self._training_started = True
            assert hasattr(algorithm, "max_path_length"), "Need to specify a max path length in the algorithm"

            # Convert back to GPU when training
            self.gc_model.to(get_global_device())
            if self._cycle_dataset:
                end_loss = self.train_cycle(self._num_train_steps_per_iter, algorithm.max_path_length)
            else:
                end_loss = self.train_sample(self._num_train_steps_per_iter, algorithm.max_path_length)

            self._gc_model_loss = end_loss
            self._total_grad_steps += self._num_train_steps_per_iter
            # self._query_time = 0
        self._iter_training_time = time.time() - start

    def train_cycle(self, num_steps, path_length):
        TRAIN_LOG_FREQ = 200

        obs = self._replay_buffer._obs['observation'][:self._replay_buffer._size]
        acs = self._replay_buffer._actions[:self._replay_buffer._size]
        # terminals = self._replay_buffer._terminals[:self._replay_buffer._size]
        # NOTE: This is manually replacing terminals with the episode boundaries
        # to prevent sampling across resets
        terminals = np.zeros(obs.shape[0])
        terminals[np.arange(path_length - 1, len(terminals), path_length)] = 1

        grad_steps = 0
        i = 0

        batches_s_t, batches_action_seqs, batches_s_T = get_shuffled_minibatches(
            obs, acs, terminals, batch_size=128, T=self.gc_model.N)
        assert len(batches_s_t) == len(batches_action_seqs) == len(batches_s_T)
        num_minibatches = len(batches_s_t)

        while grad_steps < num_steps:
            if i == num_minibatches:
                i = 0
                batches_s_t, batches_action_seqs, batches_s_T = get_shuffled_minibatches(
                    obs, acs, terminals, batch_size=self._batch_size, T=self.gc_model.N)
                assert len(batches_s_t) == len(batches_action_seqs) == len(batches_s_T)
                num_minibatches = len(batches_s_t)

            batch_s_t = batches_s_t[i]
            batch_action_seqs = batches_action_seqs[i]
            batch_s_T = batches_s_T[i]

            loss = self.gc_model.update(batch_s_t, batch_action_seqs, batch_s_T)
            self._loss_history.append(loss)
            if self._verbose and grad_steps % TRAIN_LOG_FREQ == 0:
                print(f"\n============== Training Step #{grad_steps} ==============")
                print("MSE Loss =", loss)
            grad_steps += 1
            i += 1

        return loss

    def train_sample(self, num_steps, path_length):
        TRAIN_LOG_FREQ = 200

        obs = self._replay_buffer._obs['observation'][:self._replay_buffer._size]
        acs = self._replay_buffer._actions[:self._replay_buffer._size]
        # terminals = self._replay_buffer._terminals[:self._replay_buffer._size]
        # NOTE: This is manually replacing terminals with the episode boundaries
        # to prevent sampling across resets
        terminals = np.zeros(obs.shape[0])
        terminals[np.arange(path_length - 1, len(terminals), path_length)] = 1

        for i in range(num_steps):
            batch_s_t, batch_action_seqs, batch_s_T = get_random_batch(
                obs, acs, terminals, batch_size=self._batch_size, T=self.gc_model.N)
            loss = self.gc_model.update(batch_s_t, batch_action_seqs, batch_s_T)
            self._loss_history.append(loss)
            if self._verbose and i % TRAIN_LOG_FREQ == 0:
                print(f"\n============== Training Step #{i} ==============")
                print("MSE Loss =", loss)

        return loss

    def get_goal(self):
        if self._replay_buffer.num_steps_can_sample() == 0 or not self._training_started:
            # Return None if there's nothing in the replay buffer yet,
            # or if the empowerment estimator has not been trained at all
            return None

        if self._batch_size < np.inf:
            batch = self._replay_buffer.random_batch(self._batch_size)
            obs = batch["observations"]
            # obs_indices = batch["indices"]
        else:
            obs = self._replay_buffer._obs['observation'][:self._replay_buffer._size]
            # obs_indices = np.arange(self._replay_buffer._size)

        if self._sample_goal:
            if self._skewfit_candidate_sampling:
                # Sample candidates based on skewfit probs
                bin_counts = self._env.get_bin_counts(obs)
                skewed_goal_densities = np.power(bin_counts, self._skewfit_alpha)
                skewed_goal_probs = skewed_goal_densities / np.sum(skewed_goal_densities)
                candidate_idxs = np.random.choice(
                    np.arange(obs.shape[0]), p=skewed_goal_probs, size=self._skewfit_sample_size)
                candidates = obs[candidate_idxs]

                # Of the candidates, sample one based on empowerment value
                empowerment_probs = self.get_empowerment_probs(candidates)
                goal_idx = np.random.choice(np.arange(candidates.shape[0]), p=empowerment_probs)
                goal = candidates[goal_idx]
            else:
                empowerment_probs = self.get_empowerment_probs(obs)
                goal_idx = np.random.choice(np.arange(obs.shape[0]), p=empowerment_probs)
                goal = obs[goal_idx]
        else:
            start = time.time()
            self.gc_model.to("cpu")  # converting to CPU makes things faster
            empowerment_vals = self.gc_model.empowerment(obs)
            elapsed = time.time() - start
            self._query_time += elapsed
            goal = obs[np.argmax(empowerment_vals)]

        self._cur_epoch_goal_history.append(goal)
        return goal

    def get_empowerment_probs(self, obs):
        # start = time.time()
        timer.start_timer("gc_model query time", unique=False)
        self.gc_model.to("cpu")  # converting to CPU makes SVD calculation faster
        empowerment_vals = self.gc_model.empowerment(obs)
        # elapsed = time.time() - start
        # self._query_time += elapsed
        timer.stop_timer("gc_model query time")

        # empowerment_probs = empowerment_vals / empowerment_vals.sum()
        if empowerment_vals.std() > 1e-4:
            normalized_empowerment = (empowerment_vals - empowerment_vals.mean()) / empowerment_vals.std()
        else:
            normalized_empowerment = empowerment_vals
        empowerment_probs = exp_normalize(
            normalized_empowerment, temperature=self._softmax_temperature)
        return empowerment_probs

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            gaussian_channel_model=self.gc_model,
            training_loss_history=self._loss_history,
        )
        return snapshot

    def get_diagnostics(self):
        return {
            'gc_model/loss': self._gc_model_loss,
            'gc_model/total_grad_steps': self._total_grad_steps,
            'gc_model/training': self._iter_training_time,
        }

