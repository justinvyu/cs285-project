import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization.video import VideoSaveFunction

import multiworld

def experiment(**variant):
    from multiworld import register_all_envs
    register_all_envs()

    env_kwargs = variant['env_kwargs']
    env_name = env_kwargs.pop("env_name")
    eval_env = gym.make(env_name, **env_kwargs)
    expl_env = gym.make(env_name, **env_kwargs)

    observation_key = 'observation'
    replay_buffer = ObsDictReplayBuffer(
        env=eval_env,
        observation_key=observation_key,
        save_data_in_snapshot=True,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces[observation_key].low.size
    action_dim = eval_env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    eval_policy = MakeDeterministic(policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
    )
    expl_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )

    video_func = VideoSaveFunction(eval_env, variant)
    algorithm.post_train_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()

def run_sac_experiment(
        env_name="Point2DMazeEvalMedium-v0",
        reward_type="sparse",
        use_count_reward=False,
        debug=False,
    ):
    # print(wall_shape, reward_type, use_count_reward)
    variant = dict(
        algorithm='SAC',
        version='normal',
        env_kwargs=dict(
            env_name=env_name,
            # wall_shape=wall_shape,
            reward_type=reward_type,
            # use_count_reward=use_count_reward,
            use_dict_obs=False,
        ),
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=500,
            num_eval_steps_per_epoch=100,  # 2 * 250 = 2 eval episodes
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=100,
            use_env_count_bonus=use_count_reward,
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        logger_config=dict(
            snapshot_mode='gap_and_last',
            snapshot_gap=20,
        ),
        dump_video_kwargs=dict(
            imsize=128,
            save_video_period=20,
            image_format='HWC'
        )
    )
    exp_name = (
        f'sac-env={env_name}-count_bonus={use_count_reward}'
        + ("-debug" if debug else "")
    )
    run_experiment(
        experiment,
        exp_name=exp_name,
        mode='local',
        variant=variant,
        #exp_id=0,
        use_gpu=True,
        gpu_id=0,
    )

from doodad.easy_sweep.hyper_sweep import run_sweep_parallel
from experiments.jonathan.sac import run_sac_experiment

if __name__ == "__main__":
    sweep_args = {
        "env_name": [
            #"Point2DMazeEvalEasy-v0",
            # "Point2DMazeEvalMedium-v0",
            # "Point2DMazeEvalHard-v0"
            # "Point2DDoubleMazeEval-v0",
            # "Point2DDoubleMazeSingleGoalEval-v0",
            # "Point2DRooms-v0",
            "Point2DRoomsLarge-v0",
        ],
        "use_count_reward": [True, False],
        "debug": [False],
    }
    run_sweep_parallel(run_sac_experiment, sweep_args, repeat=2)
