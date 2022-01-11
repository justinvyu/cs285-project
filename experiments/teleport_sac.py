import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import ObsDictPathCollector, GoalGenerationObsDictPathCollector
from rlkit.goal_generation.goal_generator import (
    UCBGoalGenerator, RandomGoalGenerator, RewardGoalGenerator,
    OracleSkewFitGoalGenerator, RNDGoalGenerator, ValueGoalGenerator,
    EmpowermentGoalGenerator
)
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization.video import VideoSaveFunction
from rlkit.policies.simple import RandomPolicy

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
    # Evaluation environment
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
    )
    # Training environment
    goal_generator_kwargs = variant["goal_generator_kwargs"]
    goal_generator_type = goal_generator_kwargs.pop("type", "UCBGoalGenerator")
    if goal_generator_type == "UCBGoalGenerator":
        goal_generator = UCBGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "RandomGoalGenerator":
        goal_generator = RandomGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "RewardGoalGenerator":
        goal_generator = RewardGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "OracleSkewFitGoalGenerator":
        goal_generator = OracleSkewFitGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "RNDGoalGenerator":
        goal_generator = RNDGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "ValueGoalGenerator":
        goal_generator = ValueGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    elif goal_generator_type == "EmpowermentGoalGenerator":
        goal_generator = EmpowermentGoalGenerator(expl_env, replay_buffer, **goal_generator_kwargs)
    else:
        raise NotImplementedError

    expl_path_collector = GoalGenerationObsDictPathCollector(
        goal_generator,
        expl_env,
        RandomPolicy(expl_env.action_space),  # Use a random policy for data collection
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

    # Add the goal generator's post epoch train function
    algorithm.post_train_funcs.append(goal_generator.train)

    # TODO: This affects the counts through saving videos, might want to change
    video_func = VideoSaveFunction(eval_env, variant, expl_path_collector=expl_path_collector)
    algorithm.post_train_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()

def run_sac_experiment(
        env_name="Point2DMazeEvalMedium-v0",
        max_path_length=100,
        reward_type="sparse",
        use_count_reward=False,
        goal_generator_type="UCBGoalGenerator",
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
            use_count_reward=use_count_reward,
            use_dict_obs=False,
        ),
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=500,
            num_eval_steps_per_epoch=max_path_length,  # 1 * 100 = 1 eval episodes
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=2000,
            min_num_steps_before_training=2000,
            max_path_length=max_path_length,
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
        goal_generator_kwargs=dict(
            type=goal_generator_type,
            verbose=True,
        ),
        logger_config=dict(
            snapshot_mode='gap_and_last',
            snapshot_gap=10
        ),
        dump_video_kwargs=dict(
            imsize=128,
            save_video_period=20,
            image_format='HWC'
        )
    )
    exp_name = (
        f"teleport-SAC-env={env_name}-goal_generator={goal_generator_type}"
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

if __name__ == "__main__":
    # run_sac_experiment(env_name='Point2DDoubleMazeSingleGoalEval-v0')
    # run_sac_experiment(env_name="Point2DMazeEvalMedium-v0", goal_generator_type="RandomGoalGenerator")
    # run_sac_experiment(env_name="Point2DMazeEvalMedium-v0", goal_generator_type="EmpowermentGoalGenerator")
    # run_sac_experiment(env_name="Point2DMazeEvalMedium-v0", goal_generator_type="OracleSkewFitGoalGenerator")
    # run_sac_experiment(env_name="Point2DMazeEvalMedium-v0", goal_generator_type="RewardGoalGenerator")

    sweep_args = {
        "env_name": [
            # "Point2DMazeEvalEasy-v0",
            # "Point2DMazeEvalMedium-v0",
            # "Point2DMazeEvalHard-v0"
            # "Point2DDoubleMazeEval-v0",
            # "Point2DDoubleMazeSingleGoalEval-v0",
            # "Point2DRooms-v0",
            "Point2DRoomsLarge-v0",
        ],
        "max_path_length": [
            # 100,
            200,
        ],
        "goal_generator_type": [
            # "UCBGoalGenerator",
            "RandomGoalGenerator",
            # "RewardGoalGenerator",
            "OracleSkewFitGoalGenerator",
            "EmpowermentGoalGenerator",
        ],
        "debug": [False],
    }
    run_sweep_parallel(run_sac_experiment, sweep_args, repeat=2)
