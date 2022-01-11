import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector, GoalGenerationGoalConditionedPathCollector)
from rlkit.goal_generation.goal_generator import (
    UCBGoalGenerator, RandomGoalGenerator, RewardGoalGenerator,
    OracleSkewFitGoalGenerator, RNDGoalGenerator, ValueGoalGenerator,
    EmpowermentGoalGenerator
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.policies.cql_policy import TanhGaussianPolicyCQL, MakeDeterministicCQL
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization.video import VideoSaveFunction
from rlkit.policies.simple import RandomPolicy
from rlkit.misc.schedule import *


import multiworld

def experiment(**variant):
    from multiworld import register_all_envs
    register_all_envs()

    trainer_type = variant.get("trainer_type", "SAC")

    env_kwargs = variant['env_kwargs']
    env_name = env_kwargs.pop("env_name")
    eval_env = gym.make(env_name, **env_kwargs)
    expl_env = gym.make(env_name, **env_kwargs)

    observation_key = 'observation'
    desired_goal_key = 'state_desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    if trainer_type == "SAC":
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim + goal_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
        )
        eval_policy = MakeDeterministic(policy)
    elif trainer_type == "CQL":
        policy = TanhGaussianPolicyCQL(
            obs_dim=obs_dim + goal_dim,
            action_dim=action_dim,
            **variant['policy_kwargs']
        )
        eval_policy = MakeDeterministicCQL(policy)

    if trainer_type == "SAC":
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['sac_trainer_kwargs']
        )
    elif trainer_type == "CQL":
        trainer = CQLTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['cql_trainer_kwargs']
        )
    trainer = HERTrainer(trainer)

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

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

    expl_path_collector = GoalGenerationGoalConditionedPathCollector(
        goal_generator,
        expl_env,
        RandomPolicy(expl_env.action_space),
        save_goal_reaching_rollout=True,
        goal_reaching_policy=policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant.get("expl_path_collector_kwargs", {})
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

    video_func = VideoSaveFunction(eval_env, variant)
    algorithm.post_train_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()


def run_goal_conditioned_explore_experiment(
        trainer_type="SAC",
        env_name="Point2DMazeEvalMedium-v0",
        reward_type="sparse",
        use_count_reward=False,
        goal_generator_type="UCBGoalGenerator",
        max_path_length=100,

        skewfit_candidate_sampling=True,
        skewfit_sample_size=500,
        skewfit_alpha=-1,
        softmax_temperature=0.3,
        epsilon_schedule_type="constant_1",
        debug=False,
):
    type_to_schedule = {
        "constant_1": ConstantSchedule(1),
        "constant_0.5": ConstantSchedule(0.5),
        "constant_0": ConstantSchedule(0),

        "linear_500000_0_1": LinearSchedule(500000, 0, 1),
        "linear_250000_0_1": LinearSchedule(250000, 0, 1),
        "linear_100000_0.1_1": LinearSchedule(100000, 0.1, 1),
        "linear_50000_0.1_1": LinearSchedule(50000, 0.1, 1),
    }
    epsilon_schedule = type_to_schedule.get(epsilon_schedule_type, None)
    if epsilon_schedule is None:
        raise NotImplementedError

    variant = dict(
        algorithm=f'HER-{trainer_type}',
        version='normal',
        env_kwargs=dict(
            env_name=env_name,
            reward_type=reward_type,
            # use_count_reward=use_count_reward,
            use_dict_obs=False,
        ),
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=500,
            num_eval_steps_per_epoch=max_path_length,
            num_expl_steps_per_train_loop=2000, # Make this 2000?
            num_trains_per_train_loop=2000,     # Make this 2000
            min_num_steps_before_training=2000, # Make this 2000
            max_path_length=max_path_length,
            use_env_count_bonus=use_count_reward,
        ),
        trainer_type=trainer_type,  # Corresponding trainer params added below
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
            save_data_in_snapshot=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        goal_generator_kwargs=dict(
            type=goal_generator_type,

            l2_lambda=5e-5,
            skewfit_candidate_sampling=skewfit_candidate_sampling,
            skewfit_sample_size=skewfit_sample_size,
            skewfit_alpha=skewfit_alpha,
            softmax_temperature=softmax_temperature,
        ),
        expl_path_collector_kwargs=dict(
            random_exploration_epsilon_schedule=epsilon_schedule,
            max_goal_reaching_path_length=max_path_length,
        ),
        logger_config=dict(
            snapshot_mode='gap_and_last',
            snapshot_gap=10
        ),
        dump_video_kwargs=dict(
            imsize=256,
            save_video_period=10,
            image_format='HWC'
        )
    )

    if trainer_type == "SAC":
        variant.update(
            sac_trainer_kwargs=dict(
                discount=0.99,
                soft_target_tau=5e-3,
                target_update_period=1,
                policy_lr=3E-4,
                qf_lr=3E-4,
                reward_scale=1,
                use_automatic_entropy_tuning=True,
            ),
        )
    elif trainer_type == "CQL":
        variant.update(
            cql_trainer_kwargs=dict(
                discount=0.99,
                soft_target_tau=5e-3,
                policy_lr=1e-4,
                qf_lr=3e-4,
                reward_scale=1,
                use_automatic_entropy_tuning=True,

                # Target nets/ policy vs Q-function update
                policy_eval_start=0,
                num_qs=2,

                # min Q
                temp=1.0,
                min_q_version=3,
                min_q_weight=5.0,

                # lagrange
                with_lagrange=False,  # Defaults to False
                lagrange_thresh=5.0,

                # extra params
                num_random=1,
                max_q_backup=False,
                deterministic_backup=False,
            ),
        )
    else:
        raise NotImplementedError

    exp_name = (
        f"gc_explore_{trainer_type}-env={env_name}-goal_generator={goal_generator_type}"
        f"-schedule={epsilon_schedule_type}"
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
    sweep_args = {
        "trainer_type": [
            "SAC",
            # "CQL"
        ],
        "env_name": [
            # "Point2DMazeEvalEasy-v0",
            # "Point2DMazeEvalMedium-v0",
            "Point2DMazeEvalHard-v0",
            # "Point2DDoubleMazeEval-v0",
            # "Point2DDoubleMazeSingleGoalEval-v0",
            # "Point2DRooms-v0",
            # "Point2DRoomsLarge-v0",
            # "Point2DRoomsExplore-v0",  # unsupervised exploration version
        ],
        "max_path_length": [
            100,
            # 200
        ],
        "goal_generator_type": [
            # "UCBGoalGenerator",
            "RandomGoalGenerator",
            # "RewardGoalGenerator",
            "OracleSkewFitGoalGenerator",
            "EmpowermentGoalGenerator",
        ],
        "epsilon_schedule_type": [
            # "constant_1",
            # "constant_0.5",
            # "constant_0",
            # "linear_500000_0_1",
            # "linear_250000_0_1",
            # "linear_100000_0.1_1",
            "linear_50000_0.1_1",
        ],
        "softmax_temperature": [
            # 0.3, 0.5
            0.5,
        ],
        "skewfit_alpha": [
            -1,
        ],
        "skewfit_sample_size": [
            500,
        ],
        "skewfit_candidate_sampling": [True],
        "debug": [False],
    }
    run_sweep_parallel(run_goal_conditioned_explore_experiment, sweep_args, repeat=2)