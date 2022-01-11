from functools import partial

import numpy as np

create_rollout_function = partial


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(a.copy())
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )


# ==== START ====
def goal_generation_rollout(
        env,
        agent,
        goal_generator=None,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,

        # Goal conditioned reaching + exploration
        goal_reaching_policy=None,
        # Turn this on by default so that we get some online data + random exploration
        # Otherwise, the problem is fully offline training, which is tough.
        save_goal_reaching_rollout=True,
        max_goal_reaching_path_length=100,
        # If rand() < epsilon: do random expl
        random_exploration_epsilon=1,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    goal_reaching_path_length = 0
    agent.reset()
    o = env.reset()

    # TODO: use a more generic interface than this
    assert (
        hasattr(env, "set_position")
        and hasattr(env, "_get_obs")
        and hasattr(env, "get_target_position")
        and hasattr(env, "set_goal")
    )

    goal = goal_generator.get_goal()
    original_goal = env.get_target_position()
    if goal is not None:
        if goal_reaching_policy is None:
            # If a goal reaching policy doesn't exist
            # Generate a new goal, set that as the position (teleport)
            env.set_position(goal.squeeze())
            o = env._get_obs()
        else:
            env.set_goal(goal.squeeze())
            while goal_reaching_path_length < max_goal_reaching_path_length:
                if save_goal_reaching_rollout:
                    raw_obs.append(o)
                o_for_agent = preprocess_obs_for_policy_fn(o)

                # Use the goal-conditioned reaching policy to get to the waypoint
                # for further exploration
                a, agent_info = goal_reaching_policy.get_action(o_for_agent, **get_action_kwargs)

                if full_o_postprocess_func:
                    full_o_postprocess_func(env, agent, o)

                next_o, r, d, env_info = env.step(a.copy())
                if render:
                    env.render(**render_kwargs)

                # Add these transitions to the replay buffer
                if save_goal_reaching_rollout:
                    observations.append(o)
                    rewards.append(r)
                    terminals.append(d)
                    actions.append(a)
                    next_observations.append(next_o)
                    raw_next_obs.append(next_o)
                    agent_infos.append(agent_info)
                    env_infos.append(env_info)

                goal_reaching_path_length += 1
                if d:
                    break
                o = next_o
            env.set_goal(original_goal)

    rand = np.random.rand()
    if rand < random_exploration_epsilon:
        explore_agent = agent
    else:
        explore_agent = goal_reaching_policy

    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)

    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = explore_agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, explore_agent, o)

        next_o, r, d, env_info = env.step(a.copy())
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )
# ==== END ====

