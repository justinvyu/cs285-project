import numpy as np

def get_random_batch(obs, acs, terminals, batch_size=128, T=8):
    """
    Parameters
    ----------
    obs : (m x obs dim)   m = dataset size
    acs : (m x action dim)
    terminals : (m x 1)

    Returns
    -------
    Starting states s_t : (batch_size x observation dim)
    T step action sequence a_t^T-1 = (a_t, a_t+1, ..., a_t+T-1) : (batch_size x T x action dim)
    Ending states s_t+T : (batch_size x observation dim)
    """
    m = len(obs)
    assert m >= T

    episode_boundaries = np.where(terminals == True)[0]
    if episode_boundaries.size == 0: # No terminal states, can just sample wherever
        random_start_idxs = np.random.randint(m - T + 1, size=batch_size).astype(int)
    else:
        end_idxs = np.random.randint(len(episode_boundaries), size=batch_size)
        ends = episode_boundaries[end_idxs]
        starts = np.array([
            (episode_boundaries[idx - 1] if idx > 0 else 0)
            for idx in end_idxs
        ])
        random_start_idxs = np.array([
            np.random.randint(start, end - T + 1)
            for start, end in zip(starts, ends)
        ])

    s_t = obs[random_start_idxs]
    action_sequences = np.array([acs[idx:idx + T] for idx in random_start_idxs])
    s_T = obs[random_start_idxs + T]
    return s_t, action_sequences, s_T

def get_shuffled_minibatches(obs, acs, terminals, batch_size=128, T=8):
    m = len(obs)
    assert m >= T

    episode_boundaries = np.where(terminals == True)[0]
    end_idxs = np.random.randint(len(episode_boundaries), size=batch_size)
    ends = episode_boundaries[end_idxs]
    starts = np.array([
        (episode_boundaries[idx - 1] if idx > 0 else 0)
        for idx in end_idxs
    ])

    start_idxs = np.concatenate([
        np.arange(start, end - T + 1)
        for start, end in zip(starts, ends)
    ])
    s_t = obs[start_idxs]
    action_sequences = np.array([acs[idx:idx + T] for idx in start_idxs])
    s_T = obs[start_idxs + T]

    perm = np.random.permutation(len(s_t))
    s_t = s_t[perm]
    action_sequences = action_sequences[perm]
    s_T = s_T[perm]

    batched_s_t = np.array_split(s_t, len(s_t) // batch_size)
    batched_action_sequences = np.array_split(action_sequences, len(action_sequences) // batch_size)
    batched_s_T = np.array_split(s_T, len(s_T) // batch_size)

    return batched_s_t, batched_action_sequences, batched_s_T