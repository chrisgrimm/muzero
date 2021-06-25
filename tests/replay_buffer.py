import numpy as np

from replay_buffers import trajectory_replay_buffer as trb

def _set_up_buffer_no_priority():
    specs = [
        trb.ReplaySpec('obs', shape=(), dtype=np.int32, on_reset=True, oob_handling=trb.oob_identity),
        trb.ReplaySpec('a', shape=(), dtype=np.uint8, on_reset=False, oob_handling=trb.make_oob_random_action(2)),
        trb.ReplaySpec('r', shape=(), dtype=np.float32, on_reset=False, oob_handling=trb.oob_zero),
        trb.ReplaySpec('done', shape=(), dtype=np.bool, on_reset=False, oob_handling=trb.oob_only_reset_at_end_of_traj)
    ]
    buffer = trb.TrajectoryReplayBuffer(10, specs, use_priority=False)
    i = 0
    buffer.reset(obs=i)
    i += 1
    while i < 35:
        done = np.random.uniform() < 0.2
        buffer.step(obs=i, a=0, r=i, done=done)
        i += 1
        if done:
            buffer.reset(obs=i)
            i += 1
    return buffer


def _set_up_buffer_priority():
    specs = [
        trb.ReplaySpec('obs', shape=(), dtype=np.int32, on_reset=True, oob_handling=trb.oob_identity),
        trb.ReplaySpec('a', shape=(), dtype=np.uint8, on_reset=False, oob_handling=trb.make_oob_random_action(2)),
        trb.ReplaySpec('r', shape=(), dtype=np.float32, on_reset=False, oob_handling=trb.oob_zero),
        trb.ReplaySpec('done', shape=(), dtype=np.bool, on_reset=False, oob_handling=trb.oob_only_reset_at_end_of_traj)
    ]
    buffer = trb.TrajectoryReplayBuffer(10, specs, use_priority=True)
    buffer.reset(obs=0, priority=5)
    buffer.step(obs=1, a=0, r=1, done=False, priority=1)
    buffer.step(obs=2, a=0, r=2, done=False, priority=1)
    buffer.step(obs=3, a=0, r=3, done=False, priority=1)
    buffer.step(obs=4, a=0, r=4, done=False, priority=1)
    buffer.step(obs=5, a=0, r=5, done=False, priority=1)
    buffer.step(obs=6, a=0, r=6, done=False, priority=0)
    buffer.step(obs=7, a=0, r=7, done=False, priority=0)
    buffer.step(obs=8, a=0, r=8, done=False, priority=0)
    buffer.step(obs=9, a=0, r=9, done=False, priority=0)
    return buffer


def test_replay_buffer_indexing(repeat=100):
    for _ in range(repeat):
        buffer = _set_up_buffer_no_priority()
        random_before, random_after = np.random.randint(1, 2, size=(2,))
        num_samples = 10
        traj = buffer.sample_traj(num_samples, (-random_before, random_after))
        # trajectories should be length: (random_before + 1 + random_after) for "on_reset" quantities
        assert len(traj['obs'][1]) == random_before + 1 + random_after
        for batch_idx, buffer_idx in enumerate(traj['indices']):
            # assert  that the batches are "centered" correctly.
            assert traj['obs'][batch_idx][random_before] == buffer._buffers['obs'][buffer_idx]
            # make sure that all the indices from before the "center" are < and after >.
            assert np.all(traj['obs'][batch_idx][random_before+1:] >= traj['obs'][batch_idx][random_before])
            assert np.all(traj['obs'][batch_idx][:random_before] <= traj['obs'][batch_idx][random_before])

def test_replay_buffer_priority():
    buffer = _set_up_buffer_priority()
    num_samples = 10_000
    from replay_buffers import sum_tree
    i = 0
    # while True:
    #     i += 1
    #     x = sum_tree.sample(buffer._priority_sampler)
    #     if x == 6:
    #         print('Impossible!')
    #         raise Exception()
    #     print(i)
    traj = buffer.sample_traj(num_samples, (-1, 1))
    center_obs = traj['obs'][:, 1]
    dist = np.zeros((10,), dtype=np.float32)
    for obs in center_obs:
        dist[obs] += 1
    dist /= num_samples
    # zero priority elements cannot be sampled
    assert np.sum(dist[6:10]) == 0
    # the priority 5 element should appear 50% of the time
    assert np.abs(dist[0] - 0.5) < 0.05

    # testing that rolling the buffer over correctly updates the priority
    buffer.step(obs=0, a=0, r=0, done=False, priority=10)

    traj = buffer.sample_traj(num_samples, (-1, 1))
    center_obs = traj['obs'][:, 1]
    dist = np.zeros((10,), dtype=np.float32)
    for obs in center_obs:
        dist[obs] += 1
    dist /= num_samples
    # 6:10 should still be 0 prob
    assert np.sum(dist[6:10]) == 0
    # 0 should be updated to be 2/3.
    assert np.abs(dist[0] - 0.66) < 0.05


def print_traj_nodes(buffer: trb.TrajectoryReplayBuffer):
    node = buffer._traj_node
    values = []
    for _ in range(10):
        if node.value in values:
            break
        values.append(node.value)
        node = node.next
    print(values)


buffer = _set_up_buffer_no_priority()
test_replay_buffer_indexing()
test_replay_buffer_priority()
