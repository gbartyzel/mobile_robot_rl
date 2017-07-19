class AgentConfig(object):

    action_bound = 1
    action_dim = 2
    state_dim = 7

    alayer_1 = 64
    alayer_2 = 48
    clayer_1 = 64
    clayer_2 = 48

    gamma = 0.99
    alearning_rate = 1e-2
    clearning_rate = 2.5e-3
    seed = 667
    tau = 0.001

    batch_size = 64
    explore = 200000
    memory_size = 1000000
    num_episode = 30000
    start_learning = 10000

    test_step = 100
    test_trial = 10

    mu = 0.0
    sigma = 0.20
    theta = 0.15

    sim_time = 2400
