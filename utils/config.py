class AgentConfig(object):

    action_bound = 1
    action_dim = 2
    state_dim = 7

    alayer_1 = 100
    alayer_2 = 75
    clayer_1 = 100
    clayer_2 = 75

    gamma = 0.99
    alearning_rate = 1e-4
    clearning_rate = 1e-3
    seed = 667
    tau = 0.001

    batch_size = 64
    explore = 100000
    memory_size = 1000000
    num_episode = 30000
    start_learning = 64

    test_step = 10
    test_trial = 5

    mu = 0.0
    sigma = 0.2
    theta = 0.15

    sim_time = 2400
