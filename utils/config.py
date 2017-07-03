class AgentConfig(object):

    action_bound = 1
    action_dim = 2
    state_dim = 7

    alayer_1 = 400
    alayer_2 = 300
    clayer_1 = 400
    clayer_2 = 300

    gamma = 0.99
    alearning_rate = 1e-4
    clearning_rate = 1e-3
    seed = 1337
    tau = 0.001

    batch_size = 64
    memory_size = 1000000
    num_episode = 30000
    start_learning = 10000

    mu = 0.0
    sigma = 0.2
    theta = 0.15

    port = 19999
    path = "/home/souphis/Magisterka/Simulation/"
    scene = "s_navigation_task_0.ttt"
    sim_time = 120000
    vrep = "/home/souphis/Magisterka/V-REP-GUI/vrep.sh -s" + str(sim_time) + \
           " -q "
    vrep_param = "-gREMOTEAPISERVERSERVICE_" + str(port) + "_FALSE_FALSE "
