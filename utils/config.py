class AgentConfig(object):

    action_bound = 10
    action_dim = 2
    state_dim = 7

    alayer_1 = 400
    alayer_2 = 300
    clayer_1 = 400
    clayer_2 = 300

    gamma = 0.99
    learning_rate = 0.00025
    tau = 0.001

    batch_size = 64
    memory_size = 100000
    num_episode = 10000
    start_learning = 10000

    mu = 0.0
    sigma = 0.15
    theta = 0.20

    port = 19999
    path = "/home/souphis/Magisterka/Simulation/"
    scene = "s_navigation_task_1"
    sim_time = 120000
    vrep = "/opt/V-REP/vrep.sh -h " + str(sim_time) + " -q"
    vrep_param = "-gREMOTEAPISERVERSERVICE_" + str(port) + "_FALSE_FALSE "
