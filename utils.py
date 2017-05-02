import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MultiNEAT as NEAT


class Utility(object):
    def __init__(self, config_file, task_name, tool_name):

        with open(config_file, 'r') as file:
            self.cmd_lines = file.readlines()

        self.path = self.cmd_lines[6]
        self.path = self.path.strip()
        print(self.path)
        self.task_name = task_name
        self.tool_name = tool_name

    @staticmethod
    def run_vrep(port, scene):
        i = str(scene)
        cmd_2 = "/home/souphis/Magisterka/V-REP/vrep.sh -h -s180000 -q "
        cmd_3 = "-gREMOTEAPISERVERSERVICE_" + str(port) + "_FALSE_FALSE "
        cmd_4 = "/home/souphis/Magisterka/Simulation/s_navigation_task_" + i
        os.system(cmd_2 + cmd_3 + cmd_4 + ".ttt &")

    def plotting(self, data):
        data = data[0:2]
        current_data = datetime.now().strftime("%Y_%m_%d")
        sns.set()
        sns.set_style('whitegrid')
        sns.set_context(
            "notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        plt.plot(data[0], data[1])

        if self.tool_name == 'NEAT':
            plt.xlabel('Generation')
            plt.ylabel('Best fitness')

            if self.task_name == 'avoidance':
                plot_filename = 'a_neat_best_genome_' + current_data + '.jpg'
            elif self.task_name == 'navigation':
                plot_filename = 'n_neat_best_genome_' + current_data + '.jpg'
            else:
                plot_filename = 'neat_best_genome_' + current_data + '.jpg'

        if self.tool_name == 'HyperNEAT':
            plt.xlabel('Generation')
            plt.ylabel('Best fitness')

            if self.task_name == 'avoidance':
                plot_filename = 'a_hneat_best_genome_' + current_data + '.jpg'
            elif self.task_name == 'navigation':
                plot_filename = 'n_hneat_best_genome_' + current_data + '.jpg'
            else:
                plot_filename = 'hneat_best_genome_' + current_data + '.jpg'

        plt.savefig(self.path + plot_filename, bbox_inches='tight')

        plt.show()

        print('Plot saved.')

    def saving(self, genome, data, save_data=True):
        current_data = datetime.now().strftime("%Y_%m_%d")

        net = NEAT.NeuralNetwork()
        if self.tool_name == 'NEAT':
            genome.BuildPhenotype(net)
            if self.task_name == 'avoidance':
                nn_filename = 'a_neat_best_genome_' + current_data + '.ne'
            if self.task_name == 'navigation':
                nn_filename = 'n_neat_best_genome_' + current_data + '.ne'
        if self.tool_name == 'HyperNEAT':
            genome.BuildPhenotype(net)
            if self.task_name == 'avoidance':
                nn_filename = 'a_hneat_best_genome_' + current_data + '.ne'
            if self.task_name == 'navigation':
                nn_filename = 'n_hneat_best_genome_' + current_data + '.ne'

        net.Save(self.path + nn_filename)
        print('Genome saved.')

        if save_data:
            data_filename = 'neat_data_' + current_data + '.csv'

            np.savetxt(self.path + data_filename, data.T, delimiter=',')

            print('Data saved.')
