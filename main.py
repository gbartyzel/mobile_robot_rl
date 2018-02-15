import os
import tensorflow as tf

from ddpg import DDPGAgent

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('train', True, 'enable training')
tf.app.flags.DEFINE_bool('test', False, 'enable test')
tf.app.flags.DEFINE_string('model_path', 'saved_ddpg', 'path for model')
tf.app.flags.DEFINE_string('summary_path', 'train_output', 'path for summary')

tf.app.flags.DEFINE_string('env_name', 'InvertedPendulum-v1', 'environment id')
tf.app.flags.DEFINE_integer('num_steps', 2400, 'env max steps')
tf.app.flags.DEFINE_integer('episodes', 1000, 'max episodes')
tf.app.flags.DEFINE_integer('x_goal', 2.0, 'goal x position')
tf.app.flags.DEFINE_integer('y_goal', 2.0, 'goal y position')

tf.app.flags.DEFINE_integer('test_episodes', 10, 'test episodes')
tf.app.flags.DEFINE_integer('test_trials', 5, 'number of test trials')

tf.app.flags.DEFINE_integer('warm_up', 10000, 'steps before training')
tf.app.flags.DEFINE_integer('exploration', 1000000, 'max exploration steps')
tf.app.flags.DEFINE_integer('memory_size', 1000000, 'max replay buffer size')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')

tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor')


def main(_):
    ddpg = DDPGAgent(FLAGS)
    if FLAGS.train:
        ddpg.train()
    else:
        ddpg.play()


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.app.run()
