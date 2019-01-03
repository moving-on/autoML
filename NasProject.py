import tensorflow as tf
from keras import backend as K
from Controller import Controller
from ModelMananger import SpaceManager
from NetworkManager import NetworkManager
from ModelConstructor import ModelConstructor
import csv
import os


class NasProject(object):
    """
    A Neural Architecture Search Project
    """
    def __init__(self, name, dataset, input_size, output_size):
        """
        Initialize the NAS Project
        Only the name, data set and input/output size will be specified during the initialization.
        Other hypermeters will be initialized using the default value
        :param name: The name of this AutoML Project
        :param dataset: a tuple of [x_train, y_train, x_test, y_test]
        :param input_size: the input size of the neural network
        :param name: the output size of the neural network (i.e. number of classes)
        """
        # the parameters to be set during initialization
        self.name = name
        self.dataset = dataset
        self.input_size = input_size
        self.output_size = output_size
        self.state_space = SpaceManager()
        self.config_neural_space()
        self.config_training()

    def config_neural_space(self, num_conv_layers=4, num_dense_layers=0, dense_config=[]):
        """
        Define the search space of this AutoML task
        :param num_conv_layers:
        :param num_dense_layers:
        :param dense_config: [units0, dropout0, units1, dropout1, ... ]
        """
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.dense_config = dense_config

    def config_training(self, num_trials=200, num_epochs=1, batch_size=256, exploration=0.0,
                        regularization=1e-3, num_control_cells=32, embedding_dim=20,
                        accuracy_beta=0.8, clip_rewards=0, restore_model=False):
        """
        Define the parameters for the RNN controller that will be used to generate candidates
        :param num_trials: number of models that will be generated and evaluated
        :param num_epochs: number of epochs to train each candidate
        :param batch_size: input batch size when training each candidate
        :param exploration: high exploration for the first 1000 steps
        :param regularization: regularization strength
        :param num_control_cells: number of cells in RNN controller
        :param embedding_dim: dimension of the embeddings for each state
        :param accuracy_beta: beta value for the moving average of the accuracy
        :param clip_rewards: clip rewards in the [-0.05, 0.05] range
        :param restore_model: whether to restore controller from file to continue training
        """
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.exploration = exploration
        self.regularization = regularization
        self.num_control_cells = num_control_cells
        self.embedding_dim = embedding_dim
        self.accuracy_beta = accuracy_beta
        self.clip_rewards = clip_rewards
        self.restore_model = restore_model

    def add_seach_space(self, name, values):
        self.state_space.add_space(name, values)

    def print_search_space(self):
        self.state_space.print_state_space()

    def train(self):
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        previous_acc = 0.0
        total_reward = 0.0

        policy_sess = tf.Session()
        K.set_session(policy_sess)

        with policy_sess.as_default():
            # create the Controller and build the internal policy network
            controller = Controller(policy_sess, self.num_conv_layers, self.state_space,
                                    reg_param=self.regularization,
                                    exploration=self.exploration,
                                    controller_cells=self.num_control_cells,
                                    embedding_dim=self.embedding_dim,
                                    restore_controller=self.restore_model)

        # create the Network Manager
        manager = NetworkManager(self. dataset, epochs=self.num_epochs, child_batchsize=self.batch_size,
                                 clip_rewards=self.clip_rewards, acc_beta=self.accuracy_beta)

        # create the Model Constructor
        constructor = ModelConstructor(self.input_size, self.output_size, self.num_conv_layers,
                                       self.num_dense_layers, self.dense_config)

        # get an initial random state space if controller needs to predict an
        # action from the initial state
        state = self.state_space.init_random_states(self.num_conv_layers)
        print state
        print("Initial Random State : ", self.state_space.parse_state_space_list(state))
        print()

        # clear the previous files
        controller.remove_files()

        # train for number of trails
        for trial in range(self.num_trials):
            with policy_sess.as_default():
                K.set_session(policy_sess)
                actions = controller.get_action(state)  # get an action for the previous state

            # print the action probabilities
            self.state_space.print_actions(actions)
            print("Predicted actions: ", self.state_space.parse_state_space_list(actions, local=True))

            # build a model, train and get reward and accuracy from the network manager
            reward, previous_acc = manager.get_rewards(constructor, self.state_space.parse_state_space_list(actions, local=True))
            print("Rewards: ", reward, "Accuracy: ", previous_acc)

            with policy_sess.as_default():
                K.set_session(policy_sess)

                total_reward += reward
                print("Total reward: ", total_reward)

                # actions and states are equivalent, save the state and reward
                first_state = state[0]
                state = actions
                controller.store_rollout(state, reward)

                # train the controller on the saved state and the discounted rewards
                loss = controller.train_step(first_state)
                print("Trial %d: Controller loss: %0.6f" % (trial + 1, loss))

            print()

        print("Total Reward: ", total_reward)
