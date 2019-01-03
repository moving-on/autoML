import numpy as np


class Space(object):
    """
    A search space in the SpaceManager
    It refers to one dimension of the NAS (num_filers, filer_size, etc)
    """
    def __init__(self, state_id, state_dim, state_name, state_values):
        self.id = state_id
        self.name = state_name
        self.values = state_values
        self.size = len(state_values)

        self.index_map = {}
        self.value_map = {}
        for i, val in enumerate(state_values):
            self.index_map[i] = val
            self.value_map[val] = i
        self.g_index_map = {}
        self.g_value_map = {}
        for i, val in enumerate(state_values):
            self.g_index_map[state_dim + i] = val
            self.g_value_map[val] = state_dim + i

    def onehot_encoding(self, value, space_dim):
        onehot = np.zeros((1, space_dim), dtype=np.float32)
        value_idx = self.g_value_map[value]
        onehot[0, value_idx] = onehot[0, value_idx] + 1
        return onehot

    def local_onehot_encoding(self, value):
        onehot = np.zeros((1, self.size), dtype=np.float32)
        value_idx = self.value_map[value]
        onehot[0, value_idx] = onehot[0, value_idx] + 1
        return onehot

    def print_state(self):
        print("Space ID:\t", self.id)
        print("Space name:\t", self.name)
        print("Space value:", self.values)
        print("Space size:\t", self.size)
        print("index map:\t", self.index_map)
        print("value map:\t", self.value_map)
        print("g_index map:\t", self.g_index_map)
        print("g_value map:\t", self.g_value_map)


class SpaceManager(object):
    """
    The Model Space Manager
    Manage the search space of the neural network architectures
    """
    def __init__(self):
        self.spaces = list()
        self.num_spaces = 0
        self.space_dim = 0

    def add_space(self, name, values):
        """
        :param name: The name of this search space
        :param values: The range of this search space
        :return: the total number of search spaces in the SpaceManager
        """
        self.spaces.append(Space(self.num_spaces, self.space_dim, name, values))
        self.space_dim += len(values)
        self.num_spaces += 1
        return self.num_spaces

    def init_random_states(self, num_layers):
        """
        Initialize a set of states in all the search spaces in the SpaceManager
        :param num_layers: as named
        :return: the state vector that will be passed to the RNN controller
        """
        RNN_state_input = []

        for i in range(self.num_spaces * num_layers):
            ispace = self[i]
            rand_choice = ispace.index_map[np.random.randint(0, ispace.size)]
            onehot = ispace.onehot_encoding(rand_choice, self.space_dim)
            RNN_state_input.append(onehot)

        return RNN_state_input

    def encode_state_space_list(self, state_list):
        space_dim = 0
        encode_state_list = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=1)[0] + space_dim
            space_dim += self[id].size
            state_val_idx %= self.space_dim 
            encode_state_list.append(state_val_idx)
        return encode_state_list

    def parse_state_space_list(self, state_list, local=False):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values
        :param state_list: list of one hot encoded states
        :return list of state values
        '''
        state_values = []
        space_dim = 0
        for id, state_one_hot in enumerate(state_list):
            if local:
                state_val_idx = np.argmax(state_one_hot, axis=-1)[0] + space_dim
            else:
                state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            state_val_idx %= self.space_dim
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)
            space_dim += self[id].size

        return state_values

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        Returns:
            The actual state value at given value index
        '''
        index_map = self[id].g_index_map

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def print_state_space(self):
        """ Visualize the information in the current states """
        print('=' * 20, 'STATE SPACE', '=' * 20)

        for ispace in self.spaces:
            ispace.print_state()
            print()

        print('=' * 53)

    def print_actions(self, actions):
        """ Visual the Action """
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.num_spaces == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.num_spaces) + 1), "*" * 20)

            state = self[id]
            name = state.name
            val = [(n, p) for n, p in zip(state.values, *action)]
            print("%s: " % name, val)
        print()

    def embedding_encode(self, id, state_value):
        return self[id].local_onehot_encoding(state_value)

    def __getitem__(self, n):
        return self.spaces[n % self.num_spaces]


