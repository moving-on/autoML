from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Dropout, Flatten, MaxPooling2D


class ModelConstructor(object):
    def __init__(self, input_size, num_output, num_conv, num_dense, dense_config):
        self.input_size = input_size
        self.num_output = num_output
        self.num_conv = num_conv
        self.num_dense = num_dense
        if len(dense_config) != 2 * num_dense:
            print("The number of parameters in dense layers is not correct. Program Exit.")
            exit(3)
        self.dense_config = dense_config

    def build_model(self, actions):
        if len(actions) != 3 * self.num_conv:
            print("The number of parameters in actions list is not correct. Program Exit.")
            exit(3)

        model = Sequential()

        for i in range(self.num_conv):
            kernel_size, filter_size, stride = actions[i*3: (i+1)*3]
            if i == 0:
                # specify the input shape at the first layer
                model.add(Conv2D(filters=filter_size,
                                 kernel_size=kernel_size,
                                 strides=stride,
                                 padding='same',
                                 activation='relu',
                                 input_shape=self.input_size))
            else:
                model.add(Conv2D(filters=filter_size,
                                 kernel_size=kernel_size,
                                 strides=stride,
                                 padding='same',
                                 activation='relu'))

        model.add(MaxPooling2D())
        model.add(Flatten())
        
        for i in range(self.num_dense):
            num_neuron, dropout_rate = self.dense_config[i*2: (i+1)*2]
            model.add(Dense(units=num_neuron, activation='sigmoid'))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(self.num_output, activation='softmax'))

        return model
