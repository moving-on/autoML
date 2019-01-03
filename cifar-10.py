import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

from NasProject import NasProject

INPUT_SIZE = np.array([32, 32, 3])
OUTPUT_SIZE = 10
NUM_CONV = 4
NUM_DENSE = 1
DENSE_CONFIG = [128, 0.25]

if __name__ == '__main__':
    # load training data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    dataset = [x_train, y_train, x_test, y_test]

    # create tje NAS job
    cifar10_nas = NasProject('CIFAR-10', dataset, INPUT_SIZE, OUTPUT_SIZE)
    cifar10_nas.config_neural_space(num_conv_layers=NUM_CONV,
                                    num_dense_layers=NUM_DENSE,
                                    dense_config=DENSE_CONFIG)

    # define search space
    cifar10_nas.add_seach_space(name='kernel', values=[3, 5, 7])
    cifar10_nas.add_seach_space(name='filters', values=[16, 32, 48, 64, 96])
    cifar10_nas.add_seach_space(name='strides', values=[1, 2])
    cifar10_nas.print_search_space()

    # Finally, train
    cifar10_nas.train()
