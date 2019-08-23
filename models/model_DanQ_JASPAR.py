# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras

class DanQ_JASPAR(keras.Model):
    def __init__(self):
        super(DanQ_JASPAR, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=1024,
            kernel_size=30,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=15,
            strides=15,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        forward_layer = keras.layers.LSTM(
            units=512,
            return_sequences=True,
            return_state=True)

        backward_layer = keras.layers.LSTM(
            units=512,
            return_sequences=True,
            return_state=True,
            go_backwards=True)

        self.bidirectional_rnn = keras.layers.Bidirectional(
            layer=forward_layer,
            backward_layer=backward_layer)

        self.dropout_2 = keras.layers.Dropout(0.5)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=925,
            activation='relu')

        self.dense_2 = keras.layers.Dense(
            units=919,
            activation='sigmoid')

    def build(self, input_shape):
        self.conv_1.build(input_shape=input_shape)
        self.set_weights_by_JASPAR()

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DanQ-JASPAR model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 971, 1024]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 971, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Bidirectional RNN layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.bidirectional_rnn(temp, training=training, mask=mask)
        forward_state_output = temp[1]
        backward_state_output = temp[2]

        # Dropout Layer 2
        temp = self.dropout_2(temp[0], training = training)

        # Flatten Layer 1
        # Input Tensor Shape: [batch_size, 75, 640]
        # Output Tensor Shape: [batch_size, 48000]
        temp = self.flatten(temp)

        # Fully Connection Layer 1
        # Input Tensor Shape: [batch_size, 48000]
        # Output Tensor Shape: [batch_size, 925]
        temp = self.dense_1(temp)

        # Fully Connection Layer 2
        # Input Tensor Shape: [batch_size, 925]
        # Output Tensor Shape: [batch_size, 919]
        output = self.dense_2(temp)

        return output

    def set_weights_by_JASPAR(self):
        JASPAR_motifs = np.load('./data/JASPAR_CORE_2016_vertebrates.npy', allow_pickle=True, encoding='bytes')
        JASPAR_motifs = list(JASPAR_motifs) # shape = (519, )
        reverse_motifs = [JASPAR_motifs[19][::-1, ::-1], JASPAR_motifs[97][::-1, ::-1], JASPAR_motifs[98][::-1, ::-1],
                          JASPAR_motifs[99][::-1, ::-1], JASPAR_motifs[100][::-1, ::-1], JASPAR_motifs[101][::-1, ::-1]]
        JASPAR_motifs = JASPAR_motifs + reverse_motifs # shape = (525, )

        conv_weights = self.conv_1.get_weights()
        for i in range(len(JASPAR_motifs)):
            motif = JASPAR_motifs[i][::-1, :]
            length = len(motif)
            start = np.random.randint(low=3, high=30-length+1-3)
            conv_weights[0][start:start+length, :, i] = motif - 0.25
            conv_weights[1][i] = np.random.uniform(low=-1.0, high=0.0)

        self.conv_1.set_weights(conv_weights)



