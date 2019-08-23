# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class DanQ(keras.Model):
    def __init__(self):
        super(DanQ, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=320,
            kernel_size=26,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=13,
            strides=13,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        forward_layer = keras.layers.LSTM(
            units=320,
            return_sequences=True,
            return_state=True)

        backward_layer = keras.layers.LSTM(
            units=320,
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


    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DeepSEA model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 975, 320]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 975, 320]
        # Output Tensor Shape: [batch_size, 75, 320]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Bidirectional RNN layer 1
        # Input Tensor Shape: [batch_size, 75, 320]
        # Output Tensor Shape: [batch_size, 75, 640]
        temp = self.bidirectional_rnn(temp, training = training, mask=mask)
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