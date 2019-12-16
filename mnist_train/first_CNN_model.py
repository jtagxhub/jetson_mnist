#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework import graph_io

# Prepare the dataset:
def prepare_dataset(batch_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel from 255 -> 1.0
    x_train.astype(np.float32)
    x_test.astype(np.float32)

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train = x_train[..., np.newaxis]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).repeat()
    train_steps = ( x_train.shape[0] + (batch_size - 1) ) / batch_size

    x_test = x_test[..., np.newaxis]
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).repeat()
    test_steps = ( x_test.shape[0] + (batch_size - 1) ) / batch_size

    return train_dataset, train_steps, test_dataset, test_steps

def prepare_rawdata():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel from 255 -> 1.0
    x_train.astype(np.float32)
    x_test.astype(np.float32)

    x_train = x_train/255.0
    x_test = x_test/255.0

    return x_train, y_train, x_test, y_test

def visualize_character(cc):
    cc=np.reshape(cc, (28, 28))
    for row_id in range(28):
        row=""
        for col_id in range(28):
            if cc[row_id, col_id] > 0:
                row += '@'
            else:
                row += ' '
        print(row)

# Construct the model
def mlp_model():

    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Configure the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model

def cnn_model_1():
    input_tensor=tf.keras.Input(shape=(28, 28, 1), name="in")
    x=tf.keras.layers.Conv2D(6, (5,5), padding='same', activation="relu")(input_tensor)
    x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Conv2D(16, (5,5), activation="relu")(x)
    x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Conv2D(120, (5,5), activation="relu")(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(84, activation="relu")(x)
    x=tf.keras.layers.Dense(10, activation="softmax", name="out")(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    model.summary()

# Configure the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def cnn_model_2(train=True):
    # pass
    input_tensor=tf.keras.Input(shape=(28, 28, 1))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu")(input_tensor)
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    if train:
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    if train:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    model.summary()

# Configure the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def pb_saver(model, output_path, as_text=False):
    orig_output_node_name = [node.op.name for node in model.outputs]
    sess = tf.compat.v1.keras.backend.get_session()
    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            orig_output_node_name)

    temp_pb_path = output_path
    graph_io.write_graph(constant_graph,
                         os.path.dirname(temp_pb_path),
                         os.path.basename(temp_pb_path),
                         as_text=as_text)


def train_with_rawdata(output_path=None):
    model = cnn_model_1()

    x_train, y_train, x_test, y_test = prepare_rawdata()

    x_train = x_train[..., np.newaxis]
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    x_test = x_test[..., np.newaxis]
    model.evaluate(x_test, y_test)

    if output_path :
        pb_saver(model, output_path)


def train_with_dataset(output_path=None, log=False):
    model = cnn_model_1()

    callbacks =[]
    if log:
        callbacks = [tf.keras.callbacks.TensorBoard("./log")]

    train_dataset, train_steps, val_dataset, val_steps = prepare_dataset(10)

    model.fit(train_dataset, epochs=5,
              steps_per_epoch=train_steps, callbacks=callbacks,
              validation_data=val_dataset,
              validation_steps=val_steps)

    model.evaluate(val_dataset, steps=val_steps)

    if output_path:
        pb_saver(model, output_path)


def main():
    train_with_rawdata("./test.pb")
    # train_with_dataset("./test.pb")

    # pb_saver(model, "./mnist_cnn.pb")

    # orig_output_node_name = [node.op.name for node in model_2.outputs]
    # sess = tf.compat.v1.keras.backend.get_session()
    # constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            # sess,
            # sess.graph.as_graph_def(),
            # orig_output_node_name)

    # temp_pb_path = "./mnist_cnn.pb"
    # graph_io.write_graph(constant_graph,
                         # os.path.dirname(temp_pb_path),
                         # os.path.basename(temp_pb_path),
                         # as_text=False)

# Save checkpoint weights

# Load checkpoint weights

# Save H5 weights

# Save and load entire model


if __name__=="__main__":
    main()
