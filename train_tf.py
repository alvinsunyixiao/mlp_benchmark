# MIT License
#
# Copyright (c) 2023 Alvin Sun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import typing as T
import numpy as np
import timeit

import tensorflow as tf
from tensorflow import keras as tfk

import shared


def build_model(output_dim: int, hidden_dim: int, num_hidden: int) -> tfk.Model:
    model = tfk.Sequential()
    for _ in range(num_hidden):
        model.add(tfk.layers.Dense(hidden_dim, activation="elu"))
    model.add(tfk.layers.Dense(output_dim))

    return model

@tf.function(jit_compile=True)
def compute_grad(data: tf.Tensor, model: tfk.Model):
    with tf.GradientTape() as tape:
        output = model(data)
        loss = tf.reduce_mean(tf.square(data - output))

    grad = tape.gradient(loss, model.variables)

    return grad

if __name__ == "__main__":
    data = shared.generate_data()
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    model = build_model(shared.INPUT_DIM, shared.HIDDEN_DIM, shared.NUM_HIDDEN)
    compute_grad(data, model)

    shared.print_time_stats(lambda: compute_grad(data, model))
