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
import functools
import numpy as np
import timeit

import jax
import jax.numpy as jnp
from flax import linen as jax_nn

import shared

class MLP(jax_nn.Module):
    output_dim: int
    hidden_dim: int
    num_hidden: int

    @jax_nn.compact
    def __call__(self, x):
        for _ in range(self.num_hidden):
            x = jax_nn.Dense(self.hidden_dim)(x)
            x = jax_nn.elu(x)
        x = jax_nn.Dense(self.output_dim)(x)

        return x

@functools.partial(jax.jit, static_argnums=2)
def compute_grad(data: jax.Array, params: T.Any, model: jax_nn.Module):
    def loss_func(p):
        output = model.apply({"params": p}, data)
        return jnp.mean(jnp.square(data - output))

    grad = jax.grad(loss_func)(params)

    return grad


if __name__ == "__main__":
    data = shared.generate_data()
    data = jnp.asarray(data)

    model = MLP(shared.INPUT_DIM, shared.HIDDEN_DIM, shared.NUM_HIDDEN)
    params = model.init(jax.random.PRNGKey(42), data)["params"]

    compute_grad(data, params, model)

    shared.print_time_stats(lambda: compute_grad(data, params, model))
