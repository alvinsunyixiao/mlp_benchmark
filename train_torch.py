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


import numpy as np
import timeit

import torch
from torch import nn

import shared


def build_model(output_dim: int, hidden_dim: int, num_hidden: int) -> nn.Module:
    layers = [nn.Linear(output_dim, hidden_dim),
              nn.ELU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ELU())
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers).cuda()


@torch.compile(mode="reduce-overhead")
def compute_grad(data: torch.Tensor, model: nn.Module):
    output = model(data)
    loss = torch.mean(torch.square(data - output))
    loss.backward()

    return [w.grad for w in model.parameters()]


if __name__ == "__main__":
    data = shared.generate_data()
    data = torch.tensor(data, dtype=torch.float32).cuda()

    model = build_model(shared.INPUT_DIM, shared.HIDDEN_DIM, shared.NUM_HIDDEN)
    model.train()

    compute_grad(data, model)

    shared.print_time_stats(lambda: compute_grad(data, model))

