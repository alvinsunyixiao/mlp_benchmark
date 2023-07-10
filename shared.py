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


import timeit
import numpy as np


# network configurations
BATCH_SIZE = 2**16
INPUT_DIM = 32
HIDDEN_DIM = 1024
NUM_HIDDEN = 10

# misc
NUM_RUNS = 300


def generate_data(seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.normal(size=(BATCH_SIZE, INPUT_DIM))

def print_time_stats(func):
    time_total = timeit.timeit(func, number=NUM_RUNS)
    time_avg = time_total / NUM_RUNS
    print(f"Took {time_total:.2f}s in total, averaging {time_avg:.4f}s per call")
