# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch

def measure_cpu(a, b, steps, m):

    global c
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
    end = time.perf_counter()
    c.to('cpu')
    return end - start


def measure_gpu(a, b, steps, m):

    global c
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
        # To be consistent with TPU
        # Add data dependency to prevent loop elimination
        i1 = i % m
        a[i1][0] = a[i1][0] + c[i1][0]
    torch.cuda.synchronize()
    end = time.perf_counter()
    c.to('cpu')
    return end - start


def measure_xla(a, b, steps, m):

    import torch_xla

    def sync(tensor, dev):
        torch_xla._XLAC._xla_sync_multi([tensor], devices=[str(dev)], wait=True, sync_xla_data=True)

    global c
    c = torch.mm(a, b)
    start = time.perf_counter()
    for _ in range(steps):
        # Add data dependency to prevent loop elimination
        # The PyTorch/XLA lazy evaluation will eliminate the loop
        # Simplier data dependency will not work
        b[0] = torch.min(c[0], b[0])
        c = torch.min(torch.mm(a, b), c)
    sync(c, c.device)
    end = time.perf_counter()
    c.to('cpu')
    return end - start

def run_single(args, m, n, k):

    dtype = args.dtype
    device = args.device
    warmups = args.warmups
    steps = args.steps

    dt = torch.float32
    if (dtype == "float16" or dtype == "half"):
        dt = torch.float16
    elif (dtype == "bfloat16"):
        dt = torch.bfloat16

    torch.manual_seed(0)

    elap = 0.0

    a = torch.randn(m, k).to(dt)
    b = torch.randn(k, n).to(dt)
    c = torch.zeros(m, n).to(dt)

    is_cuda = torch.cuda.is_available()

    if device == 'cpu':

        measure_cpu(a, b, warmups, m)
        elap = measure_cpu(a, b, steps, m)

    elif device == 'gpu' and is_cuda:

        ncuda = torch.cuda.device_count()
        # print("There are {} cuda devices".format(ncuda))
        # print("The first cuda device name is {} ".format(torch.cuda.get_device_name()))
        cuda0 = torch.device('cuda:0')
        with torch.cuda.device(cuda0):
            acuda = a.to(cuda0)
            bcuda = b.to(cuda0)
            measure_gpu(acuda, bcuda, warmups, m)
            elap = measure_gpu(acuda, bcuda, steps, m)

    else:
        # import torch_xla
        import torch_xla.core.xla_model as xm

        # alldev = xm.get_xla_supported_devices()
        # allrealdev = xm.xla_real_devices(alldev)
        # print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))

        dev = xm.xla_device()
        a = a.to(dev)
        b = b.to(dev)
        c = c.to(dev)
        measure_xla(a, b, warmups, m)
        elap = measure_xla(a, b, steps, m)

    return elap

def run(args, dataset):

    print("----------------------------------------------------------------")
    print("         M         N          K          Time(s)      Rate(GF/s)")
    print("----------------------------------------------------------------")
    for i in range(len(dataset)):
        m, n, k = dataset[i]
        elap = run_single(args, m, n, k)
        print("{0:10}, {1:10}, {2:10},     {3:10.6f}     {4:.3f} ".format(m, n, k, elap,
            m * n * k * 2 * 1.0 / (elap * 1000000000 / args.steps)))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the performance of GEMM using mm, or matmul")
    # model related parameters
    parser.add_argument("-m", "--msize", type=int, default=1024)
    parser.add_argument("-n", "--nsize", type=int, default=1024)
    parser.add_argument("-k", "--ksize", type=int, default=1024)
    parser.add_argument("-t", "--dtype", type=str, default="float32")
    parser.add_argument("-d", "--device", choices=['cpu', 'gpu', 'tpu'], type=str, default='cpu')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()

    d = [(args.msize, args.nsize, args.ksize)]
    run(args, d)

