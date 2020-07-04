
# measuring gemm (matmul, mm) performance using pytorch
# using one matrix

import time
import numpy as np
import sys
import torch

def measure_cpu(a, b, steps, m):
  global c
  start = time.perf_counter()
  for i in range(steps):
    c = torch.mm(a, b)
    i1 = i % m
    a[i1][0] = a[i1][0] + c[i1][0]   #prevent mm done only once
  end1 = time.perf_counter()
  c.to('cpu')
  end2 = time.perf_counter()
  return end1 - start, end2 - start

def measure_gpu(a, b, steps, m):
  global c
  torch.cuda.synchronize()
  start = time.perf_counter()
  for i in range(steps):
    c = torch.mm(a, b)
    i1 = i % m
    a[i1][0] = a[i1][0] + c[i1][0]   #prevent mm done only once
  torch.cuda.synchronize()
  end1 = time.perf_counter()
  c.to('cpu')
  end2 = time.perf_counter()
  return end1 - start, end2 - start

if __name__ == "__main__":
  import sys
  import argparse

  parser = argparse.ArgumentParser(
     description="Measure the performance of GEMM using mm, or matmul"
  )
  # model related parameters
  parser.add_argument("--msize", type=int, default=1024)
  parser.add_argument("--nsize", type=int, default=1024)
  parser.add_argument("--ksize", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--dtype", type=str, default="float32")
  parser.add_argument("--testcpu", type=int, default=1)
  parser.add_argument("--testgpu", type=int, default=0)
  parser.add_argument("--testtpu", type=int, default=0)
  parser.add_argument("--verify", type=int, default=0)
  parser.add_argument("--warmups", type=int, default=100)
  args = parser.parse_args()

  m = args.msize
  n = args.nsize
  k = args.ksize
  dt = torch.float32
  if (args.dtype == "float16" or args.dtype == "half"):
    dt = torch.float16
  elif (args.dtype == "bfloat16"):
    dt = torch.bfloat16

  print("Test problem size for m, n, k are : ", m, n, k)
  print("Test problem data type : ", dt)

  torch.manual_seed(0)

  warmups = args.warmups
  steps = args.steps
  elapsed1 = 0.0
  elapsed2 = 0.0
  elapsed3 = 0.0
  elapsed4 = 0.0

  # 1. measure on CPU first, generate flaot32 first,
  # some data type are not directly supported by randn
  a = torch.randn(m, k).to(dt)
  b = torch.randn(k, n).to(dt)
  c = torch.zeros(m, n).to(dt)

  # cpu and gpu returns the same results
  a_save0 = torch.zeros(m)
  a_save = a_save0.to(dt)
  for i in range(m):
    a_save[i] = a[i][0]

  if (not (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
    measure_cpu(a, b, warmups, m)
    elapsed1, elapsed2 = measure_cpu(a, b, steps, m)

    print("c device: ", c.device, type(c), c.dtype)
    print("c[2x2] : ", c.narrow(0, 0, 2).narrow(1, 0, 2))
    print("------")
    print("CPU Time is {0:.6f} {1:.6f} seconds, rate {2:.3f} GFlops for iter {3}".format(elapsed1, elapsed2, m*n*k*2*1.0/(elapsed2*1000000000/steps), steps))
    print("------\n")

    c.fill_(0)
    for i in range(m):
      a[i][0] = a_save[i]
  else:
    print("\nCPU not support ", a.dtype, " mm op\n")

  # 2. measure on GPU
  is_cuda = torch.cuda.is_available()
  if (is_cuda):
    ncuda = torch.cuda.device_count()
    print("There are {} cuda devices".format(ncuda))
    print("The first cuda device name is {} ".format(torch.cuda.get_device_name()))
    cuda0 = torch.device('cuda:0')
    with torch.cuda.device(cuda0):
      acuda = a.to(cuda0)
      bcuda = b.to(cuda0)
      measure_gpu(acuda, bcuda, warmups, m)
      elapsed1, elapsed2 = measure_gpu(acuda, bcuda, steps, m)

      print("c device: ", c.device, type(c), c.dtype)
      print("c[2x2] : ", c.narrow(0, 0, 2).narrow(1, 0, 2))
      print("------")
      print("GPU Time is {0:.6f} {1:.6f} seconds, rate {2:.3f} GFlops for iter {3} ".format(elapsed1, elapsed2, m*n*k*2*1.0/(elapsed2*1000000000/steps), steps))
      print("------\n")


