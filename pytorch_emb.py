# measuring embeddingbag performance using pytorch
# 1. Table lookup index is generated together for all iterations
#    Currently we see very similar performance across iterations,
#    even with same indices for each iteration
#    So, use same indices for different iteration now. 
#
# 2. Test case:
#    python3 pytorch_emb.py --features=30000000 --embdim=128 --nnz=100 --batch=8192 --testgpu=1 --verify=1 --steps=10
#

import time
import sys
import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
  import sys
  import argparse

  parser = argparse.ArgumentParser(
     description="Measure the performance of pytorch embeddingbag"
  )
  # model related parameters
  parser.add_argument("--features", type=int, default=1024)
  parser.add_argument("--embdim", type=int, default=64)
  parser.add_argument("--nnz", type=int, default=10)
  parser.add_argument("--batch", type=int, default=1000)
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--warmups", type=int, default=2)
  parser.add_argument("--randomseed", type=int, default=0)
  parser.add_argument("--testcpu", type=int, default=1)
  parser.add_argument("--testgpu", type=int, default=0)
  parser.add_argument("--testtpu", type=int, default=0)
  parser.add_argument("--verify", type=int, default=0)
  args = parser.parse_args()

  num_features = args.features
  embed_dim   = args.embdim
  nnz          = args.nnz
  batch_size   = args.batch
  steps        = args.steps
  warmups      = args.warmups

  random_seed  = args.randomseed

  print("Test problem size:")
  print("Number of features : ", num_features)
  print("Embedding size     : ", embed_dim)
  print("Nnz_per_input      : ", nnz)
  print("Number of batches  : ", batch_size)

  torch.manual_seed(random_seed)

  # 1. measure on CPU first
  # h_indices  = torch.randint(0, num_features, (warmups+steps, batch_size, nnz))
  h_indices  = torch.randint(0, num_features, (batch_size, nnz))
  print("Finished generating indices")
  h_emb = nn.EmbeddingBag(num_features, embed_dim, mode='sum')
  print("Finished generating tables")
  h_results = torch.zeros(batch_size, embed_dim)
  g_results = torch.zeros(batch_size, embed_dim)
  t_results = torch.zeros(batch_size, embed_dim)

  total_bytes = batch_size * nnz * embed_dim * h_emb.weight.element_size()

  if (args.testcpu):
    total_times = 0
    start1 = time.perf_counter()
    for i in range(warmups + steps):
      start = time.perf_counter()
      h_results = h_emb(h_indices)
      end  = time.perf_counter()    
      print("Time {0:.6f} ".format(end - start))
      if (i >= warmups):
        total_times += end - start
    end1 = time.perf_counter()
       
    print("CPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(end1-start1, total_times, steps))
    print("CPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0*steps/total_times/1.0e9))
    print("CPU results: ", h_results)

  if (args.testgpu):
    cuda_avail = torch.cuda.is_available()
    if (cuda_avail):
      ncuda = torch.cuda.device_count()
      print("There are {} cuda devices".format(ncuda))
      print("The current cuda device name is {} ".format(torch.cuda.get_device_name()))
      cuda0 = torch.device('cuda:0')
      total_times = 0
      with torch.cuda.device(cuda0):
        g_emb      = h_emb.to(cuda0)
        g_indices  = h_indices.to(cuda0)
        torch.cuda.synchronize()

        start1 = time.perf_counter()
        for i in range(warmups + steps):
          start = time.perf_counter()
          results = g_emb(g_indices)
          torch.cuda.synchronize()
          end  = time.perf_counter()
          print("Time: {0:.6f} ".format(end - start))
 #         print(results)

          if (i >= warmups):
            total_times += end - start

        end1 = time.perf_counter()
        print("---------")
        print("GPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(end1-start1, total_times, steps))
        print("GPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0*steps/total_times/1.0e9))
        print("---------")
        print("GPU results: ", results)
        g_results = results.to('cpu')

    if (args.verify and args.testcpu):
      if (torch.equal(h_results, g_results)):
          print("Success! CPU results and GPU results match!\n")
      else:
          print("Failed!  CPU and GPU results does not match!\n")
          print("CPU results:")
          print(h_results)
          print("GPU results:")
          print(g_results)

  if (args.testtpu):
    import torch_xla
    import torch_xla.core.xla_model as xm

    def syncTPU(tensor):
      torch_xla._XLAC._xla_sync_multi([tensor], devices=[], wait=True, sync_xla_data=True)

    alldev = xm.get_xla_supported_devices()
    allrealdev = xm.xla_real_devices(alldev)
    print("Found {0} devices: {1}".format(len(allrealdev), allrealdev))

    dev = xm.xla_device()   
    # dev = xm.xla_device(n=2, devkind='TPU')
    t_emb = h_emb.to(dev)
    t_indices = h_indices.to(dev)
    
    start1 = time.perf_counter()
    for i in range(warmups + steps):
      start = time.perf_counter()
      results = t_emb(t_indices)
      syncTPU(results)
      end  = time.perf_counter()
      print("Time: {0:.6f} ".format(end - start))
      if (i >= warmups):
        total_times += end - start

    end1 = time.perf_counter()
    print("---------")
    print("GPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(end1-start1, total_times, steps))
    print("GPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0*steps/total_times/1.0e9))
    print("---------")
    print("GPU results: ", results)
    t_results = results.to('cpu')

    if (args.verify and args.testcpu):
      if (torch.equal(h_results, t_results)):
        print("Success! CPU results and TPU results match!\n")
      else:
        print("Failed!  CPU and TPU results does not match!\n")

