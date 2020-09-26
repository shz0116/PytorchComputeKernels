
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm

m = 4096
n = 4092
k = 1024
h = min(n,k)
N = 11

alldev = xm.get_xla_supported_devices()
allrealdev = xm.xla_real_devices(alldev)
print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))
print(torch.__version__)

device = xm.xla_device()
a = torch.randn(m,k).to(device).to(torch.bfloat16)
b = torch.randn(k,n).to(device).to(torch.bfloat16)
total_delta = 0

print("Now")
s = time.time()
for i in range(0, N):
  c = torch.mm(a, b)
  a[:, 0:h] = torch.min(a[:, 0:h], c[:, 0:h])
  if i == 0:
    t1 = time.time()
  print("loop ", i)

print(torch_xla._XLAC._get_xla_tensors_text([c]))
torch_xla._XLAC._xla_sync_multi([c], devices=[], wait=True, sync_xla_data=True)
total_delta = time.time() - s
c.cpu()
total_time = time.time() - s
print("TIME: ", total_time, total_delta)

FLOPS = 2 * (m * n *k) * (N - 1) / total_delta
print('{:.2f} GFLOPS'.format(FLOPS * 1e-9))

