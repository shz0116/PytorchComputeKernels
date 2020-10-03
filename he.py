
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm

m = 4096
n = 40928
k = 1024
h = min(n,k)
N = 201

#alldev = xm.get_xla_supported_devices()
#allrealdev = xm.xla_real_devices(alldev)
#print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))
print("torch version: ", torch.__version__)

device = xm.xla_device()
a = torch.randn(m,k).to(device)
b = torch.randn(k,n).to(device)
total_delta = 0

s = time.time()
c = torch.mm(a, b)
for i in range(0, N):
#  a[:, 0:h] = torch.min(a[:, 0:h], c[:, 0:h])
#  a[0, 0:h] = torch.min(a[0, 0:h], c[0, 0:h])
  b[0] = torch.min(b[0], c[0])
  c += torch.mm(a, b)
# xm.mark_step()
# print(torch_xla._XLAC._get_xla_tensors_text([c]))
# xm.mark_step()
torch_xla._XLAC._xla_sync_multi([c], devices=[], wait=True, sync_xla_data=True)
total_delta = time.time() - s
c.cpu()
total_time = time.time() - s
print("TIME: ", total_time, total_delta)

FLOPS = 2.0 * (m * n *k) * (N - 1) / total_delta
print('{:.2f} GFLOPS'.format(FLOPS * 1e-9))
FLOPS = 2.0 * (m * n *k) * (N - 1) / total_time
print('{:.2f} GFLOPS'.format(FLOPS * 1e-9))
