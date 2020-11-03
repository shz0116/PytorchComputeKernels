import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# "Map function": acquires a corresponding Cloud TPU core, creates a tensor on it,
# and prints its core
def simple_map_fn(index, flags):
  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(1234)

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()  

  # Creates a tensor on this process's device
  t = torch.randn((2, 2), device=device)

  print("device: ", device, " str: ", str(device), " real: ", xm.xla_real_devices([str(device)]))
  print("master: ", xm.is_master_ordinal())
  print("Process", index ,"is using", xm.xla_real_devices([str(device)])[0])

  # Barrier to prevent master from exiting before workers connect.
  xm.rendezvous('init')

# Spawns eight of the map functions, one for each of the eight cores on
# the Cloud TPU
flags = {}
# Note: Colab only supports start_method='fork'
xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')

