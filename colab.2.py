
# https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb#scrollTo=l50-R2kwFY7Z

class XlaEmbeddingBag(nn.Module):
    """ 
    nn.EmbeddingBag is not lowered just yet to xla.
    This performs the same functionality, in an xla compatible, sub-optimal way.
    Warning!: only works with constant offsets atm.
    """

    def __init__(self, n, m, mode, offset, *args, **kwargs):
        super(XlaEmbeddingBag, self).__init__()
        self.n = n 
        self.m = m 
        self.mode = mode
        self.offset = offset
        self.embtable = nn.Embedding(n, m, *args, **kwargs)

    def forward(self, sparse_index_group_batch, sparse_offset_group_batch):
        emb = self.embtable(sparse_index_group_batch)
        # XXX: only works w/ constant offset atm
        bsz = emb.size(0) // self.offset
        emb = emb.reshape(bsz, self.offset, *emb.size()[1:])
        reduce_fn = getattr(torch, self.mode)
        return reduce_fn(emb, axis=1)
        #return reduce_fn(self.embtable(_) for _ in inp_list)

    @property
    def weight(self):
        return self.embtable.weight

print(dev)
import time
import torch_xla.debug.metrics as met

num_features=3000000
batch_size=8192
nnz = 100
embed_dim = 128
time1 = time.perf_counter()
h_indices  = torch.randint(0, num_features, (batch_size*nnz,)).to(dev)
h_offsets  = torch.zeros(batch_size, dtype=torch.int64)
  for i in range(batch_size):
    h_offsets[i] = i * nnz
time2 = time.perf_counter()
print(h_indices.device)
print("Creating inputs take ", time2 - time1, " seconds")
time3 = time.perf_counter()
h_emb = XlaEmbeddingBag(num_features, embed_dim, mode='sum', nnz).to(dev)
time4 = time.perf_counter()
print("Creating emb take ", time4 - time3, " seconds")
h_results = torch.zeros(batch_size, embed_dim)
total_bytes = batch_size * nnz * embed_dim * h_emb.weight.element_size()

for i in range(4):
  time5 = time.perf_counter()
  h_results = h_emb(h_indices, h_offsets)
  time6 = time.perf_counter()
  torch_xla._XLAC._xla_sync_multi([h_results], devices=[], wait=True, sync_xla_data=True)
  time7 = time.perf_counter()
  print("EMB take {0:0.6f} sec with sync {1:.6f} algo bw {2:.3f} GB/s".format(time6 - time5, time7- time5,  )
  print("TPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0/(time7- time5)/1.0e9))

h_results = h_results.cpu()
print(met.metrics_report())
print(h_results)
print(h_results.device)

output
========
xla:1
xla:1
Creating indices take  0.1654432209998049  seconds
Creating emb take  19.96984430199973  seconds
Table lookup take  6.9231922339995435  seconds 6.96734616699996
Table lookup take  6.927355632999934  seconds 6.971949914999641
Table lookup take  6.994838251999681  seconds 7.0386971949992585
Table lookup take  6.990675027999714  seconds 7.034779283999342
tensor([[ -6.1681,   2.6904,  -6.4269,  ...,   2.8487, -17.6653,  26.0074],
        [  1.7584, -13.2719,   5.0239,  ...,  18.4227,   8.4663, -16.1359],
        [ 12.6307,  11.7968,   3.7383,  ...,   7.3508,  -3.6538,  -1.7788],
        ...,
        [-10.4265,   6.9984,  -2.6491,  ...,   4.7410,   6.6626,  -9.4342],
        [  0.1223,   0.9951, -17.2891,  ..., -15.3233, -14.7186,   5.0303],
        [ 18.5981,  23.5458,   3.6765,  ...,  -9.4396,  -3.4477,   7.6991]],
       grad_fn=<CopyBackwards>)
cpu
