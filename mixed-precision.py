import argparse 
import sys
import time

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


# controls which optimizer to use
optim_cls = torch.optim.SGD


class MixedPrecisionOptimizer(object):
    """
    We do the forward/backward in bfloat16 (i.e., activations and 
    gradients are in bfloat16) and compute the loss in fp32. We also
    keep a master copy of the weights in fp32 and do the optimization
    step in fp32.
    """ 
    def __init__(self, params, **kwargs):
        self.bf16_params = list(params)
        self.fp32_params = init_fp32_params_from_bf16(self.bf16_params)
        self.fp32_optimizer = optim_cls(self.fp32_params, **kwargs)

    def step(self):
        # cast bf16 gradients to fp32
        copy_bf16_grads_to_fp32(self.bf16_params, self.fp32_params)
        # optimize the fp32 master weights using the fp32 grads
        self.fp32_optimizer.step()
        # copy the updated master weights back to bf16
        copy_fp32_params_to_bf16(self.fp32_params, self.bf16_params)

    def zero_grad(self):
        for p in self.fp32_params:
            p.grad.data.zero_()
        for p in self.bf16_params:
            p.grad.data.zero_()


def init_fp32_params_from_bf16(bf16_params):
    device = bf16_params[0].device
    fp32_buf = [
        torch.zeros(p.shape,dtype=torch.float32, device=device)
        for p in bf16_params
    ]
    for p16, p32 in zip(bf16_params, fp32_buf):
        p32.copy_(p16.data)
    fp32_params = [torch.nn.Parameter(fp32) for p32 in fp32_buf]
    for p32 in fp32_params:
        p32.grad = fp32.new_zeros(p32.shape)
    return fp32_params


def copy_bf16_grads_to_fp32(bf16_params, fp32_params):
    for p16, p32 in zip(bf16_params, fp32_params):
        assert p16.requires_grad and p16.grad is not None
        p32.grad.data.copy_(p16.grad.data)

def copy_fp32_params_to_bf16(fp32_params, bf16_params):
    for p16, p32 in zip(bf16_params, fp32_params):
        assert p32.requires_grad and p32.grad is not None
        p16.data.copy_(p32.data)

def reduce_gradients(params):
    # compared to xm.reduce_gradients, this takes the params directly
    # instead of extracting them from an optimizer instance
    count = torch_xla._XLAC._xla_get_replication_devices_count()
    if count > 1:
        gradients = [p.grad for p in params if p.grad is not None]
        xm.all_reduce('sum', gradients, scale=1.0 / count, groups=None)


class Net(nn.Module):
    def __init__(self, num_embed=32768, embed_dim=768, num_layers=12):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embed, embedding_dim=embed_dim, padding_idx=0
        )
        self.layers_a = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3*embed_dim),  # q, k, v input projection
                nn.Linear(3*embed_dim, embed_dim),  # skip self-attention
                nn.Linear(embed_dim, embed_dim),    # output projection
                # nn.Dropout(),
            )
            for i in range(num_layers)
        ])
        self.layers_b = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 4*embed_dim),  # FFN
                nn.ReLU(),
                nn.Linear(4*embed_dim, embed_dim),  # FFN
                # nn.Dropout(0.1),
            )
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, num_embed)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer_a, layer_b in zip(self.layers_a, self.layers_b):
            x = x + layer_a(x)
            x = x + layer_b(x)
        x = self.out_proj(x)
        return x


def main(rank):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=[
        'bf16',  # pure bfloat16 training mode
        'fp32',  # pure float32 training mode
        'mixed',  # mixed precision training mode (see MixedPrecisionOptimizer for details)
    ])
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--measurement_steps', type=int, default=50)
    args = parser.parse_args()

    print("initializing dataloader")
    item = torch.arange(1, args.seqlen + 1, dtype=torch.long)
    dataloader = torch.utils.data.DataLoader(
        [item for _ in range(args.bsz * 1000)],
        batch_size=args.bsz,
    )

    device = xm.xla_device()
    if args.mode in {'bf16', 'mixed'}:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    xm.master_print("initializing model")
    model = Net().to(device=device, dtype=dtype)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    xm.master_print("num model params: {}".format(sum(p.numel() for p in model.parameters())))

    xm.master_print("initializing optimizer")
    if args.mode == 'mixed':
        optimizer = MixedPrecisionOptimizer(model.parameters(), lr=0.001)
    else:
        optimizer = optim_cls(model.parameters(), lr=0.001)

    xm.master_print("initializing paraloader")
    itr = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)

    xm.master_print("beginning warmup")
    for i, batch in enumerate(itr):
        if i == args.warmup_steps:
            print("end warmup, begin measurement")
            start_time = time.time()

        x = model(batch)
        x = x.float()  # compute loss in fp32
        loss = loss_fn(
            x.view(-1, x.size(-1)),
            target=batch.view(-1)
        )
        loss.backward()

        reduce_gradients(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

        if i == args.warmup_steps + args.measurement_steps:
            measured_time = time.time() - start_time
            print(
                "end measurement, time for rank {}: {}"
                .format(rank, measured_time)
            )
            break


if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=1)

