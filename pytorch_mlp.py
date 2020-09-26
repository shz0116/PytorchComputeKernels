#!/usr/bin/env python3

import time

# import apex
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(
        self, batch_size, num_batches, input_size, output_size, transform=None
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return self.batch_size * self.num_batches

    def __getitem__(self, idx):
        input_sample = torch.FloatTensor(self.input_size).uniform_(-1, 1)
        output_label = torch.randint(0, self.output_size, (1,), dtype=torch.long)[0]
        return input_sample, output_label


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(Net, self).__init__()
        self.layer_num = layer_num
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hid = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for _ in range(self.layer_num):
            x = self.linear_hid(x)
            x = F.relu(x)
        x = self.linear_out(x)
        x = F.softmax(x, dim=1)
        return x

def train_cpu(
    model,
    device,
    train_loader,
    optimizer,
    data_type,
    args,
):
    loss_f = nn.CrossEntropyLoss().to(device)

    model.train()
    start_time = time.time()

    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(args.num_batches):
        data = torch.randn(args.batch_size, args.input_size, device=device)
        target = torch.randint(
            args.output_size, [args.batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)
        if data_type == "float16":
            data = data.half()

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()

    return start_time, loss

def train_gpu(
    model,
    device,
    train_loader,
    optimizer,
    data_type,
    args,
):
    loss_f = nn.CrossEntropyLoss().to(device)

    if data_type == "float16":
        model = apex.fp16_utils.network_to_half(model)

    model.train()
    start_time = time.time()

    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(args.num_batches):
        data = torch.randn(args.batch_size, args.input_size, device=device)
        target = torch.randint(
            args.output_size, [args.batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)
        if data_type == "float16":
            data = data.half()

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()

    return start_time, loss

def train_tpu(
    model,
    device,
    train_loader,
    optimizer,
    data_type,
    args,
):
    import torch_xla.core.xla_model as xm
    loss_f = nn.CrossEntropyLoss().to(device)

    model.train()
    start_time = time.time()

    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(args.num_batches+10):
        data = torch.randn(args.batch_size, args.input_size, device=device)
        target = torch.randint(
            args.output_size, [args.batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)
        if data_type == "float16":
            data = data.half()

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        if i < 10:
            start_time = time.time()

    return start_time, loss

def train(
    model,
    device,
    train_loader,
    optimizer,
    data_type,
    args,
):

    if device.type == 'cpu':
        start_time, loss = train_cpu(model, device, train_loader, optimizer, data_type, args)

    if device.type == 'cuda':
        start_time, loss = train_gpu(model, device, train_loader, optimizer, data_type, args)

    if device.type == 'xla':
        start_time, loss = train_tpu(model, device, train_loader, optimizer, data_type, args)

    print("Loss is ", loss)
    total_time = time.time() - start_time
    total_examples = args.batch_size * args.num_batches

    print("---------------------------------------------")
    print("QPS: {:.6f}".format(total_examples / total_time))
    print("global_step/sec: {:.6f}".format(args.num_batches / total_time))
    print("Total time: {:.6f}".format(total_time))
    hidden_size = args.hidden_size
    flops = args.batch_size * (
        hidden_size * hidden_size * args.layer_num
        + hidden_size * args.input_size
        + hidden_size * args.output_size
    )
    # Forward 2x and Backward 4x
    flops *= 6 * args.num_batches
    print("TFLOPS: {}".format(flops / total_time / 1e12))
   
def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the performance of MLP"
    )
    parser.add_argument("--device", required=True, choices=['cpu', 'gpu', 'tpu'])
    parser.add_argument("--optimizer-type", default="sgd", help="Optimizer: SGD", choices=["sgd"])
    parser.add_argument("--data-type", default="float", help="data type", choices=["float", "float16", "bfloat16"])
    parser.add_argument("--layer-num", type=int, default=20, help="Number of Linear layers")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--input-size", type=int, default=1024, help="Input layer size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Number of hidden_sizes per layer")
    parser.add_argument("--output-size", type=int, default=1024, help="Output layer size")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches to train")

    args = parser.parse_args()
    device = args.device
    optimizer_type = args.optimizer_type
    data_type = args.data_type
    layer_num = args.layer_num
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    num_batches = args.num_batches     

    dtype = torch.float32
    if data_type == "float16":
        dtype = torch.float16
    if data_type == "bfloat16":
        dtype = torch.bfloat16

    torch.manual_seed(1)

    if device == 'cpu':
        dev = torch.device("cpu")
        print("Using device:", dev)
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)        
        if optimizer_type == "sgd":
            optimizer = optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.01,
            )
        else:
            assert 0, "Unsupported optimizer type"
        kwargs = {}

    if device == 'gpu' and torch.cuda.is_available():
        
        import apex

        dev = torch.device("cuda:0")
        print("Using device:", dev)
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
        if optimizer_type == "sgd":
            optimizer = apex.optimizers.FusedSGD(
                model.parameters(),
                lr=0.01,
                set_grad_none=True,
            )
        else:
            assert 0, "Unsupported optimizer type"

        if data_type == "float16":
            apex.amp.initialize(
                model, optimizer, opt_level="O2", verbosity=1
                # model, optimizer, opt_level="O3", verbosity=1
            )

        kwargs = {"num_workers": 1, "pin_memory": True}


    if device == "tpu":
        
        import torch_xla
        import torch_xla.core.xla_model as xm

        alldev = xm.get_xla_supported_devices()
        allrealdev = xm.xla_real_devices(alldev)
        print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))

        dev = xm.xla_device()
        print("Using device:", dev)
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
        if optimizer_type == "sgd":
            optimizer = optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.01,
            )
        else:
            assert 0, "Unsupported optimizer type"

        kwargs = {"num_workers": 1} 

    train_loader = torch.utils.data.DataLoader(
        myDataset(batch_size, num_batches, input_size, output_size),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    train(
        model,
        dev,
        train_loader,
        optimizer,
        dtype,
        args,
    )

if __name__ == "__main__":
    main()

