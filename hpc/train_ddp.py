#!/usr/bin/env python3
"""DDP‑aware TinyLM training with synthetic data.
- Runs on single GPU (no torch.distributed init) or multi‑GPU via torchrun.
- Aggregates basic throughput metrics and saves a checkpoint on rank 0.
"""
import argparse, os, time, math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ----------------------------- Dataset & Model -----------------------------
class RandomTokenDataset(Dataset):
    def __init__(self, length=20000, seq_len=256, vocab_size=4096, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (length, seq_len), generator=g)
        self.targets = torch.randint(0, vocab_size, (length, seq_len), generator=g)
        self.vocab_size = vocab_size
    def __len__(self): return self.data.size(0)
    def __getitem__(self, i): return self.data[i], self.targets[i]

class TinyLM(nn.Module):
    def __init__(self, vocab_size=4096, d_model=256, nhead=8, nlayers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x)
        h = self.encoder(h)
        return self.lm_head(h)

# ----------------------------- Helpers -----------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--vocab_size', type=int, default=4096)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--nlayers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--outdir', type=str, default='runs/${SLURM_JOB_ID:-local}')
    args = ap.parse_args()

    # Initialize distributed if launched with torchrun
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if get_rank() == 0:
        os.makedirs(args.outdir, exist_ok=True)

    # Data
    ds = RandomTokenDataset(seq_len=args.seq_len, vocab_size=args.vocab_size)
    if is_dist():
        sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    else:
        sampler = None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    # Model & opt
    model = TinyLM(vocab_size=args.vocab_size, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type=='cuda' else None)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    step = 0
    last = time.time()
    tokens_since = 0
    for epoch in range(args.epochs):
        if is_dist():
            dl.sampler.set_epoch(epoch)
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(x)
                loss = loss_fn(logits.view(-1, args.vocab_size), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tokens_since += x.numel()
            if step % args.log_interval == 0:
                now = time.time()
                dt = now - last
                tok_s_local = tokens_since / dt if dt > 0 else 0.0
                # All‑reduce throughput across ranks
                t = torch.tensor([tok_s_local], device=device)
                if is_dist():
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                tok_s_global = t.item()
                if get_rank() == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.3f} tok/s(global)={tok_s_global:,.0f}", flush=True)
                tokens_since = 0
                last = now
            step += 1

    # Save from rank 0
    if get_rank() == 0:
        # unwrap if DDP
        to_save = model.module if hasattr(model, 'module') else model
        torch.save(to_save.state_dict(), os.path.join(args.outdir, 'model.pt'))
        print('saved to', os.path.join(args.outdir, 'model.pt'))

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
