# dafab-summer-school25
Repo containing the material for the DaFAB summer school 

# Summer School Day2 Plan

# 3‑Hour Hands‑On: HPC Training & K8s Deployment of an LLM (Instructor Pack)

**Format:** 3h live lab (2 × 45 min blocks + 2 × 10 min breaks + optional 45 min challenge)

**What you get here:** 
- instructor notes,
- exact commands,
- minimal training script,
- SLURM job files,
- FastAPI inference server,
- Dockerfile,
- Kubernetes manifests.

Everything is copy‑pasteable and sized for quick demos, with optional stretch goals.

---

## High‑Level Schedule

| Time | Section | What you’ll do |
|---|---|---|
| 00:00–00:05 | HPC intro | Cluster overview, SLURM basics |
| 00:05–00:10 | Perf tools | `htop`, `nvidia-smi`, `iostat`, `ibstat`, `nvtop` |
| 00:10–00:20 | Simple run | Single‑GPU torch training via SLURM |
| 00:20–00:35 | Run at scale | Multi‑GPU / multi‑node with DDP (optional DeepSpeed) |
| 00:35–00:45 | Monitor & tune | Bottlenecks, batch/precision/parallelism |
| 00:45–00:55 | **Break** | — |
| 00:55–01:00 | K8s intro | Control plane, nodes, pods, services, ingress |
| 01:00–01:05 | Perf tools | `kubectl top`, `logs`, `describe`, dashboard/Lens |
| 01:05–01:15 | Simple run | Deploy FastAPI model server, port‑forward, test |
| 01:15–01:30 | Run at scale | Replicas + HPA; load test with `hey` |
| 01:30–01:40 | Monitor & tune | Startup, model load, batching, CPU/GPU |
| 01:40–01:50 | **Break** | — |
| 01:50–02:35 | (Optional) Challenge | End‑to‑end: package + push + autoscale |
| 02:35–02:45 | Wrap‑up | Best practices cheat‑sheet + Q&A |

---

## 0) Instructor Prep Checklist (do once)

**Accounts & access**
- [ ] HPC accounts working; students can `ssh` to login node.
- [ ] SLURM partitions visible (CPU/GPU), project/association set.
- [ ] Scratch path for students (e.g., `/scratch/$USER/llm-lab`).
- [ ] Modules or conda available for PyTorch.

**GPU & fabric**
- [ ] `nvidia-smi` reports driver + CUDA.
- [ ] IB device present (e.g., `ibstat` OK) if doing multi‑node.

**Kubernetes**
- [ ] Students have `kubectl` context against a test cluster/namespace.
- [ ] `metrics‑server` and (optionally) NGINX Ingress installed.
- [ ] If using GPUs: NVIDIA device plugin installed; nodes labeled.
- [ ] Docker/registry access for image push/pull.

**Files to pre‑stage (optional)**
- [ ] Provide this repo skeleton in `/shared/llm-lab` so students can copy.

```
llm-lab/
├─ hpc/
│  ├─ train.py
│  ├─ requirements.txt
│  ├─ sbatch_single.sh
│  ├─ sbatch_ddp.sh
│  └─ deepspeed_config.json (optional)
├─ k8s/
│  ├─ server/
│  │  ├─ app.py
│  │  ├─ model.py
│  │  └─ requirements.txt
│  ├─ Dockerfile
│  ├─ deploy.yaml
│  ├─ service.yaml
│  ├─ hpa.yaml
│  └─ ingress.yaml (optional)
└─ tools/
   └─ loadtest.sh
```

---

## Part 1 — HPC Training/Fine‑tuning on SLURM (45 min)

### 1.1 Intro to HPC (5 min) — speaking points
**Learning objectives**: Understand which node to use, how jobs are queued/placed, where data should live, and what knobs you control in SLURM.

**Whiteboard sketch**
- **Login node** → interactive, compilers, small tests; **no heavy compute**.
- **Compute nodes (CPU)** → preprocessing, dataloaders, CPU-bound jobs.
- **GPU nodes** → training/finetuning; may differ by GPU type (A100/H100/RTX...).
- **Interconnect** → IB/NVLink for fast all-reduce.
- **Storage tiers** → `$HOME` (quota, backed-up), `/scratch` or `/work` (large, fast, ephemeral), **local NVMe** (job-local).

**SLURM basics in 90 seconds**
- **Submit**: `sbatch script.sh`  • **Interactive**: `salloc` / `srun --pty bash`
- **Inspect**: `squeue -u $USER` (queue), `sacct -j <id>` (history), `sinfo -p <partition>` (capacity)
- **Cancel**: `scancel <id>` (job) or `scancel -u $USER` (all)
- **Key directives** in a script: `-p <partition>`, `--gpus=<n>` or `--gres=gpu:<type>:<n>`, `--nodes`, `--cpus-per-task`, `--mem`, `-t` (time), `-A` (account), `-J` (name)
- **Scheduling policies**: fairshare & QoS; short jobs often start sooner; GPU types are separate partitions.

**Data placement tips**
- Put datasets/checkpoints on **/scratch**; avoid `$HOME` hotspots.
- Prefer **local NVMe** for temporary shards (env var or symlink). Stage-in at job start, stage-out on completion.
- Use **`--mem=0`** to request all node RAM when needed (cluster dependent).

**Checkpointing policy**
- Save to `runs/$SLURM_JOB_ID/` so you can correlate artifacts/logs. Consider epoch/step checkpoints every N minutes, not every step.

---
### 1.2 Perf tools (5 min)
**CPU & memory**
- `htop` → press **H** to toggle threads, **F2** to add columns (MEM%, CPU core), **F6** to sort by CPU.
- `pidstat -urdh 1` → per‑process CPU, RSS, IO read/write, context switches.

**GPU**
- `nvidia-smi -l 1` → utilization, memory, power, temperature.
- `nvidia-smi dmon -s pucm -d 1` → per‑second Perf (Pwr, Util, Clk, Mem ctrl). Useful to see compute vs memory bound.
- `nvtop` (if available) → multi‑GPU live dashboard.

**Disk I/O**
- `iostat -xz 1` → `%util` (>=80% saturated), `r/s`, `w/s`, `await` (latency), `rkB/s`, `wkB/s`.

**Network / Fabric**
- `ibstat` / `ibdev2netdev` → check link is **Active** and speed (e.g., EDR/HDR/NDR).
- `nvidia-smi topo -m` → PCIe/NVLink topology to understand cross‑GPU bandwidth.

**Interpreting signals**
- **High GPU mem, low util** → input starvation (dataloader, I/O, small batch) or sync waits.
- **High `%util` on disk** → bump dataloader `num_workers`, cache to NVMe, pre‑shard.
- **GPU util oscillating** → synchronization stalls (DDP), small global batch.

**One‑liner dashboard**
```bash
watch -n 1 '\
  date; \
  echo "==== QUEUE ===="; squeue -u $USER | sed -n '1,10p'; \
  echo "==== GPU ===="; nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; \
  echo "==== IO ===="; iostat -xz 1 1 | tail -n +7 | head -n 5'
```

---
### 1.3 Minimal training job (10 min)
**Goal:** submit a tiny job, read logs, verify throughput metric (`tok/s`).

**Step‑by‑step**
1. **Stage lab**
```bash
mkdir -p ~/scratch/llm-lab && cd ~/scratch/llm-lab
# (if shared skeleton exists) cp -r /shared/llm-lab/hpc ./hpc
```
2. **Inspect script** `hpc/sbatch_single.sh` → fill `<project>`, `<gpu-partition>` (e.g., `a100`, `gpu`), optional `--constraint=a100`.
3. **Submit** `sbatch hpc/sbatch_single.sh`
4. **Watch**
```bash
squeue -u $USER
# After it starts, tail logs
tail -f hpc/logs/tiny-lm-single-<JOBID>.out
```
5. **Verify**
   - Look for lines like: `epoch=0 step=0 loss=... tok/s=...`
   - Expect GPU util >70% after a few steps; memory use < 2–4 GiB.
6. **Collect stats**
```bash
sacct -j <JOBID> --format=JobID,JobName,Elapsed,MaxRSS,State
seff <JOBID>
```

**Common quick fixes**
- Pending in queue → try shorter `-t 00:05:00`, different partition, or fewer GPUs.
- ImportError torch → ensure correct module/conda; try `python -c "import torch; print(torch.__version__)"`.
- `CUDA out of memory` → lower `--batch_size`, `--seq_len`, or dims.

**Optional enhancements**
- Toggle `--fp16` and compare `tok/s`.
- Increase `--num_workers` in DataLoader (in code) and watch IO utilization drop.

---
### 1.4 Run at Scale (15 min)
**Concepts (2 min)**
- **DDP**: each GPU = one process; gradients all‑reduced each step.
- **World size** = `nodes × gpus_per_node`; **rank** identifies process; **rendezvous** tells processes how to meet.

**Script anatomy (3 min)**
- `--nodes`, `--gpus-per-node`, `--ntasks-per-node` (often = GPUs per node)
- `RANK0_HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)`
- `torchrun --nnodes $SLURM_NNODES --nproc_per_node $SLURM_GPUS_PER_NODE --rdzv_endpoint $RANK0_HOST:29500`
- NCCL env for IB: `NCCL_SOCKET_IFNAME=ib0`, `NCCL_IB_HCA=mlx5`, `NCCL_DEBUG=INFO`

**Run (5 min)**
```bash
cd ~/scratch/llm-lab/hpc
sbatch sbatch_ddp.sh
# or for a quick interactive sanity (single node):
salloc -p <gpu-partition> --gpus=4 --cpus-per-task=16 -t 00:20:00
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 16 --fp16
```

**Live monitoring (3 min)**
```bash
watch -n1 nvidia-smi
sacct -j <JOBID> --format=JobID,Elapsed,AllocTRES%40,State
# Look for even GPU util across processes and stable tok/s increase vs single‑GPU
```

**If it hangs**
- Verify RANK0 host reachable (firewall), same CUDA/driver, clocks set.
- Mismatch in `--nproc_per_node` vs actual GPUs; check `nvidia-smi -L`.
- Force TCP for diagnosis: `export NCCL_IB_DISABLE=1` (slower but isolates IB issues).

**(Optional) DeepSpeed**
- Start with ZeRO‑1 (optimizer state sharding). Keep batch/grad‑accum small to stabilize demo.

---
### 1.5 Monitoring & Optimization (10 min)
**Bottleneck diagnosis checklist**
- **GPU idle** (>30% of time): raise `--batch_size`; enable `--fp16`; turn on persistent workers `DataLoader(..., persistent_workers=True)`; overlap host→device copy (`pin_memory=True`).
- **I/O bound**: `iostat` `%util` high + `await` high → shard dataset per node; pre-stage to NVMe; compress fewer, larger files.
- **Comm bound** (multi‑node): high variance in step time; NCCL INFO shows retries → ensure IB up; reduce gradient size via FP16; try gradient accumulation to increase compute/comm ratio.
- **Underutilized CPUs**: set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`; match dataloader workers to CPU cores reserved.

**Levers**
- **Batch size**: exponential growth until OOM; then step back 10–20%.
- **Precision**: AMP (FP16/BF16) typically 1.5–2.5× faster.
- **Parallelism**: data parallel first; pipeline/tensor parallel only for very large models.
- **Checkpoint freq**: too frequent → I/O stalls.

**Quick exercise**
- Double batch size; record `tok/s` and `loss` every 100 steps; discuss stability.

---

## Part 2 — Kubernetes Deployment (45 min) — Kubernetes Deployment (45 min)

### 2.1 Intro to K8s (5 min) — speaking points
**Learning objectives:** Map training artifact → container → running service with a stable endpoint and scaling knobs.

**Core components**
- **Control plane** (API server, scheduler, controller manager, etcd)
- **Nodes** (kubelet + container runtime)
- **Pods** (1+ containers) managed by **Deployments** (replicas + rollouts)
- **Service** (stable cluster IP / DNS) → **Ingress** (HTTP routing)

**Deployment options**
- Cloud (GKE/EKS/AKS) vs on‑prem (bare metal, OpenShift, Rancher). For on‑prem, LoadBalancer often needs MetalLB/ingress‑nginx.

**Container images**
- Use slim base, pin Python version, avoid dev tools in runtime; push to a registry accessible by the cluster.

**GPU on K8s** (if used)
- NVIDIA device plugin DaemonSet; request `resources.limits."nvidia.com/gpu": 1`; match nodes via labels `nvidia.com/gpu.present=true`.

---
### 2.2 Perf tools (5 min)
**Cluster view**
```bash
kubectl get nodes -o wide
kubectl get deploy,po,svc -n <ns>
```
**Resource usage**
```bash
kubectl top nodes
kubectl top pods -n <ns>
```
> If `top` fails, install/enable **metrics-server**.

**Debugging**
```bash
kubectl logs deploy/llm-infer -f
kubectl describe pod <pod>
kubectl get events --sort-by=.lastTimestamp | tail -n 20
kubectl exec -it deploy/llm-infer -- sh
```
**Rollouts**
```bash
kubectl rollout status deploy/llm-infer
kubectl rollout history deploy/llm-infer
```

**Interpretation**
- **CrashLoopBackOff** → app crashes before ready; read logs.
- **ImagePullBackOff** → registry secret or image tag error.
- **Pending** → no capacity or missing node selector/taints.

---
### 2.3 Simple run (10 min)
**Build & push** (replace registry/user)
```bash
cd k8s
docker build -t <registry>/<user>/llm-infer:lab .
docker push <registry>/<user>/llm-infer:lab
```

**Deploy**
```bash
kubectl apply -f deploy.yaml -f service.yaml
kubectl get pods -w
```
Wait for `READY 1/1` and `STATUS Running`.

**Test locally with port-forward**
```bash
kubectl port-forward svc/llm-svc 8080:80
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"tokens":[1,2,3,4],"steps":10}' | jq
```
Expected: `{"next_tokens":[...]} `

**Explain probes**
- **Readiness** gate = only receive traffic when model is ready.
- **Liveness** restarts container if stuck.

**Optional**: expose via NodePort/Ingress if available.

---
### 2.4 Run at Scale (15 min)
**Manual scale**
```bash
kubectl scale deploy/llm-infer --replicas=4
kubectl get pods -l app=llm-infer -w
```

**Autoscale (HPA)**
```bash
kubectl apply -f hpa.yaml
kubectl get hpa -w
```
> HPA targets CPU utilization average across pods; scale‑up reactions are faster than scale‑down by default.

**Load testing with hey**
```bash
# Terminal A: port-forward
kubectl port-forward svc/llm-svc 8080:80
# Terminal B: generate steady load
hey -z 60s -c 50 -q 5 -m POST \
  -H 'content-type: application/json' \
  -d '{"tokens":[1,2,3,4],"steps":10}' \
  http://localhost:8080/generate
```
**Interpret output**: `Requests/sec`, `Latency distribution`, `Status code distribution` (non‑2xx implies errors/backpressure).

**Observe**
```bash
kubectl top pods -l app=llm-infer
kubectl describe hpa llm-infer-hpa
```

**Notes**
- For GPU inference, add `resources.limits.nvidia.com/gpu: 1` and install NVIDIA plugin; one GPU per pod unless using MPS.
- Scale‑to‑zero requires event metrics (KEDA) or knative.

---
### 2.5 Monitor & Optimize (10 min)
**Bottlenecks**
- **Cold start**: large wheels; mitigate with multi‑stage builds, pre‑warmed nodes, lazy model load.
- **Throughput**: add **server workers** (e.g., `uvicorn --workers $(nproc)`), enable HTTP keep‑alive, batch small requests.
- **Latency**: avoid noisy neighbors; set `requests/limits` appropriately; use **PodAntiAffinity** to spread replicas.

**Patterns**
- **Batching**: accumulate requests for N ms, run one forward pass; expose `batch_max_ms` env.
- **Warmup**: run a few dummy requests on startup before declaring ready.
- **Readiness gate**: block traffic until model loaded (already in probes).

**GPU-specific**
- Pin model to GPU 0 in container (`CUDA_VISIBLE_DEVICES=0`), ensure drivers match; consider **Triton** or **vLLM** for tensor parallel/batching.

**Cost controls**
- Right-size CPU/mem requests to unlock bin‑packing; use **HPA minReplicas=0** with KEDA for idle hours.

---

## Part 3 — Optional End‑to‑End Challenge (45 min) — Optional End‑to‑End Challenge (45 min)

**Goal:** Ship the model trained in Part 1 into K8s and hit a throughput/latency target.

### 3.1 Integration (15 min)
**Option A — Bake model into image**
1. Copy from HPC:
```bash
scp <user>@<login>:/path/to/runs/<jobid>/model.pt k8s/model/
```
2. Update Dockerfile:
```dockerfile
COPY model/model.pt /models/model.pt
```
3. Build/push & roll out:
```bash
docker build -t <registry>/<user>/llm-infer:prod .
docker push <registry>/<user>/llm-infer:prod
kubectl set image deploy/llm-infer server=<registry>/<user>/llm-infer:prod
kubectl rollout status deploy/llm-infer
```

**Option B — Pull at startup (initContainer)**
- Store in S3/MinIO; use an initContainer to download to an **emptyDir** volume; main container reads `/models/model.pt`.

**Option C — PersistentVolume**
- Mount a ReadOnlyMany PVC with the artifact for all replicas.

**Versioning**
- Tag images by git SHA or date; annotate Deployment with model version label for traceability.

---
### 3.2 Performance Challenge (20 min)
**Targets** (tune to cluster size)
- **A:** P95 < 100 ms at 200 RPS
- **B:** Max RPS with P95 < 300 ms

**Allowed knobs**
- Replicas, CPU limits, HPA bounds, server workers, batching window, model dims (TinyLM params), quantization (toy), connection reuse (`hey -c` vs `-q`).

**How to measure**
1. Start port‑forward; run `hey` for 60–120s.
2. Record `Requests/sec`, `Average`, `90%`, `95%`, `99%` latencies.
3. Capture `kubectl top pods` and HPA decisions.

**Scoreboard template**
```
Team | Replicas | Workers/Pod | RPS | P95 (ms) | CPU/Pod | Notes
---- | -------- | ----------- | --- | -------- | ------- | -----
```

**Hints**
- Use **more workers** up to CPU core count.
- **Batching** yields big throughput gains at small latency tax.
- Ensure **readiness** blocks traffic until warmed up.

---
### 3.3 Wrap‑Up (10 min)
**Key takeaways**
- HPC: scale by adding GPUs/nodes; watch IO/comm to keep GPUs fed; AMP boosts perf.
- K8s: scale by replicas/HPA; manage cold starts and request batching; right-size resources.
- Artifacts: clear lineage from training run → image → deployment.

**Suggested next steps**
- Replace TinyLM with HF model; add real dataset and checkpointing.
- Introduce **vLLM/Triton/KServe** for production‑grade inference.
- Automate pipeline with CI/CD (build, scan, deploy) and GitOps.

**Q&A prompts**
- What was your bottleneck and how did you detect it?
- If you had 10× more traffic or 10× bigger model, what changes first?


---

# Appendix — Full Code Listings (Copy/Paste)

> This appendix gathers **complete** files referenced in the steps so you can copy them as‑is. Paths assume the skeleton layout from the top of the doc.

## A) HPC — Training & Scaling

### `hpc/train_ddp.py` — DDP‑aware tiny LM training (single‑GPU or multi‑GPU/multi‑node)
```python
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
```

### `hpc/sbatch_single.sh` — single GPU job (refined)
```bash
#!/usr/bin/env bash
#SBATCH -J tiny-lm-single
#SBATCH -A <project>
#SBATCH -p <gpu-partition>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# --- Environment (choose one stack) ---
# module purge; module load CUDA/12.2 Python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh; conda activate torch

python3 -m pip install --user -r requirements.txt --no-warn-script-location
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 train_ddp.py \
  --epochs 1 --batch_size 16 --fp16 \
  --outdir runs/$SLURM_JOB_ID
```

### `hpc/sbatch_ddp.sh` — 2 nodes × 4 GPUs example
```bash
#!/usr/bin/env bash
#SBATCH -J tiny-lm-ddp
#SBATCH -A <project>
#SBATCH -p <gpu-partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# module purge; module load CUDA/12.2 Python/3.10
python3 -m pip install --user -r requirements.txt --no-warn-script-location

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0

RANK0_HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun --label bash -lc '
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${RANK0_HOST}':29500 \
    train_ddp.py --epochs 1 --batch_size 16 --fp16 --outdir runs/$SLURM_JOB_ID
'
```

### `hpc/sbatch_deepspeed.sh` — (optional) DeepSpeed launcher
```bash
#!/usr/bin/env bash
#SBATCH -J tiny-lm-ds
#SBATCH -A <project>
#SBATCH -p <gpu-partition>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
python3 -m pip install --user -r requirements.txt --no-warn-script-location
python3 -m pip install --user deepspeed --no-warn-script-location

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun deepspeed --num_nodes=$SLURM_NNODES --num_gpus=$SLURM_GPUS_PER_NODE \
  train_ddp.py --epochs 1 --batch_size 16 --fp16 --outdir runs/$SLURM_JOB_ID \
  --deepspeed hpc/deepspeed_config.json
```

### `hpc/deepspeed_config.json`
```json
{
  "train_batch_size": 64,
  "fp16": {"enabled": true},
  "zero_optimization": {"stage": 1},
  "gradient_accumulation_steps": 1
}
```

### `hpc/requirements.txt`
```
torch
```

### `tools/gpu_watch.sh` — quick GPU/queue dashboard
```bash
#!/usr/bin/env bash
watch -n 1 '\
  date; \
  echo "==== QUEUE ===="; squeue -u $USER | sed -n '1,10p'; \
  echo "==== GPU ===="; nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; \
  echo "==== IO ===="; iostat -xz 1 1 | tail -n +7 | head -n 5'
```

---

## B) Kubernetes — Inference Service

### `k8s/server/tiny.py` — tiny model (inference‑friendly dims)
```python
import torch.nn as nn
class TinyLM(nn.Module):
    def __init__(self, vocab_size=4096, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.lm_head(self.encoder(self.emb(x)))
```

### `k8s/server/model.py` — wrapper with batch generate
```python
import os, torch
from .tiny import TinyLM

MODEL_PATH = os.environ.get('MODEL_PATH', '/models/model.pt')
VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', '4096'))

class ModelWrapper:
    def __init__(self):
        self.device = torch.device('cpu')  # set to cuda if GPU container
        self.model = TinyLM(vocab_size=VOCAB_SIZE).to(self.device)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, batch_tokens, steps=1):
        # batch_tokens: List[List[int]] with variable lengths
        max_len = max(len(t) for t in batch_tokens)
        import torch
        x = torch.full((len(batch_tokens), max_len), 0, dtype=torch.long)
        for i, t in enumerate(batch_tokens):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        x = x.to(self.device)
        logits = self.model(x)[:, -1, :]
        next_ids = logits.softmax(-1).argmax(-1).tolist()
        return [[n] for n in next_ids]
```

### `k8s/server/app_simple.py` — minimal FastAPI (single request)
```python
from fastapi import FastAPI
from pydantic import BaseModel
from .model import ModelWrapper

app = FastAPI()
model = ModelWrapper()

class Inp(BaseModel):
    tokens: list[int]
    steps: int = 1

@app.get('/healthz')
def health():
    return {'status': 'ok'}

@app.post('/generate')
def generate(inp: Inp):
    out = model.generate_batch([inp.tokens], inp.steps)
    return {'next_tokens': out[0]}
```

### `k8s/server/app_batched.py` — micro‑batching with asyncio (higher throughput)
```python
import asyncio, time
from fastapi import FastAPI
from pydantic import BaseModel
from .model import ModelWrapper

class Inp(BaseModel):
    tokens: list[int]
    steps: int = 1

app = FastAPI()
model = ModelWrapper()

BATCH_MAX_MS = float(os.getenv('BATCH_MAX_MS', '5')) / 1000.0
BATCH_MAX_SIZE = int(os.getenv('BATCH_MAX_SIZE', '32'))

queue = asyncio.Queue()

@app.on_event('startup')
async def _start_worker():
    async def worker():
        while True:
            reqs = []
            futs = []
            # Wait for at least one item
            item = await queue.get()
            reqs.append(item['tokens']); futs.append(item['fut'])
            t0 = time.perf_counter()
            # Accumulate within window/size
            while (time.perf_counter() - t0) < BATCH_MAX_MS and len(reqs) < BATCH_MAX_SIZE:
                try:
                    item = queue.get_nowait()
                    reqs.append(item['tokens']); futs.append(item['fut'])
                except asyncio.QueueEmpty:
                    await asyncio.sleep(BATCH_MAX_MS/10)
            # Run batch
            outs = model.generate_batch(reqs)
            for f, o in zip(futs, outs):
                f.set_result({'next_tokens': o})
    asyncio.create_task(worker())

@app.get('/healthz')
async def health():
    return {'status': 'ok'}

@app.post('/generate')
async def generate(inp: Inp):
    fut = asyncio.get_event_loop().create_future()
    await queue.put({'tokens': inp.tokens, 'fut': fut})
    return await fut
```

### `k8s/server/requirements.txt`
```
fastapi
uvicorn[standard]
torch
```

### `k8s/uvicorn_start.sh` — configurable workers
```bash
#!/usr/bin/env bash
set -euo pipefail
: "${PORT:=8000}"
: "${WORKERS:=1}"
exec uvicorn app_simple:app --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
```

### `k8s/Dockerfile` — CPU‑only image
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY server/ /app/
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT=8000 MODEL_PATH=/models/model.pt
EXPOSE 8000
CMD ["uvicorn", "app_simple:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `k8s/Dockerfile.gpu` — GPU image (if cluster has NVIDIA plugin)
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/ /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
ENV PORT=8000 MODEL_PATH=/models/model.pt
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app_simple:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `k8s/deploy.yaml` — CPU deployment (readiness/liveness probes)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-infer
spec:
  replicas: 1
  selector:
    matchLabels: { app: llm-infer }
  template:
    metadata:
      labels: { app: llm-infer }
    spec:
      containers:
      - name: server
        image: <registry>/<user>/llm-infer:lab
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "2"
        resources:
          requests: { cpu: "200m", memory: "256Mi" }
          limits:   { cpu: "1",    memory: "512Mi" }
        readinessProbe:
          httpGet: { path: /healthz, port: 8000 }
          initialDelaySeconds: 3
          periodSeconds: 5
        livenessProbe:
          httpGet: { path: /healthz, port: 8000 }
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
        - name: model
          mountPath: /models
      volumes:
      - name: model
        emptyDir: {}
```

### `k8s/deploy-gpu.yaml` — GPU deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-infer-gpu
spec:
  replicas: 1
  selector:
    matchLabels: { app: llm-infer-gpu }
  template:
    metadata:
      labels: { app: llm-infer-gpu }
    spec:
      containers:
      - name: server
        image: <registry>/<user>/llm-infer:gpu
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "1"
            memory: "2Gi"
          requests:
            nvidia.com/gpu: 1
            cpu: "200m"
            memory: "1Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_PATH
          value: "/models/model.pt"
        volumeMounts:
        - name: model
          mountPath: /models
      volumes:
      - name: model
        emptyDir: {}
```

### `k8s/service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-svc
spec:
  selector: { app: llm-infer }
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### `k8s/hpa.yaml` — CPU‑based HPA
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-infer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-infer
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

### `k8s/ingress.yaml` — (optional) Ingress example (nginx)
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "16m"
spec:
  ingressClassName: nginx
  rules:
  - host: llm.example.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-svc
            port:
              number: 80
```

### `k8s/init-s3.yaml` — fetch model from S3/MinIO at startup
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-infer-s3
spec:
  replicas: 1
  selector:
    matchLabels: { app: llm-infer-s3 }
  template:
    metadata:
      labels: { app: llm-infer-s3 }
    spec:
      initContainers:
      - name: fetch-model
        image: amazon/aws-cli:2.15.0
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom: { secretKeyRef: { name: s3-cred, key: accessKey } }
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom: { secretKeyRef: { name: s3-cred, key: secretKey } }
        - name: AWS_DEFAULT_REGION
          value: us-east-1
        command: ["sh", "-lc", "aws s3 cp s3://mybucket/model.pt /models/model.pt"]
        volumeMounts:
        - name: model
          mountPath: /models
      containers:
      - name: server
        image: <registry>/<user>/llm-infer:lab
        ports: [{containerPort: 8000}]
        volumeMounts:
        - name: model
          mountPath: /models
      volumes:
      - name: model
        emptyDir: {}
```

### `k8s/configmap.yaml` — tweak batching and workers
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
data:
  BATCH_MAX_MS: "5"
  BATCH_MAX_SIZE: "32"
  WORKERS: "2"
```

---

## C) Load Testing Utilities

### `tools/loadtest.sh` — shell wrapper around hey
```bash
#!/usr/bin/env bash
URL=${1:-http://localhost:8080/generate}
DUR=${2:-30s}
CONC=${3:-50}
QPS=${4:-5}
BODY='{"tokens":[1,2,3,4],"steps":1}'
hey -z "$DUR" -c "$CONC" -q "$QPS" -m POST -H 'content-type: application/json' -d "$BODY" "$URL"
```

### `tools/bench.py` — async Python load tester (no external deps beyond httpx)
```python
#!/usr/bin/env python3
import asyncio, time, json
import httpx

URL = 'http://localhost:8080/generate'
CONC = 100
DUR = 30
BODY = {"tokens": [1,2,3,4], "steps": 1}

async def worker(client, stop, results):
    while time.time() < stop:
        t0 = time.perf_counter()
        r = await client.post(URL, json=BODY)
        lat = (time.perf_counter() - t0)*1000
        results.append(lat)

async def main():
    stop = time.time() + DUR
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [worker(client, stop, results) for _ in range(CONC)]
        await asyncio.gather(*tasks)
    results.sort()
    def pct(p):
        return results[int(len(results)*p/100)] if results else 0
    print(json.dumps({
        'count': len(results),
        'avg_ms': sum(results)/len(results) if results else 0,
        'p50_ms': pct(50), 'p90_ms': pct(90), 'p95_ms': pct(95), 'p99_ms': pct(99)
    }, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
```

---

## D) Makefile (optional convenience)

### `Makefile` (at repo root)
```makefile
REG ?= <registry>/<user>
IMG ?= $(REG)/llm-infer:lab
NS  ?= default

.PHONY: build push deploy hpa logs pf scale clean

build:
	docker build -t $(IMG) k8s

push:
	docker push $(IMG)

deploy:
	kubectl apply -f k8s/deploy.yaml -f k8s/service.yaml

hpa:
	kubectl apply -f k8s/hpa.yaml

logs:
	kubectl logs -f deploy/llm-infer

pf:
	kubectl port-forward svc/llm-svc 8080:80

scale:
	kubectl scale deploy/llm-infer --replicas=4

clean:
	kubectl delete deploy/llm-infer svc/llm-svc hpa/llm-infer-hpa || true
```

---

**Usage recap per step**
- **Part 1.3** submit `sbatch_single.sh` → watch `logs/` and `tok/s`.
- **Part 1.4** submit `sbatch_ddp.sh` → confirm even GPU util & higher global `tok/s`.
- **Part 2.3** `make build push deploy && make pf` → curl health & generate.
- **Part 2.4** `make scale` or `make hpa` → run `tools/loadtest.sh` and observe.
- **Part 3.1** bake `model.pt` (or use `init-s3.yaml`) → `kubectl rollout restart`.



---

# RBAC for Students (Lens/K8s) — Reduced‑Permission Service Accounts

This section gives **ready‑to‑apply YAML** and **commands** to create a namespace‑scoped *student* ServiceAccount with least privilege. You can paste these in **Lens → (+) Create Resource** or apply via `kubectl`.

## 1) Pick or create a namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: workshop
```

Apply:
```bash
kubectl apply -f namespace.yaml
```

## 2) Create the ServiceAccount
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: student-sa
  namespace: workshop
```

## 3) Choose a permission preset

### A) **Read‑only viewer** (safe default)
Grants **get/list/watch** on common resources, plus logs; **no exec/port‑forward**, **no secrets**.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: student-view-role
  namespace: workshop
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "endpoints", "events"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
```

Bind it to the ServiceAccount:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: student-view-binding
  namespace: workshop
subjects:
- kind: ServiceAccount
  name: student-sa
  namespace: workshop
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: student-view-role
```

> **Shortcut:** Instead of creating a Role, you can bind the built‑in ClusterRole **`view`** to this namespace using a **RoleBinding** (limits it to the namespace):
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: student-view-binding
  namespace: workshop
subjects:
- kind: ServiceAccount
  name: student-sa
  namespace: workshop
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: view
```

### B) **Basic developer** (deploy & debug in namespace)
Allows CRUD on Deployments/Pods/Services/ConfigMaps, **logs + exec + port‑forward**, but **no secrets**.
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: student-dev-role
  namespace: workshop
rules:
# Core
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "endpoints", "events"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["pods/exec", "pods/portforward"]
  verbs: ["create"]
# Workloads
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

Bind it:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: student-dev-binding
  namespace: workshop
subjects:
- kind: ServiceAccount
  name: student-sa
  namespace: workshop
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: student-dev-role
```

> If you **don’t** want students to exec/port‑forward, remove the subresource rule for `pods/exec` and `pods/portforward`.

## 4) Generate a token & kubeconfig (Kubernetes ≥ 1.24)
Since auto‑generated SA secrets were removed in 1.24, use `kubectl create token`.

```bash
NS=workshop
SA=student-sa
# Get a short‑lived token (adjust duration)
TOKEN=$(kubectl -n $NS create token $SA --duration=24h)

# Grab cluster info from your current context
CLUSTER_NAME=$(kubectl config view --minify -o jsonpath='{.contexts[0].context.cluster}')
SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
CA_DATA=$(kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')

cat > student-sa.kubeconfig <<EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: ${CA_DATA}
    server: ${SERVER}
  name: ${CLUSTER_NAME}
contexts:
- context:
    cluster: ${CLUSTER_NAME}
    namespace: ${NS}
    user: ${SA}-${NS}
  name: ${SA}-${NS}@${CLUSTER_NAME}
current-context: ${SA}-${NS}@${CLUSTER_NAME}
users:
- name: ${SA}-${NS}
  user:
    token: ${TOKEN}
EOF
```

Now give students the `student-sa.kubeconfig` file to **Add Cluster** in **Lens**.

> **Pre‑1.24 clusters**: you can still create a Secret of type `kubernetes.io/service-account-token` linked to the SA and read the token from it.

## 5) Optional safety rails for the namespace

**ResourceQuota**: cap CPU/memory/objects so a class can’t overwhelm nodes.
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: workshop-quota
  namespace: workshop
spec:
  hard:
    requests.cpu: "2"
    requests.memory: 4Gi
    limits.cpu: "4"
    limits.memory: 8Gi
    pods: "20"
    services: "10"
    configmaps: "50"
```

**LimitRange**: set sane defaults & max for pods.
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: workshop-limits
  namespace: workshop
spec:
  limits:
  - type: Container
    default:
      cpu: "500m"
      memory: 512Mi
    defaultRequest:
      cpu: "200m"
      memory: 256Mi
    max:
      cpu: "2"
      memory: 2Gi
```

**NetworkPolicy**: block cross‑namespace traffic, allow DNS only (adjust to your CNI).
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-by-default
  namespace: workshop
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
  ingress: []
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system  # e.g., DNS/CoreDNS
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
```

## 6) Test the permissions
```bash
KUBECONFIG=student-sa.kubeconfig kubectl auth can-i list pods -n workshop
KUBECONFIG=student-sa.kubeconfig kubectl auth can-i get secrets -n workshop  # should be "no"
KUBECONFIG=student-sa.kubeconfig kubectl get deploy -n workshop
```

## 7) Multi‑team pattern
Create **one SA per team** and bind in their namespace(s).
```bash
for TEAM in team-a team-b; do
  kubectl create ns ${TEAM}
  kubectl -n ${TEAM} create sa student-sa
  kubectl -n ${TEAM} create rolebinding student-view --clusterrole=view \
    --serviceaccount=${TEAM}:student-sa
  # Optional: generate kubeconfig per team (reuse script above with NS=${TEAM})
done
```

**Cleanup**
```bash
kubectl delete rolebinding/student-view-binding role/student-view-role -n workshop
kubectl delete sa/student-sa -n workshop
kubectl delete ns workshop
```


