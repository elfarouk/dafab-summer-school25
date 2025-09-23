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
| 00:20–00:35 | Run at scale | Multi‑GPU / multi‑node with DDP |
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
│  
├─
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


### Step‑by‑Step Lab Guide (Ollama)
**Goal:** Stand up an Ollama inference service, pull a model, test locally, scale out, and monitor.

> You can do all YAML via **Lens → (+) Create Resource** or with `kubectl apply -f -`.

#### 0) Preconditions (2 min)
- Your `kubectl` context points to the right cluster/namespace.
- (GPU path) **NVIDIA device plugin** is installed and at least one worker has a GPU: `kubectl get ds -A | grep nvidia` and `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"	"}{.status.allocatable.nvidia\.com/gpu}{"
"}{end}'`.
- **metrics-server** is installed (for `kubectl top` / HPA).

#### 1) Create a namespace (1 min)
```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: Namespace
metadata:
  name: workshop
YAML
```

#### 2) Create storage for model cache (PVC) (1 min)
```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-cache-pvc
  namespace: workshop
spec:
  accessModes: [ "ReadWriteOnce" ]
  resources: { requests: { storage: 50Gi } }
  # storageClassName: <your-class>
YAML
```
Check it bound:
```bash
kubectl -n workshop get pvc ollama-cache-pvc
```

#### 3) Deploy Ollama (choose GPU **or** CPU) (3–5 min)
**GPU (preferred with big models):**
```bash
kubectl apply -f - <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-gpu
  namespace: workshop
spec:
  replicas: 1
  selector: { matchLabels: { app: ollama-gpu } }
  template:
    metadata: { labels: { app: ollama-gpu } }
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports: [{ containerPort: 11434 }]
        env: [{ name: OLLAMA_HOST, value: "0.0.0.0" }]
        lifecycle:
          postStart:
            exec: { command: ["sh","-lc","sleep 2; ollama pull llama3.1:8b"] }
        volumeMounts: [{ name: cache, mountPath: /root/.ollama }]
        resources:
          requests: { nvidia.com/gpu: 1, cpu: "500m", memory: 2Gi }
          limits:   { nvidia.com/gpu: 1, cpu: "2",    memory: 8Gi }
      volumes:
      - name: cache
        persistentVolumeClaim: { claimName: ollama-cache-pvc }
YAML
```
**CPU (smaller models/tests):**
```bash
kubectl apply -f - <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-cpu
  namespace: workshop
spec:
  replicas: 1
  selector: { matchLabels: { app: ollama-cpu } }
  template:
    metadata: { labels: { app: ollama-cpu } }
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports: [{ containerPort: 11434 }]
        env: [{ name: OLLAMA_HOST, value: "0.0.0.0" }]
        lifecycle:
          postStart:
            exec: { command: ["sh","-lc","sleep 2; ollama pull llama3.1:8b"] }
        volumeMounts: [{ name: cache, mountPath: /root/.ollama }]
        resources:
          requests: { cpu: "500m", memory: 2Gi }
          limits:   { cpu: "2",    memory: 8Gi }
      volumes:
      - name: cache
        persistentVolumeClaim: { claimName: ollama-cache-pvc }
YAML
```
Watch rollout:
```bash
kubectl -n workshop rollout status deploy/ollama-gpu || \
kubectl -n workshop rollout status deploy/ollama-cpu
```

#### 4) Create Service and port‑forward (2 min)
```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: Service
metadata:
  name: ollama
  namespace: workshop
spec:
  selector: { app: ollama-gpu }  # change to ollama-cpu if you used CPU
  ports:
  - name: http
    port: 11434
    targetPort: 11434
  type: ClusterIP
YAML
```
Port‑forward to your laptop:
```bash
kubectl -n workshop port-forward svc/ollama 11434:11434
```

#### 5) Test the API (2 min)
```bash
# Simple generate (streams JSON lines; final field is .response)
curl -s http://localhost:11434/api/generate \
  -H 'content-type: application/json' \
  -d '{"model":"llama3.1:8b","prompt":"Say hello from Kubernetes."}' | jq -r '.response'

# Chat style
curl -s http://localhost:11434/api/chat \
  -H 'content-type: application/json' \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Give me 3 Kubernetes objects."}]}' | jq
```
If the first call stalls, the model is still pulling—watch logs:
```bash
kubectl -n workshop logs -f deploy/ollama-gpu
```

#### 6) Scale out (3–5 min)
```bash
# Manual scale
kubectl -n workshop scale deploy/ollama-gpu --replicas=3

# Basic load test from your laptop (install hey: https://github.com/rakyll/hey)
hey -z 30s -c 20 -m POST -H 'content-type: application/json' \
  -d '{"model":"llama3.1:8b","prompt":"One fun fact."}' \
  http://localhost:11434/api/generate
```
> For real autoscaling, prefer queue-length based scalers (e.g., KEDA with a custom metric). CPU HPA is not a good signal for GPU loads.

#### 7) Monitor & debug (5 min)
```bash
# Resource usage (needs metrics-server)
kubectl -n workshop top pods
kubectl -n workshop top nodes

# Logs / events / describe
kubectl -n workshop logs -f deploy/ollama-gpu
kubectl -n workshop describe deploy/ollama-gpu
kubectl -n workshop get events --sort-by=.lastTimestamp | tail -n 20

# Exec a shell to inspect cache size
kubectl -n workshop exec -it deploy/ollama-gpu -- sh -lc 'du -sh /root/.ollama && ollama list'
```

#### 8) Optional: expose via Ingress (3 min)
```bash
kubectl apply -f - <<'YAML'
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ollama-ingress
  namespace: workshop
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "32m"
spec:
  ingressClassName: nginx
  rules:
  - host: ollama.example.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ollama
            port:
              number: 11434
YAML
```
Update DNS or `/etc/hosts` accordingly. Secure it (auth/mTLS) before exposing publicly.

#### 9) Cleanup (1 min)
```bash
kubectl delete ns workshop
```

#### Troubleshooting quick hits
- **Pod Pending**: insufficient GPU? Check `kubectl -n workshop describe pod` for `0/… nodes are available: … nvidia.com/gpu`. Switch to CPU deploy or pick a GPU node selector.
- **Model pull slow**: the first `ollama pull` downloads GBs. Keep the PVC to cache; later pods reuse it. Increase image pull bandwidth or pre‑bake models into an image if needed.
- **OOMKilled**: raise memory limit on the Deployment, use a smaller model, or switch to GPU variant.
- **`kubectl top` empty**: metrics-server missing; install it first.

---
