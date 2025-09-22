#!/usr/bin/env bash
watch -n 1 '\
  date; \
  echo "==== QUEUE ===="; squeue -u $USER | sed -n '1,10p'; \
  echo "==== GPU ===="; nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; \
  echo "==== IO ===="; iostat -xz 1 1 | tail -n +7 | head -n 5'
