#!/bin/bash
# Example: link telemetry around each run of a multi-node Slurm job.
# One monitor per node (never one per rank), one markers file, a per-run summary at the end.
#SBATCH -J link-monitor-example
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:20:00

set -euo pipefail
PP=${PP:-$HOME/PerformanceProfiling}            # this repository (or 3rdParty/PerformanceProfiling)
M=$PP/Slurm/job_link_monitor
OUT=${OUT:-$PWD/run-$SLURM_JOB_ID}
TEL=$OUT/telemetry
mkdir -p "$TEL"

# 1. Start the per-node monitors (InfiniBand every 100 ms, NVLink every 0.5 s, nvidia-smi dmon every 1 s).
srun --ntasks-per-node=1 --overlap python3 "$M/node_link_monitor.py" \
     --output-dir "$TEL" --stop-file "$TEL/STOP" > "$TEL/monitors.log" 2>&1 &
MONITORS=$!
sleep 5   # let every node take its first samples

# 2. Wrap each run with start/end markers (labels: no commas).
for np in 1 2; do
  python3 "$M/run_marker.py" "$TEL/markers.csv" wrap "app-np$np" --meta nodes="$np" -- \
      srun -N "$np" --ntasks-per-node=1 ./my_app input.json > "$OUT/app-np$np.log" 2>&1
done

# 3. Stop the monitors and summarize per run.
touch "$TEL/STOP"
wait "$MONITORS" || true
python3 "$M/window_summary.py" --markers "$TEL/markers.csv" --data-dir "$TEL" -o "$OUT/link_summary.csv"
