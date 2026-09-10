# Intel PCM 202604 monitoring on Sapphire Rapids

These dependency-free Python wrappers collect native CSV output from Intel
Performance Counter Monitor (PCM) 202604 on the local dual-socket Sapphire
Rapids system. They add consistent file naming, duration handling, diagnostics,
safe overwrite behavior, binary discovery, and clean signal forwarding.

The default binaries are detected in this order:

1. `PCM_BIN`, `PCM_PCIE_BIN`, or `PCM_IIO_BIN`, as appropriate
2. `/home/bagus/Dissagregated_PD/pcm_source/pcm/build-202604/bin`
3. the corresponding command on `PATH`

Use `--binary` to select another build explicitly. The CSV schema is deliberately
left native because PCM changes available columns according to CPU model,
topology, kernel support, access mode, and PCM release.

## Tools

| Script | Native tool | Best use |
| --- | --- | --- |
| `pcm_cpu_monitor.py` | `pcm` | CPU IPC/frequency/cache/memory metrics and per-link UPI traffic/utilization |
| `pcm_pcie_monitor.py` | `pcm-pcie` | Aggregate PCIe transaction classes and estimated bandwidth per CPU socket |
| `pcm_iio_monitor.py` | `pcm-iio` | PCIe bandwidth per IIO stack, root port, bus, and device; best choice for CPU-to-GPU attribution |
| `pcm_common.py` | — | Shared implementation; not run directly |

All scripts default to a one-second interval, run until interrupted, and create
a UTC-prefixed CSV plus a sibling `_stderr.log`. Existing outputs are protected
unless `--overwrite` is supplied.

## Quick start

```bash
cd /home/bagus/PerformanceProfiling/Intel/pcm202604_sapphirerapids

python3 pcm_cpu_monitor.py -W 60 --no-cores
python3 pcm_pcie_monitor.py -W 60
python3 pcm_iio_monitor.py -W 60  # requests sudo when MCFG is unreadable
```

`-W 60` is converted to enough complete PCM samples to cover approximately 60
seconds. Use `-i 60` when an exact number of samples matters. Without either
option, collection intentionally continues until Ctrl+C. `pcm-iio` may take
several seconds to initialize and write its first CSV rows. The wrapper reports
whether a partial CSV was saved when an open-ended capture is interrupted.

Choose explicit output paths when collecting beside a benchmark:

```bash
python3 pcm_cpu_monitor.py \
  -w 1 -W 300 --no-cores \
  -o runs/example/pcm_cpu.csv \
  --stderr-log runs/example/pcm_cpu_stderr.log
```

Preview the actual command and environment without starting PCM:

```bash
python3 pcm_iio_monitor.py -W 60 --dry-run
```

Each wrapper accepts repeatable native options for advanced use. Options that
begin with a dash must use the equals form:

```bash
python3 pcm_cpu_monitor.py --pcm-arg=-m=1 -W 10
```

## Which PCIe tool to use

`pcm-pcie` and `pcm-iio` observe different uncore PMUs and answer different
questions.

### Selection guide for this server

| Measurement goal | Use | Interpretation |
| --- | --- | --- |
| Per-GPU/root-port PCIe bandwidth | `pcm_iio_monitor.py` | Primary quantitative tool; H2D is normally `IB read`, D2H is normally `IB write` |
| UPI bandwidth and link utilization | `pcm_cpu_monitor.py --no-cores` | `dataIn` is incoming payload; `trafficOut` includes data and protocol traffic |
| Socket-level PCIe activity | `pcm_pcie_monitor.py` | Directional/debug cross-check, not the primary bandwidth value |
| PCIe and UPI for one workload | Separate `pcm-iio` and `pcm` passes | Safest comparison because each capture owns its PMUs and cleanup lifecycle |

Controlled TransferBench validation on this Xeon 8460Y+ measured approximately
55--56 GB/s with both TransferBench and `pcm-iio`. `pcm-pcie` reported only
about 19.2 GB/s (roughly 35% of the observed payload rate), although its
read/write direction was correct. Therefore use `pcm-iio` for numerical
CPU--GPU PCIe bandwidth on this host and retain `pcm-pcie` only as an optional
activity indicator.

Do not run `pcm` and `pcm-pcie` together: both program CHA/C-box counters on
Sapphire Rapids. Although `pcm-iio` and `pcm` target distinct IIO and UPI PMUs,
separately started PCM processes perform broad uncore cleanup when they exit.
Use isolated repeated workload passes when the last sample and reproducibility
matter.

### `pcm-pcie`

`pcm-pcie` counts PCIe-related transaction classes seen by each socket. With
the wrapper defaults, native `-B` multiplies transfers by 64 bytes to estimate
read/write byte volume and `-e` adds LLC total/miss/hit rows. Its output is
useful for a low-overhead, socket-level indication that PCIe traffic occurred.

Although PCM's help describes the `PCIe Rd (B)` and `PCIe Wr (B)` columns as
bytes per second, PCM 202604 calculates 64 times the transfers counted during
the sample. The values are therefore directly bytes/second only at the default
one-second interval. For another interval, divide each value by the interval in
seconds to obtain the rate.

It cannot identify which GPU or root port generated that traffic. Its bandwidth
can also be too high when many transfers contain less than one full 64-byte
cache line. PCM 202604 repeats its header for each sample and does not put a
timestamp in `pcm-pcie` CSV, so align it to other telemetry by sample order and
the requested interval.

The direction labels are from the PCIe device's DMA perspective:

- PCIe read: a device reads host memory, normally host-to-device payload flow.
- PCIe write: a device writes host memory, normally device-to-host payload flow.

### `pcm-iio`

`pcm-iio` reports bytes per second by socket, IIO stack, root port, PCI bus, and
device, and its CSV includes a timestamp. This is the recommended primary tool
for CPU-to-GPU PCIe bandwidth because GPU BDFs can be matched to the PCIe
topology. Root-port rows are enabled by default in this wrapper.

Create a static topology capture before a run:

```bash
python3 pcm_iio_monitor.py --list-topology -o pcm_iio_topology.csv
lspci -Dnn | grep -iE 'vga|display|3d'
```

For GPU bulk DMA, the most relevant IIO metrics are:

- `IB read`: the PCIe device requested reads from host memory (typically H2D).
- `IB write`: the PCIe device requested writes to host memory (typically D2H).
- `OB read/write`: CPU MMIO accesses to the device, usually control traffic
  rather than the large DMA payload.

The exact H2D/D2H interpretation should still be validated with a one-direction
copy workload because peer-to-peer DMA, IOMMU behavior, and platform routing can
change which counters observe a transfer.

## UPI utilization

Use `pcm_cpu_monitor.py` (native `pcm`) for UPI. On a supported multi-socket
system, its socket/system CSV contains incoming and outgoing traffic for each
UPI link and utilization fields. `--no-cores` is convenient when the goal is
only socket, memory, and UPI behavior; it makes the very wide CSV much smaller.

Neither `pcm-pcie` nor `pcm-iio` measures UPI. A GPU attached to one socket can
cause both PCIe/IIO traffic and UPI traffic if its buffers or submitting CPU are
on the other NUMA node, so collect `pcm-iio` and `pcm` over the same benchmark
window when investigating cross-socket GPU access. `pcm-pcie` is optional as a
socket-level cross-check.

## Permissions and troubleshooting

The wrappers default to:

```text
PCM_NO_MSR=1
PCM_KEEP_NMI_WATCHDOG=1
```

This requests Linux `perf_event` operation and leaves the NMI watchdog enabled.
The host generally needs `/proc/sys/kernel/perf_event_paranoid=-1` for
unprivileged collection. Use `--direct-msr` only when direct MSR and PCI
configuration access has deliberately been provided.

`pcm-iio` additionally needs PCI topology information, including access to the
ACPI MCFG table on this platform. When the wrapper detects that MCFG is not
readable by the current account, it launches only the native `pcm-iio` command
through an interactive sudo process. Sudo prompts and PCM errors are relayed to
the terminal while also being retained in the diagnostic log. New CSV output
is pre-created by the wrapper so it and the diagnostic log remain owned by the
calling account. A cached sudo credential does not prompt again. Use
`--no-sudo` to disable this behavior or `--sudo` to request it even when MCFG
appears readable.

The elevated command receives only the two PCM access variables explicitly;
other environment overrides are not preserved by default. An empty CSV with a
nonzero exit is treated as an error by the wrapper.

PCM 202604 repeatedly reports unmapped internal CPU-bus IDs and nonexistent
Sapphire Rapids stacks 10/11 while it cycles through IIO events. Those known
non-fatal lines are suppressed from both the terminal and diagnostic log by
default, then replaced with one summary containing the number suppressed. All
other diagnostics remain unchanged. Use `--show-topology-warnings` to retain
the original messages when investigating PCM or kernel topology support.

Only one process should own a given PMU event set. If concurrent PCM programs
report busy counters, permission errors, or implausible zeros, collect them in
separate validation runs first. Always inspect the diagnostic logs and compare
idle traffic with a controlled H2D/D2H or NUMA-crossing workload before relying
on the counters in an experiment.

## Controlled TransferBench validation

`validate_pcm_transferbench.py` automates that comparison with the TransferBench
build at
`/home/bagus/Dissagregated_PD/TransferBench_Source/TransferBench/TransferBench`.
It runs six sequential cases: H2D and D2H under `pcm-pcie`, H2D and D2H under
`pcm-iio`, and CPU-local and CPU-cross-socket copies under `pcm`. The sequential
schedule prevents the PCM utilities from competing for uncore counters.

```bash
cd /home/bagus/PerformanceProfiling/Intel/pcm202604_sapphirerapids
python3 validate_pcm_transferbench.py
```

The default case lasts 10 seconds, transfers 256 MiB repeatedly, uses GPU 0 and
CPU NUMA node 0 for PCIe, and compares NUMA node 0 with NUMA node 1 for UPI.
The runner asks for sudo once before its IIO cases and stores every CSV and log
under `runs/<UTC>_transferbench_pcm_validation/`. Preview or select cases with:

```bash
python3 validate_pcm_transferbench.py --dry-run
python3 validate_pcm_transferbench.py --cases pcie_h2d iio_h2d upi_cross
python3 validate_pcm_transferbench.py --gpu 4 --cpu-node 1
```

For an attached GPU, use its local CPU NUMA node for the initial H2D/D2H test;
then deliberately select the other CPU node in a separate run if cross-socket
GPU access is the behavior of interest. The local CPU case is a baseline, while
`upi_cross` should produce the strong UPI signal. TransferBench requires access
to `/dev/kfd` and the GPU render devices, so run the validation from the normal
host login shell rather than a device-isolated container.

## Full CPU-to-GPU matrix

`validate_pcm_cpu_gpu_matrix.py` tests H2D and D2H between CPU NUMA nodes 0 and
1 and GPUs 0 through 7. For each of the 32 paths it repeats the transfer under
`pcm-iio` for PCIe and under `pcm` for UPI, producing 64 isolated captures.
`pcm-pcie` is intentionally excluded based on the selection guidance above.

```bash
python3 validate_pcm_cpu_gpu_matrix.py
```

The default 10-second workload run can take roughly 20 minutes. Results are
organized under `runs/<UTC>_cpu_gpu_matrix/cpuN_gpuN/`, with a matrix manifest,
TransferBench topology, per-transfer benchmark logs, native PCM CSVs, and PCM
diagnostic logs. Preview the matrix or run a shorter subset with:

```bash
python3 validate_pcm_cpu_gpu_matrix.py --dry-run
python3 validate_pcm_cpu_gpu_matrix.py --duration 3 --gpus 0 4
```

### Validation: full CPU-to-GPU matrix (2026-09-10)

The complete 10-second matrix in
`runs/20260910T220632Z_cpu_gpu_matrix` finished all 64 isolated captures with
status 0. GPUs 0--3 are local to CPU NUMA node 0 and GPUs 4--7 are local to CPU
NUMA node 1. All bandwidth values below are decimal GB/s.

The theoretical local-path peak is 63.0 GB/s per direction: PCIe 5.0 x16 at 32
GT/s, adjusted for 128b/130b encoding. For a remote path, the configured
three-link UPI connection is the tighter limit at approximately 54 GB/s per
direction, based on the link speeds detected by PCM during this run. This is
the server's configured operating peak, not the Sapphire Rapids architectural
maximum.

TransferBench (`TB`) is the mean of the two independent 10-second repetitions
used for IIO and UPI collection. PCM columns contain `median (observed peak)`:
IIO uses root-port `IB read` for H2D and `IB write` for D2H; UPI uses system
`TotalUPIin`, which counts incoming payload rather than total bidirectional
protocol traffic. The median is taken from the eight highest one-second samples
to represent the active workload plateau. `Peak UPI in` is the highest incoming
data-link utilization observed on either socket.

| Path | Route | Theoretical path peak | TB H2D | IIO H2D | UPI H2D | TB D2H | IIO D2H | UPI D2H | Peak UPI in |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU0–GPU0 | local | 63.0 | 54.95 | 56.03 (57.70) | 0.03 (0.11) | 56.37 | 56.44 (56.46) | 0.04 (0.10) | 0% |
| CPU0–GPU1 | local | 63.0 | 54.89 | 57.23 (58.03) | 0.04 (0.08) | 56.36 | 54.63 (56.44) | 0.03 (0.08) | 0% |
| CPU0–GPU2 | local | 63.0 | 54.94 | 56.28 (58.07) | 0.03 (0.09) | 56.36 | 56.38 (56.44) | 0.03 (0.08) | 0% |
| CPU0–GPU3 | local | 63.0 | 54.91 | 55.60 (57.85) | 0.04 (0.08) | 56.38 | 56.44 (56.48) | 0.03 (0.06) | 0% |
| CPU0–GPU4 | remote | ~54 | 54.28 | 52.11 (52.82) | 25.06 (25.35) | 52.68 | 51.36 (52.38) | 25.04 (26.30) | 52% |
| CPU0–GPU5 | remote | ~54 | 54.52 | 52.25 (55.47) | 25.06 (25.26) | 52.66 | 51.91 (53.56) | 24.75 (25.02) | 48% |
| CPU0–GPU6 | remote | ~54 | 54.48 | 52.38 (57.37) | 25.06 (25.31) | 52.60 | 51.21 (52.05) | 24.20 (24.77) | 49% |
| CPU0–GPU7 | remote | ~54 | 54.54 | 54.60 (57.43) | 25.01 (25.34) | 52.25 | 52.34 (52.64) | 23.60 (23.98) | 49% |
| CPU1–GPU0 | remote | ~54 | 54.26 | 54.65 (57.01) | 26.80 (27.00) | 53.14 | 53.14 (53.24) | 26.32 (26.66) | 51% |
| CPU1–GPU1 | remote | ~54 | 54.36 | 54.69 (57.57) | 26.84 (27.14) | 53.75 | 53.45 (53.75) | 26.72 (26.94) | 52% |
| CPU1–GPU2 | remote | ~54 | 54.56 | 55.12 (57.37) | 26.99 (27.11) | 54.22 | 54.04 (56.44) | 26.87 (27.04) | 53% |
| CPU1–GPU3 | remote | ~54 | 54.25 | 55.84 (57.47) | 26.85 (27.09) | 54.21 | 53.78 (55.21) | 26.88 (27.21) | 51% |
| CPU1–GPU4 | local | 63.0 | 55.00 | 51.98 (54.77) | 0.07 (0.64) | 56.43 | 56.26 (57.63) | 0.06 (0.67) | 0% |
| CPU1–GPU5 | local | 63.0 | 55.00 | 52.59 (55.70) | 0.05 (0.63) | 56.43 | 54.21 (54.96) | 0.06 (0.08) | 0% |
| CPU1–GPU6 | local | 63.0 | 54.97 | 52.02 (53.04) | 0.04 (0.62) | 56.39 | 54.45 (55.22) | 0.05 (0.61) | 0% |
| CPU1–GPU7 | local | 63.0 | 54.99 | 56.05 (58.12) | 0.07 (0.60) | 56.42 | 56.25 (56.33) | 0.04 (0.59) | 0% |

The local TransferBench averages are 54.96 GB/s H2D and 56.39 GB/s D2H.
Remote averages are 54.41 GB/s H2D and 53.19 GB/s D2H. Local transfers show
only background incoming UPI traffic, while every remote path shows the
expected 23.6--27.2 GB/s incoming UPI payload and 48--53% peak incoming-link
utilization. `TotalUPIin` should not be compared one-to-one with TransferBench:
it counts one UPI payload direction, whereas the transfer rate and UPI protocol
accounting have different semantics. IIO is the appropriate direct comparison
to TransferBench PCIe throughput.
