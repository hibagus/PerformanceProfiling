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
python3 pcm_iio_monitor.py -W 60
```

`-W 60` is converted to enough complete PCM samples to cover approximately 60
seconds. Use `-i 60` when an exact number of samples matters. Without either
option, press Ctrl+C to stop collection and let PCM flush its output.

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
ACPI MCFG table on this platform. If its diagnostic log says it cannot open
`/sys/firmware/acpi/tables/MCFG`, run under an appropriately privileged service
or account and follow the site's security policy. An empty CSV with a nonzero
exit is treated as an error by the wrapper.

Only one process should own a given PMU event set. If concurrent PCM programs
report busy counters, permission errors, or implausible zeros, collect them in
separate validation runs first. Always inspect the diagnostic logs and compare
idle traffic with a controlled H2D/D2H or NUMA-crossing workload before relying
on the counters in an experiment.
