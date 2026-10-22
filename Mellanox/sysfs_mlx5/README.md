<a id="readme-top"></a>

<div align="center">

# InfiniBand Port Bandwidth Monitor

### Per-port InfiniBand throughput from the Linux RDMA sysfs counters

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/NVIDIA%20Mellanox-mlx5-76B900?logo=nvidia&logoColor=white)](#tested-environment)

A dependency-free Python monitor that turns the cumulative `port_xmit_data` /
`port_rcv_data` counters under `/sys/class/infiniband` into per-port receive and
transmit bandwidth and utilization, as analysis-ready CSV. No root and no
`infiniband-diags` are needed.

</div>

## About the tool

| Script | Purpose |
| --- | --- |
| `ib_port_bw_monitor.py` | Sample every active InfiniBand port at a fixed interval and write rates per port and direction |

It replaces the legacy Bash counter loop in
[`Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter`](../legacy/Mellanox_Infiniband_Throughput_Counter/),
which assumed device names and port counts and printed integer MiB/s.

- Ports are discovered by reading sysfs, so `mlx5_0`, `mlx5_4` or any other device naming works.
- Rates use the measured time between samples (sub-second intervals are fine; the reads are cheap).
- Utilization is against the port's own `rate` file (e.g. `400 Gb/sec (4X NDR)` gives 50 GB/s per
  direction), or a capacity you pass.
- A counter that goes backwards (reset or wrap) produces a row with `counter_status=reset_or_wrap` and blank rates.
- Inactive ports and Ethernet (RoCE) ports are skipped unless asked for.

### Tested environment

| Component | Version or configuration |
| --- | --- |
| Python | 3.9.25 and 3.10.12 (unit tests with a synthetic sysfs tree) |
| Hardware | **Not yet validated on a live HCA.** The first hardware check is planned on TACC Vista (ConnectX-7, NDR 400) |

Record the validation here when it is done (port list, a loopback or `ib_write_bw` run against the reported rate).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

```bash
cd Mellanox/sysfs_mlx5
python3 ib_port_bw_monitor.py --help
python3 ib_port_bw_monitor.py --list                  # ports found, state, capacity
python3 ib_port_bw_monitor.py -w 0.1 -W 10 --stdout   # 10 s at 100 ms
```

## Usage

```bash
# Every active InfiniBand port, 100 ms samples, until interrupted, into a timestamped file.
python3 ib_port_bw_monitor.py

# One device, 60 s, a chosen file.
python3 ib_port_bw_monitor.py -d mlx5_0 -W 60 -o runs/ib/node1_ib.csv

# Stop when a file appears (used by the Slurm job monitor).
python3 ib_port_bw_monitor.py -o node1_ib.csv --stop-file STOP

# Utilization against a different capacity (e.g. the effective data rate).
python3 ib_port_bw_monitor.py --capacity-gb-s 48
```

| Option | Meaning |
| --- | --- |
| `-d / --devices` | Devices to watch (default: all with counters) |
| `-w / --interval` | Target sampling interval in seconds (default 0.1) |
| `-W / --duration` | Stop after this many seconds |
| `--stop-file` | Stop when this file exists |
| `--capacity-gb-s` | Per-direction capacity used for utilization (default: `rate` / 8) |
| `--include-inactive`, `--include-ethernet` | Also watch ports that are down, or RoCE / Ethernet ports |
| `--sysfs-root` | Read another tree instead of `/sys/class/infiniband` (tests) |
| `--list` | Print the ports found and exit (exit status 1 if none) |
| `--stdout`, `--both`, `-o`, `--output-dir`, `--filename`, `--overwrite`, `--append` | Output handling, as in the NVIDIA toolkit |

## CSV reference

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp_epoch` / `timestamp_utc` | Midpoint of the current counter read | Unix epoch s / ISO 8601 UTC |
| `interval_seconds` | Measured time since the previous read | s |
| `device` / `port` | RDMA device name and port number | — |
| `link_layer` / `state` | From sysfs (`InfiniBand`, `4: ACTIVE`, ...) | — |
| `rate_gb_s_nominal` | The port's signalling rate from `rate` | Gb/s |
| `rx_counter` / `tx_counter` | Raw cumulative counters | 4-byte units |
| `rx_delta_bytes` / `tx_delta_bytes` | Data bytes in the interval (counter delta × 4) | B |
| `rx_gb_s` / `tx_gb_s` / `total_gb_s` | Rates | decimal GB/s |
| `per_direction_capacity_gb_s` | Capacity used for utilization | decimal GB/s |
| `rx_utilization_pct` / `tx_utilization_pct` / `bidirectional_utilization_pct` | Rate over capacity | % |
| `counter_status` | `ok` or `reset_or_wrap` | — |

## Units and counter semantics

`port_xmit_data` and `port_rcv_data` count data octets divided by 4 (one unit per four bytes,
whatever the lane count), on all virtual lanes, excluding link-level overhead. On mlx5 the kernel
exposes the 64-bit extended counters through these files. The nominal capacity is the signalling
rate divided by 8 bits; the achievable data rate is a few percent lower after encoding and headers.

## Known limitations

- The counters cover the whole port: traffic of other jobs on a shared node is included.
- One sample per interval gives the average over the interval; a burst shorter than the interval is
  spread over it. Use a short interval (0.05-0.1 s) for bursty traffic, or a profiler for the peak.
- RoCE ports report the same counters but are skipped by default.

## Tests

```bash
python3 -m unittest discover -s Mellanox/sysfs_mlx5/tests -v
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
