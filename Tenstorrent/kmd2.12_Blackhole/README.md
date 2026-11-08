<a id="readme-top"></a>

<div align="center">

# Tenstorrent Blackhole Monitoring Toolkit

### Power, clocks, PCIe and Ethernet telemetry of p150 cards, without touching the workload

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Tenstorrent-Blackhole%20p150-6F2DA8)](#tested-environment)
[![Driver](https://img.shields.io/badge/tt--kmd-2.12-555)](#requirements)

Dependency-free Python monitors that turn what the Tenstorrent driver and firmware already keep (sysfs
telemetry, the PCIe tile's traffic counters, the firmware's telemetry table and the Ethernet MAC counters) into
analysis-ready CSV, without instrumenting kernels, without firmware messages and without changing the cards'
power state.

</div>

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-toolkit">About the toolkit</a></li>
    <li><a href="#how-non-invasive-is-each-source">How non-invasive is each source</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#csv-fields">CSV fields</a></li>
    <li><a href="#counter-semantics-and-units">Counter semantics and units</a></li>
    <li><a href="#validation">Validation</a></li>
    <li><a href="#limitations">Limitations</a></li>
    <li><a href="#tests">Tests</a></li>
  </ol>
</details>

## About the toolkit

| Script | Reads | Purpose |
| --- | --- | --- |
| `tt_card_monitor.py` | sysfs only | Per card: AICLK/AXICLK/ARCCLK, ASIC power, current, VCORE, temperature and their limits, heartbeat, the processes holding the card, and PCIe bandwidth by direction and initiator |
| `tt_arc_telemetry_monitor.py` | firmware telemetry table (ARC tile) | Every firmware telemetry tag: board 12 V input power, which arbiter sets AICLK (TDP, TDC, thermal, ...), GDDR temperatures and IO power, Ethernet link and heartbeat masks, kernel-throttler counters, limits |
| `tt_eth_bw_monitor.py` | Ethernet tiles' base-firmware status | Per link: TX/RX bandwidth and utilization, frames, bad-FCS and dropped frames, corrected and uncorrected FEC codewords, retrains, queue resends; the peer card and tile of each link |
| `validate_pcie_counters.py` | sysfs, plus TT-NN transfers | Checks the PCIe counters against host-device copies of known size (needs tt-metal) |
| `tt_common.py` | | Shared code: card discovery, the power-neutral NoC reader, the telemetry-table and Ethernet decoders |

All monitors share the options of the other toolkits in this repository: `-c/--cards` (index `0`, PCI bus
`3f` or BDF), `-w/--interval`, `-W/--duration`, `--stop-file`, `--stdout`/`--both`/`-o`/`--output-dir`,
`--append`/`--overwrite`, and `--list`.

### Tested environment

| Component | Version or configuration |
| --- | --- |
| Hardware | Dell PowerEdge R760xa, 4 × Tenstorrent p150b (Blackhole, 140 Tensix), Gen5 x16; the cards form a ring with 4 × 400G Ethernet links per hop |
| Software | Ubuntu 24.04, kernel 6.8, tt-kmd 2.12.1 (2.12.0 or newer needed), firmware bundle 19.17.1 (CMFW 0.39.1, ETH FW 1.13.1), Python 3.12 (unit tests also on 3.9 syntax), tt-metal v0.81.1 for the validation workloads |
| Checks | PCIe counters against TT-NN transfers (fast and slow dispatch), Ethernet counters against tt-metal fabric traffic, neutrality during a 10-minute idle record; see [Validation](#validation) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How non-invasive is each source

Blackhole cards can be read at four levels of contact. The toolkit uses the first two only.

| Level | Source | What it touches | Used by |
| --- | --- | --- | --- |
| 0 | sysfs files of tt-kmd: `hwmon` (power, current, VCORE, temperature, limits), `tt_aiclk` and friends, `pcie_perf_counters/` | Never opens `/dev/tenstorrent`. The driver reads 9 firmware telemetry words from ARC memory, and the PCIe counters straight from a register of the PCIe tile mapped in BAR0 | `tt_card_monitor.py` |
| 1 | `TENSTORRENT_IOCTL_NOC_READ` on a power-neutral file descriptor | One 4- or 8-byte NoC read per call, done by the kernel through its own window. Only the ARC tile and enabled Ethernet tiles are read | `tt_arc_telemetry_monitor.py`, `tt_eth_bw_monitor.py` |
| 2 | tt-umd / pyluwen / tt-smi | Opens the card as a regular client: tt-smi and luwen open without a power request but send ARC messages; tt-umd requests full power (AICLK, Tensix, GDDR, L2CPU) for as long as it is open | not used |
| 3 | tt-metal device profiler, Tracy, fabric bandwidth telemetry (`TT_METAL_FABRIC_TELEMETRY`), watcher | Instruments kernels or the fabric router, changes timing and L1 use | not used |

**Why level 1 is safe and neutral (tt-kmd 2.12 and firmware 19.x source):**

* The device file is opened with `O_APPEND`, which makes the descriptor *power-aware* with no power flags.
  On open, tt-kmd sends nothing to the firmware for a power-aware descriptor; on close it returns early for a
  descriptor with no power contribution; and while it is open it adds nothing to the OR of all descriptors' power
  requests. The card's power state is exactly what it would be without the monitor. (Blackhole does not use the
  driver's deferred idle power-down, so the monitor's descriptor does not change power-down timing either.)
* No `SET_POWER_STATE`, no ARC message, no write. `O_NONBLOCK` makes the open fail instead of waiting if another
  process holds the card exclusively (for example a reset tool).
* The firmware clock-gates only Tensix, GDDR (MRISC) and L2CPU tiles (`bh_power.c`); a host read of a gated tile
  can take the card off the bus. The monitors read only the ARC tile (telemetry, which the driver itself reads for
  sysfs) and the Ethernet tiles listed as enabled in the firmware's `ENABLED_ETH` tag, whose base firmware runs at
  all times. Before decoding an Ethernet tile, its own chip-info block must name the card's board ID and the
  expected tile.
* The kernel's NoC window survives a card reset: the descriptor then returns `ENODEV`, and the ARC monitor reopens
  the card on the next sample.

Cost: one ARC sample is about 45 ioctls (8-byte reads of the 80-word table); one Ethernet snapshot is about 30
ioctls per link, and between snapshots the monitor reads 2 words per link to detect a new one.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Requirements

* Linux with tt-kmd **2.12.0 or newer** (`NOC_READ` ioctl and `pcie_perf_counters`); the sysfs card monitor
  works with older drivers except for the PCIe columns.
* Read access to `/sys/class/tenstorrent` (default) and read-write access to `/dev/tenstorrent/N` for the two
  NoC-reading monitors (the udev rule shipped with tt-kmd makes them 0666). No root.
* Python 3.9 or newer, standard library only. `validate_pcie_counters.py` needs tt-metal's Python environment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

```bash
cd Tenstorrent/kmd2.12_Blackhole

# The cards, and a one-shot dump of every firmware telemetry tag
python3 tt_card_monitor.py --list
python3 tt_arc_telemetry_monitor.py --dump -c 0

# Power, clocks and PCIe traffic of all cards at 10 Hz, next to a workload
python3 tt_card_monitor.py -w 0.1 --output-dir runs/ &
MON=$!; ./my_workload; kill -INT $MON

# The full telemetry table at 10 Hz (the firmware refreshes it every 100 ms)
python3 tt_arc_telemetry_monitor.py -w 0.1 --all-tags --output-dir runs/

# The Ethernet links: topology and totals, then per-link bandwidth for 60 s
python3 tt_eth_bw_monitor.py --links
python3 tt_eth_bw_monitor.py -w 0.1 -W 60 --output-dir runs/
```

`--stop-file PATH` ends a monitor when the file appears, which suits batch scripts that start the monitors
before a job and stop them after it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## CSV fields

### `tt_card_monitor.py` (one row per card and sample)

| Field | Meaning |
| --- | --- |
| `timestamp_epoch`, `timestamp_utc`, `interval_seconds` | Midpoint of the card's reads; time since the card's previous sample |
| `card`, `bdf`, `alive`, `holders` | `index:bus`, PCI address, 0 if the card reads 0xFFFFFFFF (lost), `pid:comm` of processes with the card open (refreshed every `--holders-interval` s) |
| `aiclk_mhz`, `axiclk_mhz`, `arcclk_mhz`, `heartbeat`, `therm_trip_count` | Driver sysfs files |
| `power_w`, `current_a`, `vcore_mv`, `asic_temp_c`, `fan_rpm` | hwmon: ASIC core-rail power, current and voltage, ASIC temperature |
| `power_limit_w`, `current_limit_a`, `vcore_max_mv`, `temp_throttle_c` | hwmon `*_max`: the firmware's TDP, TDC, VDD maximum and thermal throttle limit |
| `mst_wr_gb_s`, `mst_rd_gb_s` | Host-initiated (MMIO through a TLB window) writes into and reads from the card, at `--mst-flit-bytes` per count |
| `slv_rd_gb_s`, `slv_wr_gb_s` | Card-initiated reads of and writes to host memory (DMA), 64 B per count |
| `h2d_gb_s`, `d2h_gb_s` | `mst_wr + slv_rd` and `mst_rd + slv_wr` |
| `pcie_status` | `ok`, `ok_long_interval_may_miss_wraps` (interval >= 2 s), `card_dead`, `unavailable` (older driver) |
| `mst_posted_wr_data_word_sent0` ... | The 12 raw cumulative counters (6 kinds × NoC 0/1), for re-analysis |

### `tt_arc_telemetry_monitor.py` (one row per card and sample)

The default columns are the dynamic tags and the limits; `--all-tags` writes one column per published tag
(identifiers, firmware versions, enable masks too). Decoded fields of note:

| Field | Meaning |
| --- | --- |
| `fresh`, `status` | 1 if the firmware's timer heartbeat advanced since the previous sample (the table refreshes every `telem_update_ms`, 100 ms); `ok`, `card_dead` or `error:<errno>` |
| `power_w`, `current_a`, `vcore_mv` | ASIC core rail, whole W and A (the firmware truncates) |
| `input_power_w` | Board input power in W as the firmware uses it for its board-power limit: the board management controller's (DMC) reading plus a fixed `additional_board_power` from the firmware table (+20 W on P150). Checked against iDRAC node power on the tested server: loading all four cards raised the node by 920 W (AC, net of host RAPL) while the four cards' tags rose by 910 W (DC), within about 5 %. The tag covers the whole card: about 2x the ASIC rail under load (about 315 W per p150b under a BFP8 GEMM; 87 W idle) and ~0.5 s transients up to about 575 W at load steps, which the iDRAC (one update per ~25 s) cannot resolve |
| `aiclk_arb_min`, `aiclk_arb_max` | `MHz:arbiter` of the effective minimum (`fmin`, `busy`) and maximum (`fmax`, `tdp`, `fast_tdc`, `tdc`, `thm`, `board_power`, `voltage`, `gddr_thm`, `doppler_slow`, `doppler_critical`, `host_fmax`) clock arbiters |
| `aiclk_ppm_info`, `limiter` | Why AICLK is what it is, e.g. `max_arb:tdc` = the TDC limit holds the clock down; `min_arb:busy` = the busy request sets it |
| `enabled_min_arb`, `enabled_max_arb` | Which arbiters are active |
| `nop_start_count`, `nop_on_ms`, `kernel_throttler` | The firmware's kernel-NOP throttler: starts since boot, NOP time in the last update window, configuration |
| `gddr0_temp_c` ... `gddr7_temp_c`, `max_gddr_temp_c`, `gddr_west_io_power_w`, `gddr_east_io_power_w`, `gddr_corr_errs`, `gddr_uncorr_errs` | Per-module GDDR temperature (hotter of the two dies), GDDR IO rail power, EDC error counts (8-bit per module and direction) |
| `eth_heartbeat_mask`, `eth_link_mask`, `eth_links_up` | Ethernet tiles whose base-firmware heartbeat runs, links up (bit = physical tile) |
| `vreg_temp_c`, `board_temp_c` | Always 0 on firmware 19.x ("needs I2C line" in the firmware source) |

### `tt_eth_bw_monitor.py` (one row per link and firmware snapshot)

| Field | Meaning |
| --- | --- |
| `card`, `channel`, `tile`, `peer` | Card, translated channel (`x = 20 + channel, y = 25`), physical tile, and the peer `card/tile` (or board ID if the peer is not in this host) |
| `port_status`, `speed_gbps` | Base firmware's port status and trained speed (400 on p150) |
| `heartbeat`, `owner` | `eth_status.heartbeat[0..1]`; `base` if the base firmware's heartbeat advanced, `metal` if tt-metal's runtime firmware's did, `stalled` if neither |
| `fw_stamp`, `fw_clock_hz`, `dt_fw_s`, `dt_host_s`, `dt_s` | Base-firmware snapshot timestamp and its clock (`--eth-clock-hz`, 50 MHz; 0 calibrates against the host); the interval by each and the one used for the rates |
| `tx_bytes`, `rx_bytes`, `tx_gb_s`, `rx_gb_s`, `tx_util_pct`, `rx_util_pct` | MAC bytes in the interval, decimal GB/s, percent of `speed_gbps` |
| `tx_frames`, `rx_frames`, `tx_badfcs_frames`, `rx_badfcs_frames`, `rx_dropped_frames` | Frame counts in the interval |
| `corr_cw_per_s`, `uncorr_cw` | Corrected FEC codewords per second, uncorrected codewords in the interval |
| `retrains`, `link_flaps`, `txq_resends`, `rxq_drops`, `rx_link_up` | Link events in the interval |
| `counter_status` | `ok`, `stale` (no new snapshot for `--stale-after` s), `reset` (a counter went backwards), `error:<errno>` |
| `*_total` | Cumulative bytes and FEC counts since the link trained |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Counter semantics and units

**PCIe.** `pcie_perf_counters/*{0,1}` are the PCIe tile's NIU counters for NoC 0 and NoC 1, 32-bit and
cumulative from the last reset. They count NoC data flits, not bytes. tt-kmd's `docs/sysfs-attributes.md` says
32-byte flits, which fits Wormhole's 256-bit NoC. Measured on Blackhole with `validate_pcie_counters.py`:
* Card-initiated DMA fills the 64-byte flits: **64 B per count** (`slv_*`).
* Host MMIO through an uncached TLB window carries 32 B per access: **32 B per count** (`mst_*`). This is the
  `--mst-flit-bytes` default. With write-combined 64-byte writes it would be 64.

From the NoC side of the
PCIe tile, `mst_*` counts requests that entered the NoC from the host (MMIO writes and reads through a TLB
window) and `slv_*` counts requests a core sent to the PCIe tile (reads and writes of host memory). tt-metal's
fast dispatch moves host-to-device data as device reads of the host's command queue (`slv_rd`) and
device-to-host data as device writes into the completion queue (`slv_wr`); slow dispatch and tools such as
tt-umd move data with MMIO (`mst_*`). The counters wrap at 2^32 flits (256 GiB of DMA), every 5 s at 50 GB/s:
keep the interval well under 2 s (default 0.1 s).

**Firmware telemetry.** The table lives in ARC memory; the scratch registers `RESET_SCRATCH(13)` and `(12)`
(0x80030434, 0x80030430 on the ARC tile, NoC (8, 0)) give its address and the data array's. Each entry is
`tag | offset << 16`. Tag meanings and encodings follow `lib/tenstorrent/bh_arc/telemetry.h` of
tt-system-firmware 19.x; temperatures are signed 16.16 fixed point; power and current are integers.

**Ethernet.** `boot_results_t` at L1 0x7CC00 of each Ethernet tile (tt-metal
`tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h`): `eth_status_t` (port, training, speed, heartbeats),
SerDes and MAC/PCS training results, then `eth_live_status_t` at +0x200 with 64-bit MAC counters, and the local
and remote `chip_info_t` (board ID and tile of both ends). The base firmware copies the MAC counters there at
each link-status check: every **2.0 s** on ETH FW 1.13.1 (measured), stamped in `debug_buf_t` (0x7C400, word 466)
by a 50 MHz clock (10^8 per check; calibrated against the host to 50.03-50.06 MHz). tt-metal's fabric router runs
the same check from its context switch, so the snapshots continue while it owns the tile, about every 0.75 s, but
its path does not update that timestamp; those intervals are timed by the host (one poll interval of jitter). Byte counts are MAC frame bytes (Ethernet header and FCS
included, preamble and inter-frame gap not). Idle links are not silent: each sends about 300,000 frames/s of
64 B (19.2 MB/s, 0.04 % of 400 Gb/s) in each direction.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Validation

All on the tested server (four p150b, tt-kmd 2.12.1, firmware 19.17.1, tt-metal v0.81.1), 2026-11-08.

**Neutrality.** During a 10-minute idle record, the ARC monitor kept a descriptor open on all four cards
(sampling at 1 Hz). The cards stayed at 800 MHz and 17-19 W (ASIC), the same as with no process holding them.
The firmware's busy arbiter stayed at 800 MHz, and its kernel-throttler counters did not move.

**PCIe counters** (`validate_pcie_counters.py`, card `da`). Each transfer was a row-major BF16 tensor, and
every copy came back bit-exact.

| Path | Bytes moved | Counter family | Bytes counted / bytes moved |
| --- | --- | --- | --- |
| Fast dispatch, host to device (`ttnn.from_torch`) | 256 MiB × 3 | `slv_rd` (64 B per count) | 1.001 |
| Fast dispatch, device to host (`ttnn.to_torch`) | 256 MiB × 3 | `slv_wr` (64 B per count) | 1.002 |
| Slow dispatch, host to device | 64 MiB × 2 | `mst_wr` (32 B per count) | 1.0002 |
| Slow dispatch, device to host | 64 MiB × 2 | `mst_rd` (32 B per count) | 1.25 (plus 0.25 in `mst_wr`; not yet explained) |

At the driver's documented 32 bytes per count, the fast-dispatch rows read 0.50. So on Blackhole a DMA count is
a full 64-byte flit.

**Ethernet counters.** The cards form a 4-card ring with four 400G links per hop, and `--links` read the
ring's topology from the chip-info blocks.
* **Idle links.** Every UP link sends 599,850 ± 3 frames of exactly 64 B per 2-second snapshot (19.2 MB/s) in
  each direction.
* **Both ends agree.** During fabric traffic from `3f` to `da` (tt-metal `bench_unicast`, 64 MiB transfers,
  4 KiB pages), the sending end counted 222.19 GB transmitted and the receiving end 222.23 GB received. The
  difference, 0.02 %, is within the idle traffic of one snapshot. Resends, receive drops and bad-FCS frames were
  0. The traffic used one link (`link_idx` 0); the other three links of the hop stayed idle.
* **Frames.** The traffic averaged 1,369 B per frame: a 4 KiB page plus its fabric header goes out as about
  3 frames.
* **Wire traffic is not payload.** Each `bench_unicast` repetition put about 7.17 GB on the wire, whatever its
  payload: 0.13 GB (1 transfer pair) and 0.60 GB (9 transfers) both gave the same 7.17 GB.
  * Sustained at 9-10 GB/s for the whole repetition, this is far more than the payload, which the benchmark
    itself times at 10.75 GB/s.
  * Both MACs counted it, so the bytes really crossed the cable.
  * A 4 KiB transfer to L1 produced no such traffic.
  * The MAC counters therefore measure link occupancy. Attributing it to application payload needs the
    application's own byte count. Why the fabric moves this extra traffic is under investigation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

* Ethernet rates have the firmware's 2 s resolution; bursts shorter than that are averaged. The poll interval
  (`-w`) only sets how soon a new snapshot is seen.
* If a tile's owner stops running the link-status check (a runtime image that never context-switches), its
  snapshots freeze; the monitor then writes `stale` rows instead of rates.
* MAC bytes include the fabric's packet headers and Ethernet framing, so they exceed application payload; the
  idle frames (19.2 MB/s per link) are included too.
* Firmware power and current tags are whole watts and amperes; hwmon gives the same values.
* The two NoC-reading monitors hold `/dev/tenstorrent/N` open. That does not change the power state, but tools
  that wait until no process holds a card (an idle check, a reset with exclusive open) see the monitor as a
  holder. Start them after such a check, or stop them before a reset.
* Coordinates and layouts are for Blackhole with firmware 19.x and ETH FW 1.13; a different firmware can move
  them. The Ethernet monitor refuses to decode a tile whose chip-info block does not match the card.
* Tensix, DRAM-tile and L2CPU counters (e.g. per-channel DRAM bandwidth from the GDDR tiles' NIU counters) are
  deliberately not read: those tiles are clock-gated whenever no client requests power, and reading a gated tile
  loses the card.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tests

```bash
python3 -m unittest discover -s Tenstorrent/kmd2.12_Blackhole/tests -v
```

The tests build a synthetic sysfs tree and answer the NoC-read ioctl from a fake memory (the telemetry table,
`boot_results_t`), so they need neither cards nor tt-kmd.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
