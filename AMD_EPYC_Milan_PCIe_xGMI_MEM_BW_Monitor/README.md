This is a config file to use with AMDuProfPCM to collect bus throughput on AMD EPYC Milan CPUs.
AMD EPYC Milan CPUID is family 0x19, model 0x01. 
The data collected are:
* PCIe bus throughput (utilization): there are 128 lanes of PCIe in each AMD EPYC Milan CPUs, 64 of them are used for intersocket communication.
* Memory channel throughput (utilization): there are 8 memory channels (A to H) in each AMD EPYC Milan CPUs.
* xGMI bus throughput (utilization): Global Memory Interconnect used for intersocket communication (taken from 64 lanes of PCIe --> 4 xGMI links)