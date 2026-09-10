#!/bin/bash
# Default Value
NUM_NIC=${1:-1}
NUM_PORT=${2:-1}
#SECOND_DELAY=${3:-1}

BYTEHISTORY=()

# MLX5_0 TX | MLX5_0 RX | MLX5_1 TX | MLX5_1 RX | ...
# Read NVIDIA-SMI Output
readarray -t OUTPUT < <(nvidia-smi nvlink -gt d)

# Initialize BYTEHISTORY
for (( NIC = 0; NIC < $NUM_NIC; NIC=NIC+1)) 
do
  BYTEHISTORY+=(0)
  BYTEHISTORY+=(0)
  for (( PORT = 0; PORT < $NUM_PORT; PORT=PORT+1))
  do
    #Total number of data octets, divided by 4 (lanes), transmitted on all VLs. This is 64 bit counter.
    TX=($(cat /sys/class/infiniband/mlx5_${NIC}/ports/$(($PORT+1))/counters/port_xmit_data)*4)
    #Total number of data octets, divided by 4 (lanes), received on all VLs. This is 64 bit counter.
    RX=($(cat /sys/class/infiniband/mlx5_${NIC}/ports/$(($PORT+1))/counters/port_rcv_data)*4)

    INDEXTX=$(($NIC*($NUM_PORT*2)+$PORT*2+0))
    INDEXRX=$(($NIC*($NUM_PORT*2)+$PORT*2+1))

    BYTEHISTORY[$INDEXTX]=$TX 
    BYTEHISTORY[$INDEXRX]=$RX 
  done 
done
echo ${BYTEHISTORY[@]}
#OUTPUT=($(nvidia-smi nvlink -gt d))

echo "NIC,Port,TX (MiB/s),RX (MiB/s),Total (MiB/s)"
#Loop Until Terminated
for (( ; ; ))
do
  AGGRTX=0
  AGGRRX=0 
  for (( NIC = 0; NIC < $NUM_NIC; NIC=NIC+1)) 
  do
    for (( PORT = 0; PORT < $NUM_PORT; PORT=PORT+1))
    do
      TX=($(cat /sys/class/infiniband/mlx5_${NIC}/ports/$(($PORT+1))/counters/port_xmit_data)*4)
      RX=($(cat /sys/class/infiniband/mlx5_${NIC}/ports/$(($PORT+1))/counters/port_rcv_data)*4)
  
      INDEXTX=$(($NIC*($NUM_PORT*2)+$PORT*2+0))
      INDEXRX=$(($NIC*($NUM_PORT*2)+$PORT*2+1))

      THROUGHPUTTX=$((($TX-${BYTEHISTORY[$INDEXTX]})/1048576))
      THROUGHPUTRX=$((($RX-${BYTEHISTORY[$INDEXRX]})/1048576))
      THROUGHPUTTOT=$(($THROUGHPUTTX+$THROUGHPUTRX))
      
      echo "${NIC},${PORT},${THROUGHPUTTX},${THROUGHPUTRX},${THROUGHPUTTOT}"

      BYTEHISTORY[$INDEXTX]=$TX
      BYTEHISTORY[$INDEXRX]=$RX
      AGGRTX=$(($AGGRTX+$THROUGHPUTTX))
      AGGRRX=$(($AGGRRX+$THROUGHPUTRX))
    done 
  done
  AGGRTOT=$(($AGGRTX+$AGGRRX))
  echo "S,S,${AGGRTX},${AGGRRX},${AGGRTOT}"
  sleep 1
done