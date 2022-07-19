#!/bin/bash
# Default Value
NUM_GPUS=${1:-1}
NVLINK_LANE=${2:-12}
#SECOND_DELAY=${3:-1}

BYTEHISTORY=()

# GPU_0 TX | GPU_0 RX | GPU_1 TX | GPU 1 RX | ...
# Read NVIDIA-SMI Output
readarray -t OUTPUT < <(nvidia-smi nvlink -gt d)

# Initialize BYTEHISTORY
for (( GPU = 0; GPU < $NUM_GPUS; GPU=GPU+1)) 
do
  BYTEHISTORY+=(0)
  BYTEHISTORY+=(0)
  for (( LANE = 0; LANE < $NVLINK_LANE; LANE=LANE+1))
  do
    INDEXTX=$(($GPU*($NVLINK_LANE*2)+$LANE*2+$GPU+1))
    INDEXRX=$(($GPU*($NVLINK_LANE*2)+$LANE*2+$GPU+2))
    TEMPTX=($(echo ${OUTPUT[$INDEXTX]}))
    TEMPRX=($(echo ${OUTPUT[$INDEXRX]}))

    BYTEHISTORY[$GPU*2+0]=$((${BYTEHISTORY[$GPU*2+0]}+${TEMPTX[4]}))
    BYTEHISTORY[$GPU*2+1]=$((${BYTEHISTORY[$GPU*2+1]}+${TEMPRX[4]}))
  done 
done
echo ${BYTEHISTORY[@]}
#OUTPUT=($(nvidia-smi nvlink -gt d))

echo "GPU,TX (MiB/s), RX (MiB/s)"
#Loop Until Terminated
for (( ; ; ))
do
  AGGRTX=0
  AGGRRX=0 
  # Read NVIDIA-SMI Output
  readarray -t OUTPUT < <(nvidia-smi nvlink -gt d)
  for (( GPU = 0; GPU < $NUM_GPUS; GPU=GPU+1))
  do
    SUMTX=0
    SUMRX=0
    for (( LANE = 0; LANE < $NVLINK_LANE; LANE=LANE+1))
    do
      INDEXTX=$(($GPU*($NVLINK_LANE*2)+$LANE*2+$GPU+1))
      INDEXRX=$(($GPU*($NVLINK_LANE*2)+$LANE*2+$GPU+2))
      TEMPTX=($(echo ${OUTPUT[$INDEXTX]}))
      TEMPRX=($(echo ${OUTPUT[$INDEXRX]}))
  
      SUMTX=$(($SUMTX+${TEMPTX[4]}))
      SUMRX=$(($SUMRX+${TEMPRX[4]}))
    done 
    # Calculate Throughput
    THROUGHPUTTX=$((($SUMTX-${BYTEHISTORY[$GPU*2+0]})/1024))
    THROUGHPUTRX=$((($SUMRX-${BYTEHISTORY[$GPU*2+1]})/1024))
    AGGRTX=$(($AGGRTX+$THROUGHPUTTX))
    AGGRRX=$(($AGGRRX+$THROUGHPUTRX))
    echo "${GPU},${THROUGHPUTTX},${THROUGHPUTRX}"


    # Update Bytehistory
    BYTEHISTORY[$GPU*2+0]=$SUMTX
    BYTEHISTORY[$GPU*2+1]=$SUMRX
  done
  echo "S,${AGGRTX},${AGGRRX}"
  sleep 1
done

