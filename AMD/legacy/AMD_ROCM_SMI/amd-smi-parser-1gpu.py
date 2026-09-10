# Parser for AMD ROCM-SMI CSV to more flatten CSV.
# (C) 2025 Bagus Hanindhito

#%% Load Library
import numpy as np
import pandas as pd
import argparse
import json
import os
from numpyencoder import NumpyEncoder

#%% Argument For Development Only
#args={
#    "input_csv"  : "/home/bagus/L11/baylor_telemetry/waco1/4FBRZ44_coralgemm_30min_baylor_waco1_telemetry_01232025_amd_rocm_smi_monitor.csv",
#    "output_csv" : "/home/bagus/L11/preprocessed/4FBRZ44_coralgemm_30min_baylor_waco1_telemetry_01232025_amd_rocm_smi_monitor_out.csv"
#}

#input_csv="/home/bagus/SC2025/DeepSeek/MI300X/smi/20250308_043525_3200-1600-256.csv.csv"


#%% Main Body
def main(args):
    #%% Read input CSV file to Panda DF
    input_df = pd.read_csv(args.input_csv)
    input_filename=os.path.basename(args.input_csv)
    output_path=os.path.dirname(args.input_csv)
    blas_config=input_filename.split('.')[0].split('_')
    mul=blas_config[5]
    acc=blas_config[6]
    scale=blas_config[7]
    ops=blas_config[4]

    if(ops=="gemm"):
        if   (blas_config[8]=="96K"): M=96804; N=96804; K=96804
        elif (blas_config[8]=="64K"): M=65536; N=65536; K=65536
        elif (blas_config[8]=="32K"): M=32768; N=32768; K=32768
        elif (blas_config[8]=="16K"): M=16384; N=16384; K=16384
        elif (blas_config[8]=="8K"):  M=8192;  N=8192;  K=8192
        elif (blas_config[8]=="4K"):  M=4096;  N=4096;  K=4096
    elif(ops=="gemv"):
        if   (blas_config[8]=="96K"): M=96804; N=1; K=96804
        elif (blas_config[8]=="64K"): M=65536; N=1; K=65536
        elif (blas_config[8]=="32K"): M=32768; N=1; K=32768
        elif (blas_config[8]=="16K"): M=16384; N=1; K=16384
        elif (blas_config[8]=="8K"):  M=8192;  N=1; K=8192
        elif (blas_config[8]=="4K"):  M=4096;  N=1; K=4096

    #%% Create New DF
    filtered_df = pd.DataFrame()
    filtered_df['Timestamp'] = input_df['Timestamp']
    
    
    #%% Calculate GPU Average Clock from All XCDs
    gpu_0_clk = input_df.iloc[:, [8, 13, 18, 23, 28, 33, 38, 43]]
    
    
    filtered_df['gpu0_sclk_avg'] = gpu_0_clk.mean(axis=1)
    filtered_df['gpu0_sclk_max'] = gpu_0_clk.max(axis=1)
    filtered_df['gpu0_sclk_min'] = gpu_0_clk.min(axis=1)
    
    
    #%% Calculate Min, Max, Avg Frequency across all GPUs.
    filtered_df['all_gpu_sclk_avg'] = filtered_df[['gpu0_sclk_avg']].mean(axis=1)
    filtered_df['all_gpu_sclk_min'] = filtered_df[['gpu0_sclk_min']].min(axis=1)
    filtered_df['all_gpu_sclk_max'] = filtered_df[['gpu0_sclk_max']].max(axis=1)
    
    #%% GPU Memory Clock
    filtered_df['gpu0_mclk']=input_df.iloc[:, 48]
    
    
    #%% Calculate Min, Max, Avg memory clk across all GPUs.
    filtered_df['all_gpu_mclk_avg'] = filtered_df[['gpu0_mclk']].mean(axis=1)
    filtered_df['all_gpu_mclk_min'] = filtered_df[['gpu0_mclk']].min(axis=1)
    filtered_df['all_gpu_mclk_max'] = filtered_df[['gpu0_mclk']].max(axis=1)
    
    #%% GPU Power Consumption
    filtered_df['gpu0_power']=input_df.iloc[:, 2]
    
    #%% GPU Power Consumption
    filtered_df['all_gpu_power_avg'] = filtered_df[['gpu0_power']].mean(axis=1)
    filtered_df['all_gpu_power_min'] = filtered_df[['gpu0_power']].min(axis=1)
    filtered_df['all_gpu_power_max'] = filtered_df[['gpu0_power']].max(axis=1)
    filtered_df['all_gpu_power_total'] = filtered_df[['gpu0_power']].sum(axis=1)
    
    
    #%% GPU Temperature
    filtered_df['gpu0_temp']=input_df.iloc[:, 94]
    
    #%% Remove all idle data (Memory <1000 MHz)
    no_idle_df =  filtered_df[filtered_df['all_gpu_mclk_avg']>1000]

    #no_idle_df =  (filtered_df.iloc[5:]).iloc[:-5]
    
    
    
    #%% Generate Statistics
    # total power (min, max, avg, 90-tile)
    # sclock frequency (min, max, avg, 90-tile)
    # mclock frequency (min, max, avg, 90-tile)
    # temperature (min, max, avg, 90-tile)
    statistics_dict = {}
    
    
    # This should be taken from filename
    statistics_dict['M'] = M
    statistics_dict['N'] = N
    statistics_dict['K'] = K
    statistics_dict['MUL'] = mul
    statistics_dict['ACC'] = acc
    statistics_dict['SCALE'] = scale


    
    # Aggregate GPU Power Statistics
    statistics_dict['all_gpu_power_avg']  = no_idle_df['all_gpu_power_total'].mean()
    statistics_dict['all_gpu_power_max']  = no_idle_df['all_gpu_power_total'].max()
    statistics_dict['all_gpu_power_min']  = no_idle_df['all_gpu_power_total'].min()
    statistics_dict['all_gpu_power_90th'] = no_idle_df['all_gpu_power_total'].quantile(0.9)
    statistics_dict['all_gpu_power_99th'] = no_idle_df['all_gpu_power_total'].quantile(0.99)
    
    # Individual GPU Power Statistics
    statistics_dict['gpu0_power_avg']  = no_idle_df['gpu0_power'].mean()
    statistics_dict['gpu0_power_max']  = no_idle_df['gpu0_power'].max()
    statistics_dict['gpu0_power_min']  = no_idle_df['gpu0_power'].min()
    statistics_dict['gpu0_power_90th'] = no_idle_df['gpu0_power'].quantile(0.9)
    statistics_dict['gpu0_power_99th'] = no_idle_df['gpu0_power'].quantile(0.99)
    
    # Individual GPU Temperature Statistics
    statistics_dict['gpu0_temp_avg']  = no_idle_df['gpu0_temp'].mean()
    statistics_dict['gpu0_temp_max']  = no_idle_df['gpu0_temp'].max()
    statistics_dict['gpu0_temp_min']  = no_idle_df['gpu0_temp'].min()
    statistics_dict['gpu0_temp_90th'] = no_idle_df['gpu0_temp'].quantile(0.9)
    statistics_dict['gpu0_temp_99th'] = no_idle_df['gpu0_temp'].quantile(0.99)
    
    # Aggregate GPU SoC Frequency Statistics
    
    temp_concat = pd.concat([no_idle_df['all_gpu_sclk_avg'], no_idle_df['all_gpu_sclk_min'], no_idle_df['all_gpu_sclk_max']])
    statistics_dict['all_gpu_sclk_avg']  = temp_concat.mean()
    statistics_dict['all_gpu_sclk_max']  = temp_concat.max()
    statistics_dict['all_gpu_sclk_min']  = temp_concat.min()
    statistics_dict['all_gpu_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['all_gpu_sclk_99th'] = temp_concat.quantile(0.99)
    
    # Individual GPU SoC Frequency Statistics
    temp_concat = pd.concat([no_idle_df['gpu0_sclk_avg'], no_idle_df['gpu0_sclk_min'], no_idle_df['gpu0_sclk_max']])
    statistics_dict['gpu0_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu0_sclk_max']  = temp_concat.max()
    statistics_dict['gpu0_sclk_min']  = temp_concat.min()
    statistics_dict['gpu0_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu0_sclk_99th'] = temp_concat.quantile(0.99)
       
    # Aggregate GPU Mem Frequency Statistics
    temp_concat = pd.concat([no_idle_df['all_gpu_mclk_avg'], no_idle_df['all_gpu_mclk_min'], no_idle_df['all_gpu_mclk_max']])
    statistics_dict['all_gpu_mclk_avg']  = temp_concat.mean()
    statistics_dict['all_gpu_mclk_max']  = temp_concat.max()
    statistics_dict['all_gpu_mclk_min']  = temp_concat.min()
    statistics_dict['all_gpu_mclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['all_gpu_mclk_99th'] = temp_concat.quantile(0.99)
    
    # Individual GPU Mem Frequency Statistics
    statistics_dict['gpu0_mclk_avg']  = no_idle_df['gpu0_mclk'].mean()
    statistics_dict['gpu0_mclk_max']  = no_idle_df['gpu0_mclk'].max()
    statistics_dict['gpu0_mclk_min']  = no_idle_df['gpu0_mclk'].min()
    statistics_dict['gpu0_mclk_90th'] = no_idle_df['gpu0_mclk'].quantile(0.9)
    statistics_dict['gpu0_mclk_99th'] = no_idle_df['gpu0_mclk'].quantile(0.99)
    
    #%% Dump statistics as JSONL
    jsonlname=os.path.join(output_path,input_filename.split('.')[0]+'-stat.jsonl')
    with open(jsonlname, 'w') as jsonlfile:
        json.dump(statistics_dict, jsonlfile, cls=NumpyEncoder)
        jsonlfile.write('\n')

    #%% Dump filtered dataframe as CSV
    csvname=os.path.join(output_path,input_filename.split('.')[0]+'-filtered.csv')
    filtered_df.to_csv(csvname, index=False)

#%% Startup and Argument Parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'ROCM-SMI CSV Parser for 1 GPU ROCBLAS/HIPBLASLT')
    parser.add_argument("-i", "--input_csv", help="Input CSV File from ROCM-SMI", type=str)
    args = parser.parse_args()
    main(args)
# %%""