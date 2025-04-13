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
    llm_serving_config=input_filename.split('.')[0].split('_')[2]
    input_sequence_length=llm_serving_config.split('-')[0]
    output_sequence_length=llm_serving_config.split('-')[1]
    concurrency=llm_serving_config.split('-')[2]

    #%% Create New DF
    filtered_df = pd.DataFrame()
    filtered_df['Timestamp'] = input_df['Timestamp']
    
    
    #%% Calculate GPU Average Clock from All XCDs
    gpu_0_clk = input_df.iloc[:, [8, 13, 18, 23, 28, 33, 38, 43]]
    gpu_1_clk = input_df.iloc[:, [103, 108, 113, 118, 123, 128, 133, 138]]
    gpu_2_clk = input_df.iloc[:, [198, 203, 208, 213, 218, 223, 228, 233]]
    gpu_3_clk = input_df.iloc[:, [293, 298, 303, 308, 313, 318, 323, 328]]
    gpu_4_clk = input_df.iloc[:, [388, 393, 398, 403, 408, 413, 418, 423]]
    gpu_5_clk = input_df.iloc[:, [483, 488, 493, 498, 503, 508, 513, 518]]
    gpu_6_clk = input_df.iloc[:, [578, 583, 588, 593, 598, 603, 608, 613]]
    gpu_7_clk = input_df.iloc[:, [673, 678, 683, 688, 693, 698, 703, 708]]
    
    
    filtered_df['gpu0_sclk_avg'] = gpu_0_clk.mean(axis=1)
    filtered_df['gpu0_sclk_max'] = gpu_0_clk.max(axis=1)
    filtered_df['gpu0_sclk_min'] = gpu_0_clk.min(axis=1)
    
    filtered_df['gpu1_sclk_avg'] = gpu_1_clk.mean(axis=1)
    filtered_df['gpu1_sclk_max'] = gpu_1_clk.max(axis=1)
    filtered_df['gpu1_sclk_min'] = gpu_1_clk.min(axis=1)
    
    filtered_df['gpu2_sclk_avg'] = gpu_2_clk.mean(axis=1)
    filtered_df['gpu2_sclk_max'] = gpu_2_clk.max(axis=1)
    filtered_df['gpu2_sclk_min'] = gpu_2_clk.min(axis=1)
    
    filtered_df['gpu3_sclk_avg'] = gpu_3_clk.mean(axis=1)
    filtered_df['gpu3_sclk_max'] = gpu_3_clk.max(axis=1)
    filtered_df['gpu3_sclk_min'] = gpu_3_clk.min(axis=1)
    
    filtered_df['gpu4_sclk_avg'] = gpu_4_clk.mean(axis=1)
    filtered_df['gpu4_sclk_max'] = gpu_4_clk.max(axis=1)
    filtered_df['gpu4_sclk_min'] = gpu_4_clk.min(axis=1)
    
    filtered_df['gpu5_sclk_avg'] = gpu_5_clk.mean(axis=1)
    filtered_df['gpu5_sclk_max'] = gpu_5_clk.max(axis=1)
    filtered_df['gpu5_sclk_min'] = gpu_5_clk.min(axis=1)
    
    filtered_df['gpu6_sclk_avg'] = gpu_6_clk.mean(axis=1)
    filtered_df['gpu6_sclk_max'] = gpu_6_clk.max(axis=1)
    filtered_df['gpu6_sclk_min'] = gpu_6_clk.min(axis=1)
    
    filtered_df['gpu7_sclk_avg'] = gpu_7_clk.mean(axis=1)
    filtered_df['gpu7_sclk_max'] = gpu_7_clk.max(axis=1)
    filtered_df['gpu7_sclk_min'] = gpu_7_clk.min(axis=1)
    
    #%% Calculate Min, Max, Avg Frequency across all GPUs.
    filtered_df['all_gpu_sclk_avg'] = filtered_df[['gpu0_sclk_avg', 'gpu1_sclk_avg', 'gpu2_sclk_avg', 'gpu3_sclk_avg', 'gpu4_sclk_avg', 'gpu5_sclk_avg', 'gpu6_sclk_avg', 'gpu7_sclk_avg']].mean(axis=1)
    filtered_df['all_gpu_sclk_min'] = filtered_df[['gpu0_sclk_min', 'gpu1_sclk_min', 'gpu2_sclk_min', 'gpu3_sclk_min', 'gpu4_sclk_min', 'gpu5_sclk_min', 'gpu6_sclk_min', 'gpu7_sclk_min']].min(axis=1)
    filtered_df['all_gpu_sclk_max'] = filtered_df[['gpu0_sclk_max', 'gpu1_sclk_max', 'gpu2_sclk_max', 'gpu3_sclk_max', 'gpu4_sclk_max', 'gpu5_sclk_max', 'gpu6_sclk_max', 'gpu7_sclk_max']].max(axis=1)
    
    #%% GPU Memory Clock
    filtered_df['gpu0_mclk']=input_df.iloc[:, 48]
    filtered_df['gpu1_mclk']=input_df.iloc[:, 143]
    filtered_df['gpu2_mclk']=input_df.iloc[:, 238]
    filtered_df['gpu3_mclk']=input_df.iloc[:, 333]
    filtered_df['gpu4_mclk']=input_df.iloc[:, 428]
    filtered_df['gpu5_mclk']=input_df.iloc[:, 523]
    filtered_df['gpu6_mclk']=input_df.iloc[:, 618]
    filtered_df['gpu7_mclk']=input_df.iloc[:, 713]
    
    
    #%% Calculate Min, Max, Avg memory clk across all GPUs.
    filtered_df['all_gpu_mclk_avg'] = filtered_df[['gpu0_mclk', 'gpu1_mclk', 'gpu2_mclk', 'gpu3_mclk', 'gpu4_mclk', 'gpu5_mclk', 'gpu6_mclk', 'gpu7_mclk']].mean(axis=1)
    filtered_df['all_gpu_mclk_min'] = filtered_df[['gpu0_mclk', 'gpu1_mclk', 'gpu2_mclk', 'gpu3_mclk', 'gpu4_mclk', 'gpu5_mclk', 'gpu6_mclk', 'gpu7_mclk']].min(axis=1)
    filtered_df['all_gpu_mclk_max'] = filtered_df[['gpu0_mclk', 'gpu1_mclk', 'gpu2_mclk', 'gpu3_mclk', 'gpu4_mclk', 'gpu5_mclk', 'gpu6_mclk', 'gpu7_mclk']].max(axis=1)
    
    #%% GPU Power Consumption
    filtered_df['gpu0_power']=input_df.iloc[:, 2]
    filtered_df['gpu1_power']=input_df.iloc[:, 97]
    filtered_df['gpu2_power']=input_df.iloc[:, 192]
    filtered_df['gpu3_power']=input_df.iloc[:, 287]
    filtered_df['gpu4_power']=input_df.iloc[:, 382]
    filtered_df['gpu5_power']=input_df.iloc[:, 477]
    filtered_df['gpu6_power']=input_df.iloc[:, 572]
    filtered_df['gpu7_power']=input_df.iloc[:, 667]
    
    #%% GPU Power Consumption
    filtered_df['all_gpu_power_avg'] = filtered_df[['gpu0_power', 'gpu1_power', 'gpu2_power', 'gpu3_power', 'gpu4_power', 'gpu5_power', 'gpu6_power', 'gpu7_power']].mean(axis=1)
    filtered_df['all_gpu_power_min'] = filtered_df[['gpu0_power', 'gpu1_power', 'gpu2_power', 'gpu3_power', 'gpu4_power', 'gpu5_power', 'gpu6_power', 'gpu7_power']].min(axis=1)
    filtered_df['all_gpu_power_max'] = filtered_df[['gpu0_power', 'gpu1_power', 'gpu2_power', 'gpu3_power', 'gpu4_power', 'gpu5_power', 'gpu6_power', 'gpu7_power']].max(axis=1)
    filtered_df['all_gpu_power_total'] = filtered_df[['gpu0_power', 'gpu1_power', 'gpu2_power', 'gpu3_power', 'gpu4_power', 'gpu5_power', 'gpu6_power', 'gpu7_power']].sum(axis=1)
    
    
    #%% GPU Temperature
    filtered_df['gpu0_temp']=input_df.iloc[:, 94]
    filtered_df['gpu1_temp']=input_df.iloc[:, 189]
    filtered_df['gpu2_temp']=input_df.iloc[:, 284]
    filtered_df['gpu3_temp']=input_df.iloc[:, 379]
    filtered_df['gpu4_temp']=input_df.iloc[:, 474]
    filtered_df['gpu5_temp']=input_df.iloc[:, 569]
    filtered_df['gpu6_temp']=input_df.iloc[:, 664]
    filtered_df['gpu7_temp']=input_df.iloc[:, 759]
    
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
    statistics_dict['random_input_len'] = input_sequence_length
    statistics_dict['random_output_len'] = output_sequence_length
    statistics_dict['max_concurrency'] = concurrency
    
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
    
    statistics_dict['gpu1_power_avg']  = no_idle_df['gpu1_power'].mean()
    statistics_dict['gpu1_power_max']  = no_idle_df['gpu1_power'].max()
    statistics_dict['gpu1_power_min']  = no_idle_df['gpu1_power'].min()
    statistics_dict['gpu1_power_90th'] = no_idle_df['gpu1_power'].quantile(0.9)
    statistics_dict['gpu1_power_99th'] = no_idle_df['gpu1_power'].quantile(0.99)
    
    statistics_dict['gpu2_power_avg']  = no_idle_df['gpu2_power'].mean()
    statistics_dict['gpu2_power_max']  = no_idle_df['gpu2_power'].max()
    statistics_dict['gpu2_power_min']  = no_idle_df['gpu2_power'].min()
    statistics_dict['gpu2_power_90th'] = no_idle_df['gpu2_power'].quantile(0.9)
    statistics_dict['gpu2_power_99th'] = no_idle_df['gpu2_power'].quantile(0.99)
    
    statistics_dict['gpu3_power_avg']  = no_idle_df['gpu3_power'].mean()
    statistics_dict['gpu3_power_max']  = no_idle_df['gpu3_power'].max()
    statistics_dict['gpu3_power_min']  = no_idle_df['gpu3_power'].min()
    statistics_dict['gpu3_power_90th'] = no_idle_df['gpu3_power'].quantile(0.9)
    statistics_dict['gpu3_power_99th'] = no_idle_df['gpu3_power'].quantile(0.99)
    
    statistics_dict['gpu4_power_avg']  = no_idle_df['gpu4_power'].mean()
    statistics_dict['gpu4_power_max']  = no_idle_df['gpu4_power'].max()
    statistics_dict['gpu4_power_min']  = no_idle_df['gpu4_power'].min()
    statistics_dict['gpu4_power_90th'] = no_idle_df['gpu4_power'].quantile(0.9)
    statistics_dict['gpu4_power_99th'] = no_idle_df['gpu4_power'].quantile(0.99)
    
    statistics_dict['gpu5_power_avg']  = no_idle_df['gpu5_power'].mean()
    statistics_dict['gpu5_power_max']  = no_idle_df['gpu5_power'].max()
    statistics_dict['gpu5_power_min']  = no_idle_df['gpu5_power'].min()
    statistics_dict['gpu5_power_90th'] = no_idle_df['gpu5_power'].quantile(0.9)
    statistics_dict['gpu5_power_99th'] = no_idle_df['gpu5_power'].quantile(0.99)
    
    statistics_dict['gpu6_power_avg']  = no_idle_df['gpu6_power'].mean()
    statistics_dict['gpu6_power_max']  = no_idle_df['gpu6_power'].max()
    statistics_dict['gpu6_power_min']  = no_idle_df['gpu6_power'].min()
    statistics_dict['gpu6_power_90th'] = no_idle_df['gpu6_power'].quantile(0.9)
    statistics_dict['gpu6_power_99th'] = no_idle_df['gpu6_power'].quantile(0.99)
    
    statistics_dict['gpu7_power_avg']  = no_idle_df['gpu7_power'].mean()
    statistics_dict['gpu7_power_max']  = no_idle_df['gpu7_power'].max()
    statistics_dict['gpu7_power_min']  = no_idle_df['gpu7_power'].min()
    statistics_dict['gpu7_power_90th'] = no_idle_df['gpu7_power'].quantile(0.9)
    statistics_dict['gpu7_power_99th'] = no_idle_df['gpu7_power'].quantile(0.99)
    
    # Individual GPU Temperature Statistics
    statistics_dict['gpu0_temp_avg']  = no_idle_df['gpu0_temp'].mean()
    statistics_dict['gpu0_temp_max']  = no_idle_df['gpu0_temp'].max()
    statistics_dict['gpu0_temp_min']  = no_idle_df['gpu0_temp'].min()
    statistics_dict['gpu0_temp_90th'] = no_idle_df['gpu0_temp'].quantile(0.9)
    statistics_dict['gpu0_temp_99th'] = no_idle_df['gpu0_temp'].quantile(0.99)
    
    statistics_dict['gpu1_temp_avg']  = no_idle_df['gpu1_temp'].mean()
    statistics_dict['gpu1_temp_max']  = no_idle_df['gpu1_temp'].max()
    statistics_dict['gpu1_temp_min']  = no_idle_df['gpu1_temp'].min()
    statistics_dict['gpu1_temp_90th'] = no_idle_df['gpu1_temp'].quantile(0.9)
    statistics_dict['gpu1_temp_99th'] = no_idle_df['gpu1_temp'].quantile(0.99)
    
    statistics_dict['gpu2_temp_avg']  = no_idle_df['gpu2_temp'].mean()
    statistics_dict['gpu2_temp_max']  = no_idle_df['gpu2_temp'].max()
    statistics_dict['gpu2_temp_min']  = no_idle_df['gpu2_temp'].min()
    statistics_dict['gpu2_temp_90th'] = no_idle_df['gpu2_temp'].quantile(0.9)
    statistics_dict['gpu2_temp_99th'] = no_idle_df['gpu2_temp'].quantile(0.99)
    
    statistics_dict['gpu3_temp_avg']  = no_idle_df['gpu3_temp'].mean()
    statistics_dict['gpu3_temp_max']  = no_idle_df['gpu3_temp'].max()
    statistics_dict['gpu3_temp_min']  = no_idle_df['gpu3_temp'].min()
    statistics_dict['gpu3_temp_90th'] = no_idle_df['gpu3_temp'].quantile(0.9)
    statistics_dict['gpu3_temp_99th'] = no_idle_df['gpu3_temp'].quantile(0.99)
    
    statistics_dict['gpu4_temp_avg']  = no_idle_df['gpu4_temp'].mean()
    statistics_dict['gpu4_temp_max']  = no_idle_df['gpu4_temp'].max()
    statistics_dict['gpu4_temp_min']  = no_idle_df['gpu4_temp'].min()
    statistics_dict['gpu4_temp_90th'] = no_idle_df['gpu4_temp'].quantile(0.9)
    statistics_dict['gpu4_temp_99th'] = no_idle_df['gpu4_temp'].quantile(0.99)
    
    statistics_dict['gpu5_temp_avg']  = no_idle_df['gpu5_temp'].mean()
    statistics_dict['gpu5_temp_max']  = no_idle_df['gpu5_temp'].max()
    statistics_dict['gpu5_temp_min']  = no_idle_df['gpu5_temp'].min()
    statistics_dict['gpu5_temp_90th'] = no_idle_df['gpu5_temp'].quantile(0.9)
    statistics_dict['gpu5_temp_99th'] = no_idle_df['gpu5_temp'].quantile(0.99)
    
    statistics_dict['gpu6_temp_avg']  = no_idle_df['gpu6_temp'].mean()
    statistics_dict['gpu6_temp_max']  = no_idle_df['gpu6_temp'].max()
    statistics_dict['gpu6_temp_min']  = no_idle_df['gpu6_temp'].min()
    statistics_dict['gpu6_temp_90th'] = no_idle_df['gpu6_temp'].quantile(0.9)
    statistics_dict['gpu6_temp_99th'] = no_idle_df['gpu6_temp'].quantile(0.99)
    
    statistics_dict['gpu7_temp_avg']  = no_idle_df['gpu7_temp'].mean()
    statistics_dict['gpu7_temp_max']  = no_idle_df['gpu7_temp'].max()
    statistics_dict['gpu7_temp_min']  = no_idle_df['gpu7_temp'].min()
    statistics_dict['gpu7_temp_90th'] = no_idle_df['gpu7_temp'].quantile(0.9)
    statistics_dict['gpu7_temp_99th'] = no_idle_df['gpu7_temp'].quantile(0.99)
    
    
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
    
    temp_concat = pd.concat([no_idle_df['gpu1_sclk_avg'], no_idle_df['gpu1_sclk_min'], no_idle_df['gpu1_sclk_max']])
    statistics_dict['gpu1_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu1_sclk_max']  = temp_concat.max()
    statistics_dict['gpu1_sclk_min']  = temp_concat.min()
    statistics_dict['gpu1_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu1_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu2_sclk_avg'], no_idle_df['gpu2_sclk_min'], no_idle_df['gpu2_sclk_max']])
    statistics_dict['gpu2_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu2_sclk_max']  = temp_concat.max()
    statistics_dict['gpu2_sclk_min']  = temp_concat.min()
    statistics_dict['gpu2_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu2_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu3_sclk_avg'], no_idle_df['gpu3_sclk_min'], no_idle_df['gpu3_sclk_max']])
    statistics_dict['gpu3_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu3_sclk_max']  = temp_concat.max()
    statistics_dict['gpu3_sclk_min']  = temp_concat.min()
    statistics_dict['gpu3_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu3_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu4_sclk_avg'], no_idle_df['gpu4_sclk_min'], no_idle_df['gpu4_sclk_max']])
    statistics_dict['gpu4_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu4_sclk_max']  = temp_concat.max()
    statistics_dict['gpu4_sclk_min']  = temp_concat.min()
    statistics_dict['gpu4_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu4_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu5_sclk_avg'], no_idle_df['gpu5_sclk_min'], no_idle_df['gpu5_sclk_max']])
    statistics_dict['gpu5_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu5_sclk_max']  = temp_concat.max()
    statistics_dict['gpu5_sclk_min']  = temp_concat.min()
    statistics_dict['gpu5_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu5_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu6_sclk_avg'], no_idle_df['gpu6_sclk_min'], no_idle_df['gpu6_sclk_max']])
    statistics_dict['gpu6_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu6_sclk_max']  = temp_concat.max()
    statistics_dict['gpu6_sclk_min']  = temp_concat.min()
    statistics_dict['gpu6_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu6_sclk_99th'] = temp_concat.quantile(0.99)
    
    temp_concat = pd.concat([no_idle_df['gpu7_sclk_avg'], no_idle_df['gpu7_sclk_min'], no_idle_df['gpu7_sclk_max']])
    statistics_dict['gpu7_sclk_avg']  = temp_concat.mean()
    statistics_dict['gpu7_sclk_max']  = temp_concat.max()
    statistics_dict['gpu7_sclk_min']  = temp_concat.min()
    statistics_dict['gpu7_sclk_90th'] = temp_concat.quantile(0.9)
    statistics_dict['gpu7_sclk_99th'] = temp_concat.quantile(0.99)
    
    
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
    
    statistics_dict['gpu1_mclk_avg']  = no_idle_df['gpu1_mclk'].mean()
    statistics_dict['gpu1_mclk_max']  = no_idle_df['gpu1_mclk'].max()
    statistics_dict['gpu1_mclk_min']  = no_idle_df['gpu1_mclk'].min()
    statistics_dict['gpu1_mclk_90th'] = no_idle_df['gpu1_mclk'].quantile(0.9)
    statistics_dict['gpu1_mclk_99th'] = no_idle_df['gpu1_mclk'].quantile(0.99)
    
    statistics_dict['gpu2_mclk_avg']  = no_idle_df['gpu2_mclk'].mean()
    statistics_dict['gpu2_mclk_max']  = no_idle_df['gpu2_mclk'].max()
    statistics_dict['gpu2_mclk_min']  = no_idle_df['gpu2_mclk'].min()
    statistics_dict['gpu2_mclk_90th'] = no_idle_df['gpu2_mclk'].quantile(0.9)
    statistics_dict['gpu2_mclk_99th'] = no_idle_df['gpu2_mclk'].quantile(0.99)
    
    statistics_dict['gpu3_mclk_avg']  = no_idle_df['gpu3_mclk'].mean()
    statistics_dict['gpu3_mclk_max']  = no_idle_df['gpu3_mclk'].max()
    statistics_dict['gpu3_mclk_min']  = no_idle_df['gpu3_mclk'].min()
    statistics_dict['gpu3_mclk_90th'] = no_idle_df['gpu3_mclk'].quantile(0.9)
    statistics_dict['gpu3_mclk_99th'] = no_idle_df['gpu3_mclk'].quantile(0.99)
    
    statistics_dict['gpu4_mclk_avg']  = no_idle_df['gpu4_mclk'].mean()
    statistics_dict['gpu4_mclk_max']  = no_idle_df['gpu4_mclk'].max()
    statistics_dict['gpu4_mclk_min']  = no_idle_df['gpu4_mclk'].min()
    statistics_dict['gpu4_mclk_90th'] = no_idle_df['gpu4_mclk'].quantile(0.9)
    statistics_dict['gpu4_mclk_99th'] = no_idle_df['gpu4_mclk'].quantile(0.99)
    
    statistics_dict['gpu5_mclk_avg']  = no_idle_df['gpu5_mclk'].mean()
    statistics_dict['gpu5_mclk_max']  = no_idle_df['gpu5_mclk'].max()
    statistics_dict['gpu5_mclk_min']  = no_idle_df['gpu5_mclk'].min()
    statistics_dict['gpu5_mclk_90th'] = no_idle_df['gpu5_mclk'].quantile(0.9)
    statistics_dict['gpu5_mclk_99th'] = no_idle_df['gpu5_mclk'].quantile(0.99)
    
    statistics_dict['gpu6_mclk_avg']  = no_idle_df['gpu6_mclk'].mean()
    statistics_dict['gpu6_mclk_max']  = no_idle_df['gpu6_mclk'].max()
    statistics_dict['gpu6_mclk_min']  = no_idle_df['gpu6_mclk'].min()
    statistics_dict['gpu6_mclk_90th'] = no_idle_df['gpu6_mclk'].quantile(0.9)
    statistics_dict['gpu6_mclk_99th'] = no_idle_df['gpu6_mclk'].quantile(0.99)
    
    statistics_dict['gpu7_mclk_avg']  = no_idle_df['gpu7_mclk'].mean()
    statistics_dict['gpu7_mclk_max']  = no_idle_df['gpu7_mclk'].max()
    statistics_dict['gpu7_mclk_min']  = no_idle_df['gpu7_mclk'].min()
    statistics_dict['gpu7_mclk_90th'] = no_idle_df['gpu7_mclk'].quantile(0.9)
    statistics_dict['gpu7_mclk_99th'] = no_idle_df['gpu7_mclk'].quantile(0.99)

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
    parser = argparse.ArgumentParser(description = 'ROCM-SMI CSV Parser for 8 GPUs LLM Inference Serving')
    parser.add_argument("-i", "--input_csv", help="Input CSV File from ROCM-SMI", type=str)
    args = parser.parse_args()
    main(args)
# %%""