# Parser for AMD ROCM-SMI CSV to more flatten CSV.
# (C) 2025 Bagus Hanindhito

#%% Load Library
import numpy as np
import pandas as pd
import argparse

#%% Argument For Development Only
#args={
#    "input_csv"  : "/home/bagus/L11/baylor_telemetry/waco1/4FBRZ44_coralgemm_30min_baylor_waco1_telemetry_01232025_amd_rocm_smi_monitor.csv",
#    "output_csv" : "/home/bagus/L11/preprocessed/4FBRZ44_coralgemm_30min_baylor_waco1_telemetry_01232025_amd_rocm_smi_monitor_out.csv"
#}

#%% Main Body
def main(args):
    #%% Read input CSV file to Panda DF
    input_df = pd.read_csv(args.input_csv)
    
    #%% Groupby
    #input_df['group_num'] = input_df.groupby('Timestamp')['device'].transform(lambda x: range(1, len(x)+1))
    #output_df = input_df.pivot(index='Timestamp', columns='group_num')
    #output_df.columns = [''.join([lvl1, str(lvl2)]) for lvl1, lvl2 in output_df.columns]
    
    cc = input_df.groupby(['Timestamp']).cumcount() + 1
    output_df = input_df.set_index(['Timestamp', cc]).unstack().sort_index(level=1)
    #output_df.columns = ['_'.join(map(str,i)) for i in input_df.columns]
    #output_df.reset_index()
    
    #%% Output CSV
    output_df.to_csv(args.output_csv)


#%% Startup and Argument Parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'ROCM-SMI CSV Parser')
    parser.add_argument("-i", "--input_csv", help="Input CSV File from ROCM-SMI", type=str)
    parser.add_argument("-o","--output_csv", help="Output CSV File from ROCM-SMI", type=str)
    args = parser.parse_args()
    main(args)
# %%""