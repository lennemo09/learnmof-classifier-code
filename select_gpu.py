import subprocess
import numpy as np
import sys
import os
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode('utf-8')
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print(gpu_df)
    print('GPU usage:')
    free_mem = gpu_df['memory.free'].map(lambda x: x.rstrip(' MiB')).astype(int)
    print(free_mem)
    idx = free_mem.idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx

if __name__ == '__main__':
    free_gpu_id = get_free_gpu()
    # torch.cuda.set_device(free_gpu_id)