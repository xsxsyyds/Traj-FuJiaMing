"""Small CPU preset calling the upstream trainer without changing model code."""
import os
from pathlib import Path
import sys
import time
import json
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
torch.set_num_threads(2)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
defaults = ['--path','circle_classroom','--type','social','--epochs','10',
            '--obs_length','8','--pred_length','12','--batch_size','2',
            '--hidden-dim','32','--coordinate-embedding-dim','16',
            '--pool_dim','32','--latent_dim','8','--n','4','--cell_side','1',
            '--save_every','1','--disable-cuda','--seed','42','--output','baseline']
extra = sys.argv[1:]
if any(x in extra for x in ['--path','--obs_length','--pred_length']):
    raise SystemExit('Use the fixed classroom data and 8+12 windows; edit the preset for a different task.')
def last_value(flags, default):
    value=default
    for i, token in enumerate(extra[:-1]):
        if token in flags:
            value=extra[i+1]
    return value
kind=last_value(['--type'],'social')
name=last_value(['--output','-o'],'baseline')
if '/' in name or '\\' in name or name in ['.','..']:
    raise SystemExit('Output name must be a simple experiment name')
output=ROOT/f'OUTPUT_BLOCK/circle_classroom/lstm_{kind}_{name}.pkl'
if output.exists() or Path(str(output)+'.log').exists():
    raise SystemExit('Output already exists; choose a new --output name.')
from trajnetbaselines.lstm.trainer import main
sys.argv=[sys.argv[0]]+defaults+extra
start=time.perf_counter()
main()
duration=time.perf_counter()-start
Path(str(output)+'.classroom.json').write_text(json.dumps({
    'elapsed_seconds':duration,'arguments':sys.argv[1:],
    'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','threads':2
},indent=2),encoding='utf-8')
print(f'Training + upstream validation elapsed: {duration:.2f} seconds')
print('Checkpoint:', output)
