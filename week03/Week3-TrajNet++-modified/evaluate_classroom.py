"""Evaluate primary pedestrians with history only; same scenes for model and CV."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trajnetplusplustools

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint', help='Only load a checkpoint you generated/trust')
    p.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args()
    torch.set_num_threads(2)
    ckpt=Path(args.checkpoint).resolve()
    # Upstream uses pickled predictor objects. Only load a locally generated model.
    predictor=torch.load(ckpt,map_location='cpu')
    model=predictor.model.eval()
    reader=trajnetplusplustools.Reader(str(ROOT/f'DATA_BLOCK/circle_classroom/{args.split}/circle.ndjson'),scene_type='paths')
    errors=[];cv_errors=[];example=None
    start=time.perf_counter()
    for scene_id,paths in reader.scenes():
        xy=torch.tensor(trajnetplusplustools.Reader.paths_to_xy(paths),dtype=torch.float32)
        assert xy.shape[0]==20 and torch.isfinite(xy).all()
        history=xy[:8].clone()
        with torch.no_grad():
            _,positions=model(history,torch.zeros(xy.shape[1],2),torch.tensor([0,xy.shape[1]]),n_predict=12)
        pred=positions[-12:,0]
        truth=xy[8:,0]
        cv=history[-1,0]+torch.arange(1,13)[:,None]*(history[-1,0]-history[-2,0])
        errors.append(torch.linalg.vector_norm(pred-truth,dim=-1).numpy())
        cv_errors.append(torch.linalg.vector_norm(cv-truth,dim=-1).numpy())
        if example is None:
            example=(scene_id,history[:,0].numpy(),truth.numpy(),pred.numpy(),cv.numpy())
    e=np.stack(errors);c=np.stack(cv_errors)
    if not np.isfinite(e).all():
        raise ValueError('Non-finite model predictions')
    metrics={'split':args.split,'scenes':len(e),'primary_only':True,
             'model':{'ADE':float(e.mean()),'FDE':float(e[:,-1].mean())},
             'constant_velocity':{'ADE':float(c.mean()),'FDE':float(c[:,-1].mean())},
             'elapsed_seconds':time.perf_counter()-start,
             'units':'scaled coordinate units; metres only under unverified cm assumption',
             'protocol':'all 64 neighbours, no ground-truth future passed to model; one mean rollout'}
    base=Path(str(ckpt)+'.'+args.split)
    Path(str(base)+'.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    sid,h,t,pred,cv=example
    fig,ax=plt.subplots(figsize=(6,6))
    for label,v in [('History',h),('Truth',np.vstack([h[-1],t])),('Model',np.vstack([h[-1],pred])),('CV',np.vstack([h[-1],cv]))]:
        ax.plot(v[:,0],v[:,1],'.-',label=label)
    ax.set(xlabel='x (scaled units)',ylabel='y (scaled units)',title=f'{args.split}, first scene {sid}')
    ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(str(base)+'.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
