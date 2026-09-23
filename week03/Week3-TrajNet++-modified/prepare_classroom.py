"""Convert course TXT into TrajNet++ NDJSON; stdlib only, no interpolation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=float, default=0.01)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--stride', type=int, default=8)
    a = p.parse_args()
    if min(a.scale, a.fps, a.step, a.stride) <= 0:
        raise ValueError('Parameters must be positive')
    src = ROOT / 'classroom/raw/circle-10m-64-1.txt'
    rows = {}
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        v = list(map(float, line.split()))
        if len(v) < 4 or not all(math.isfinite(x) for x in v[:4]):
            raise ValueError('Invalid row')
        ped, frame = int(v[0]), int(v[1])
        if ped != v[0] or frame != v[1] or (frame, ped) in rows:
            raise ValueError('Invalid ID/frame or duplicate')
        rows[frame, ped] = [v[2]*a.scale, v[3]*a.scale]
    frames = sorted({k[0] for k in rows})
    ids = sorted({k[1] for k in rows})
    if frames != list(range(frames[0], frames[-1]+1)):
        raise ValueError('Frames are not consecutive')
    if len(rows) != len(frames)*len(ids):
        raise ValueError('Incomplete synchronized tracks')
    sampled = frames[::a.step]
    n = len(sampled)
    # Same four primary pedestrians across all variants; all 64 remain neighbours.
    primary = [ids[i*len(ids)//4] for i in range(4)]
    bounds = {'train': (0, int(n*.6)), 'val': (int(n*.6), int(n*.8)), 'test': (int(n*.8), n)}
    audit = {'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
             'raw_columns': ['pedestrian_id','frame','x','y','extra_unused'],
             'coordinate_scale': a.scale, 'unit_status': 'cm-to-m is an unverified assumption',
             'raw_fps_assumed': a.fps, 'fps_status': 'unverified',
             'frame_step': a.step, 'obs_length': 8, 'pred_length': 12,
             'primary_ids': primary, 'neighbour_ids': ids,
             'sampled_to_original_frame': sampled, 'splits': {}}
    scene_id = 0
    for split, (lo, hi) in bounds.items():
        starts = list(range(lo, hi-20+1, a.stride))
        if not starts:
            raise ValueError('Split too short for 8+12 points')
        dest = ROOT / 'DATA_BLOCK/circle_classroom' / split / 'circle.ndjson'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', encoding='utf-8', newline='\n') as f:
            for start in starts:
                for ped in primary:
                    f.write(json.dumps({'scene': {'id':scene_id,'p':ped,'s':start,'e':start+19,
                                                  'fps':a.fps/a.step,'tag':0}})+'\n')
                    scene_id += 1
            for j in range(lo, hi):
                for ped in ids:
                    x, y = rows[sampled[j], ped]
                    f.write(json.dumps({'track':{'f':j,'p':ped,'x':x,'y':y}})+'\n')
        audit['splits'][split] = {'sampled_bounds_exclusive':[lo,hi],
                                 'original_frame_bounds':[sampled[lo],sampled[hi-1]],
                                 'windows':len(starts),'scenes':len(starts)*len(primary)}
    (ROOT/'classroom/data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit['splits'],indent=2))

if __name__ == '__main__':
    main()
