"""Run the fixed-parameter experiments; no parameter search or calibration."""
import json
from pathlib import Path
import numpy as np
from unified_sfm import Agent, OpenArea, PARAMS, DT, simulate
from run_corridor import build
from evaluate import evaluate

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)

def build_circle():
    agents=[]
    for i in range(20):
        angle=2*np.pi*i/20
        pos=50*np.array([np.cos(angle),np.sin(angle)])
        kind='ped' if i%2==0 else 'car'
        heading=angle+np.pi
        agents.append(Agent(i,kind,pos,np.zeros(2),heading=heading,goal=-pos.copy()))
    return agents,OpenArea()

def main():
    reports=[]
    cases=[('corridor_wide',5.0,True,'宽长廊'),('corridor_narrow',3.5,True,'窄长廊'),
           ('corridor_blocked',2.2,True,'空间不足'),('corridor_no_rule',3.0,False,'关闭侧移规则'),
           ('corridor_rule',3.0,True,'启用侧移规则'),('circle',None,True,'人车混合圆环对趾')]
    for name,width,rule,label in cases:
        agents,road=build_circle() if width is None else build(width,rule)
        print('RUN',name,flush=True)
        hist=simulate(agents,road,duration=150,dt=DT,sample_dt=DT,keep_right=rule)
        rep=evaluate(agents,hist,road,label)
        rep.update(name=name,width=width,keep_right=rule)
        reports.append(rep)
        np.savez_compressed(OUT/(name+'.npz'),**hist,kinds=np.array([a.kind for a in agents]),
            goals=np.array([a.goal for a in agents]),t_arrive=np.array([a.t_arrive for a in agents]))
        (OUT/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(name,'arrivals',rep['arrived_ped'],rep['arrived_car'],'collision frames',rep['collision_frames'],
              'boundary',rep['boundary_max_intrusion'],'all arrived',rep['t_all_arrived'],flush=True)
    (OUT/'parameters.json').write_text(json.dumps(dict(parameters=PARAMS,dt=DT,duration=150,seed=0),indent=2),encoding='utf-8')

if __name__=='__main__': main()
