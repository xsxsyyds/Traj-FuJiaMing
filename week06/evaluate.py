"""Vectorized, exact circle/rectangle geometry at every saved integration step."""
import numpy as np
from unified_sfm import PARAMS


def evaluate(agents, hist, road, label=''):
    t, pos, head = hist['t'], hist['pos'], hist['heading']
    active = np.array([t < a.t_arrive if a.arrived else np.ones(len(t), bool) for a in agents]).T
    hits = np.zeros(len(t), int)
    clearance = np.full(len(t), np.inf)
    by_pair = {}
    for i, a in enumerate(agents):
        for j in range(i+1,len(agents)):
            b = agents[j]
            valid = active[:,i] & active[:,j]
            delta = pos[:,j] - pos[:,i]
            if a.kind == b.kind == 'ped':
                gap = np.linalg.norm(delta,axis=1)-a.radius-b.radius
            elif a.kind != b.kind:
                ci, pi = (i,j) if a.kind == 'car' else (j,i)
                d = pos[:,pi]-pos[:,ci]
                c,s = np.cos(head[:,ci]),np.sin(head[:,ci])
                local = np.column_stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1]))
                q = np.abs(local)-np.array([PARAMS['car']['length']/2,PARAMS['car']['width']/2])
                gap = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)-PARAMS['ped']['radius']
            else:
                # SAT: positive gap is separation along an axis, negative = overlap.
                # Used for collision only, not claimed as Euclidean distance.
                ui = np.column_stack((np.cos(head[:,i]),np.sin(head[:,i])))
                uj = np.column_stack((np.cos(head[:,j]),np.sin(head[:,j])))
                vi = np.column_stack((-ui[:,1],ui[:,0]))
                vj = np.column_stack((-uj[:,1],uj[:,0]))
                gaps=[]
                for axis in (ui,vi,uj,vj):
                    ri=a.length/2*np.abs((ui*axis).sum(1))+a.width/2*np.abs((vi*axis).sum(1))
                    rj=b.length/2*np.abs((uj*axis).sum(1))+b.width/2*np.abs((vj*axis).sum(1))
                    gaps.append(np.abs((delta*axis).sum(1))-ri-rj)
                gap=np.max(gaps,axis=0)
            hit = valid & (gap < -1e-9)
            hits += hit
            key='-'.join(sorted((a.kind,b.kind)))
            by_pair[key]=by_pair.get(key,0)+int(hit.sum())
            clearance=np.minimum(clearance,np.where(valid,gap,np.inf))
    intrusion=np.zeros(len(t))
    if road.kind=='corridor':
        for i,a in enumerate(agents):
            extent = np.full(len(t),a.radius) if a.kind=='ped' else a.length/2*np.abs(np.sin(head[:,i]))+a.width/2*np.abs(np.cos(head[:,i]))
            d=np.maximum(extent-pos[:,i,1],pos[:,i,1]+extent-road.width)
            intrusion=np.maximum(intrusion,np.where(active[:,i],d,0))
    arrived=[a for a in agents if a.arrived]
    pedmask=np.array([a.kind=='ped' for a in agents])
    speeds=np.linalg.norm(hist['vel'],axis=2)
    valid=active[:,pedmask]
    finite=clearance[np.isfinite(clearance)]
    rep=dict(label=label,n_ped=int(pedmask.sum()),n_car=int((~pedmask).sum()),
        arrived_ped=sum(a.arrived and a.kind=='ped' for a in agents),
        arrived_car=sum(a.arrived and a.kind=='car' for a in agents),
        t_last_arrival=max((float(a.t_arrive) for a in arrived),default=None),
        t_all_arrived=max((float(a.t_arrive) for a in arrived),default=None) if len(arrived)==len(agents) else None,
        collision_frames=int((hits>0).sum()),collision_pairs=int(hits.sum()),collision_by_pair=by_pair,
        collision_frame_ratio=float((hits>0).mean()),
        min_clearance=float(finite.min()) if len(finite) else None,
        boundary_max_intrusion=float(intrusion.max()),boundary_bad_frames=int((intrusion>1e-9).sum()),
        mean_speed_ped=float(speeds[:,pedmask][valid].mean()) if valid.any() else None,
        time_horizon=float(t[-1]),check_dt=float(t[1]-t[0]),
        arrival_times=[float(a.t_arrive) if a.arrived else None for a in agents])
    hist['collision_count']=hits
    hist['clearance']=clearance
    hist['boundary_intrusion']=intrusion
    hist['active']=active
    return rep
