"""Regression checks for geometry, arrival masking and vehicle kinematics."""
import numpy as np
from unified_sfm import Agent,Corridor,OpenArea,pair_collision,step,simulate
from evaluate import evaluate

def main():
    car=Agent(0,'car',np.zeros(2),np.zeros(2))
    ped=Agent(1,'ped',np.array([2.40,1.05]),np.zeros(2))
    assert pair_collision(ped,car)[0], 'Rectangle corner collision missed'
    ped.pos=np.array([2.6,1.25]);assert not pair_collision(ped,car)[0]
    other=Agent(2,'car',np.array([4.6,0.]),np.zeros(2))
    assert not pair_collision(car,other)[0]
    other.pos[0]=4.4;assert pair_collision(car,other)[0]
    # Exact rectangle-corner test must also work in vectorized evaluator.
    ped.pos=np.array([2.40,1.05])
    h=dict(t=np.array([0.,.01]),pos=np.array([[car.pos,ped.pos]]*2),vel=np.zeros((2,2,2)),heading=np.zeros((2,2)))
    rep=evaluate([car,ped],h,OpenArea());assert rep['collision_frames']==2
    ped.arrived=True;ped.t_arrive=0
    rep=evaluate([car,ped],h,OpenArea());assert rep['collision_frames']==0
    car=Agent(0,'car',np.array([20.,1.75]),np.array([-1.,0.]),heading=np.pi,sigma=-1,goal=np.array([-2.,1.75]))
    step([car],Corridor(60,3.5))
    assert abs(car.vel[1])<1e-10, 'Car slides laterally in corridor'
    assert np.linalg.norm(car.vel)<=1.02+1e-9, 'Acceleration limit failed'
    # Arrival and no overshoot artifacts outside the exit.
    a=Agent(1,'ped',np.array([59.9,1.]),np.array([1.3,0]),lat_pref=-.75,goal=np.array([62.,1.]))
    hist=simulate([a],Corridor(60,3.5),duration=2,sample_dt=.01)
    assert a.arrived and a.pos[0]>60.25
    assert np.isfinite(hist['pos']).all()
    print('PASS: circle/rectangle corners; rectangle SAT; arrival exclusion; no lateral car sliding; acceleration bound; exit arrival; finite trajectory')

if __name__=='__main__':main()
