"""Create 300 dpi PNGs, compact GIFs and a measured-results table."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from PIL import Image, ImageDraw, ImageFont
from unified_sfm import rect_corners, PARAMS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
COLORS={'ped':'#2077b4','car':'#df6b25'}

def load(name):
    with np.load(OUT/(name+'.npz')) as archive:
        return {key:archive[key] for key in archive.files}

def paths(ax,z,width=None):
    for i,kind in enumerate(z['kinds']):
        mask=z['active'][:,i]
        ax.plot(z['pos'][mask,i,0],z['pos'][mask,i,1],color=COLORS[kind],lw=1,alpha=.75)
        ax.scatter(*z['pos'][0,i],s=12,color=COLORS[kind])
        ax.scatter(*z['goals'][i],s=18,marker='x',color=COLORS[kind])
    if width is not None:
        ax.axhline(0,color='black');ax.axhline(width,color='black')
        ax.set_ylim(-.3,width+.3)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.grid(alpha=.2)

def make_gif(name,title,width=None):
    z=load(name)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    size=(700,230) if width else (600,650)
    if width:
        xmin,xmax,ymin,ymax=-4,64,-1.5,width+1.5
        scale=min(640/(xmax-xmin),120/(ymax-ymin))
        ox,oy=30-xmin*scale,150+((width/2)*scale)
    else:
        scale=5.0;ox,oy=300,340
    def point(p):return (ox+p[0]*scale,oy-p[1]*scale)
    frames=[]
    # Full simulation, uniformly sampled; playback is accelerated, clock is explicit.
    for k in np.unique(np.linspace(0,len(z['t'])-1,241).astype(int)):
        im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im)
        d.text((16,8),title,font=font,fill='black')
        d.text((16,37),f"Simulation t = {z['t'][k]:.1f} s   (accelerated playback)",font=small,fill='black')
        d.text((16,size[1]-28),'Blue: pedestrian circle   Orange: vehicle rectangle   x: goal',font=small,fill='black')
        if width:
            for y in (0,width):d.line([point((0,y)),point((60,y))],fill='#777777',width=2)
        else:
            d.ellipse((ox-250,oy-250,ox+250,oy+250),outline='#cccccc',width=1)
        for i,kind in enumerate(z['kinds']):
            x,y=point(z['goals'][i]);d.line((x-3,y-3,x+3,y+3),fill=COLORS[kind]);d.line((x-3,y+3,x+3,y-3),fill=COLORS[kind])
            if not z['active'][k,i]:continue
            p=z['pos'][k,i]
            if kind=='ped':
                x,y=point(p);r=.25*scale
                d.ellipse((x-r,y-r,x+r,y+r),fill=COLORS[kind],outline='black')
            else:
                corners=rect_corners(p,z['heading'][k,i],4.5,1.8)
                d.polygon([point(c) for c in corners],fill=COLORS[kind],outline='black')
                e=np.array([np.cos(z['heading'][k,i]),np.sin(z['heading'][k,i])])
                d.line([point(p),point(p+2*e)],fill='black',width=2)
        frames.append(im)
    palette=frames[len(frames)//2].quantize(colors=64)
    frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    frames[0].save(OUT/(name+'.gif'),save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=True,disposal=2)

def main():
    rows=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(5,1,figsize=(11,10),constrained_layout=True)
    for ax,r in zip(axes,rows[:5]):
        z=load(r['name']);paths(ax,z,r['width'])
        ax.set_title(f"{r['label']}（{r['width']:g} 米）")
        ax.text(.99,.92,f"Arrived: {r['arrived_ped']}/6 pedestrians, {r['arrived_car']}/1 car\nCollision steps: {r['collision_frames']}; intrusion: {r['boundary_max_intrusion']:.4f} m",transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.savefig(OUT/'corridor_paths.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    z=load('circle')
    for ax in axes:paths(ax,z);ax.set_aspect('equal')
    for ax in axes:
        for i,kind in enumerate(z['kinds']):
            p=z['pos'][-1,i]
            if kind=='ped':
                ax.add_patch(Circle(p,.25,color=COLORS[kind]))
            else:
                ax.add_patch(Polygon(rect_corners(p,z['heading'][-1,i],4.5,1.8),facecolor=COLORS[kind],edgecolor='black',linewidth=.4))
        ax.plot([],[],color=COLORS['ped'],label='Pedestrian')
        ax.plot([],[],color=COLORS['car'],label='Vehicle')
        ax.legend(loc='upper right',fontsize=8)
    axes[0].set_title('混合圆环对趾：完整轨迹');axes[0].set_xlim(-55,55);axes[0].set_ylim(-55,55)
    axes[1].set_title('中心交互区域');axes[1].set_xlim(-18,18);axes[1].set_ylim(-18,18)
    fig.savefig(OUT/'circle_paths.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    for name in ('corridor_narrow','corridor_blocked','corridor_no_rule','corridor_rule','circle'):
        z=load(name);arrived=(~z['active']).sum(1)
        axes[0].plot(z['t'][::10],arrived[::10],label=name)
        axes[1].plot(z['t'][::10],z['collision_count'][::10],label=name)
    axes[0].set_title('到达过程');axes[0].set_ylabel('Arrived agents');axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_title('碰撞检查（圆与矩形精确几何）');axes[1].set_ylabel('Overlapping pairs')
    for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
    fig.savefig(OUT/'diagnostics.png',dpi=300);plt.close(fig)
    fields=['name','width','arrived_ped','arrived_car','t_all_arrived','collision_frames','collision_pairs','boundary_max_intrusion','mean_speed_ped','check_dt']
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if '--plots-only' not in sys.argv:
        for name,title,w in [('corridor_narrow','窄长廊：人车相向通行',3.5),('corridor_blocked','空间不足：等待',2.2),('circle','人车混合圆环对趾',None)]:
            make_gif(name,title,w)
    print('Created 3 PNGs, 3 GIFs and metrics.csv',flush=True)

if __name__=='__main__':main()
