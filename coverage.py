import itertools, numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE=set("N1 C2 N3 C4 C5 C6 N7 C8 N9 O2 N4 O4 N6 O6 N2 C7 C5M".split())
VDW=1.70; GS=0.25

def prep(recp):
    rec=[]
    for l in open(recp):
        if l.startswith('ATOM'):
            n=l[12:16].strip()
            if not n.startswith('H'):
                rec.append((n,l[72:76].strip(),int(l[22:27]),
                            np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])))
    bas={}
    for n,s,r,x in rec:
        if n in BASE: bas.setdefault((s,r),[]).append(x)
    bc={k:np.mean(v,0) for k,v in bas.items()}
    segs=sorted({k[0] for k in bc})
    s1=sorted([k for k in bc if k[0]==segs[0]],key=lambda k:k[1])
    s2=sorted([k for k in bc if k[0]==segs[1]],key=lambda k:k[1])
    bp=np.array([(bc[a]+bc[b])/2 for a,b in zip(s1,reversed(s2))])
    ctr=bp.mean(0); ax=np.linalg.svd(bp-ctr)[2][0]
    if np.dot(bp[-1]-bp[0],ax)<0: ax=-ax
    zb=[float(np.dot(b-ctr,ax)) for b in bp]
    basex=np.array([x for n,s,r,x in rec if n in BASE])
    return rec,bc,ctr,ax,zb,basex

def poses_of(dockp):
    out=[];cur=None
    for l in open(dockp):
        if l.startswith('REMARK'):
            p=l.split(':',1); k=p[0].replace('REMARK','').strip(); v=p[1].strip() if len(p)>1 else ''
            if k=='FILE_NAME': cur=dict(atoms=[],names=[]); out.append(cur)
            elif cur is not None: cur[k]=v
        elif l.startswith('ATOM') and cur is not None:
            n=l[12:16].strip()
            if not n.startswith('H'):
                cur['names'].append(n); cur['atoms'].append(np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]))
    return out

def analyse(recp,dockp):
    rec,bc,ctr,ax,zb,basex=prep(recp)
    res=[]
    for p in poses_of(dockp):
        X=np.array(p['atoms'])
        G=nx.Graph(); G.add_nodes_from(range(len(X)))
        for i,j in itertools.combinations(range(len(X)),2):
            if np.linalg.norm(X[i]-X[j])<1.80: G.add_edge(i,j)
        ring=set()
        for c in nx.cycle_basis(G): ring.update(c)
        big=max(nx.connected_components(G.subgraph(ring)),key=len)
        C=np.array([X[i] for i in sorted(big)])
        c=C.mean(0); Vt=np.linalg.svd(C-c)[2]; nrm=Vt[2]; u=Vt[0]; v=Vt[1]
        d=c-ctr; z=float(np.dot(d,ax)); rad=float(np.linalg.norm(d-z*ax))
        up=dn=0
        for b in bc.values():
            dd=b-c; h=float(np.dot(dd,nrm)); lat=float(np.linalg.norm(dd-h*nrm))
            if 2.6<=abs(h)<=5.6 and lat<4.5:
                if h>0: up+=1
                else: dn+=1
        if not(up>=1 and dn>=1 and (min(zb)+0.5)<z<(max(zb)-0.5) and rad<5.5): continue
        A=[];B=[]
        for x in basex:
            dd=x-c; h=float(np.dot(dd,nrm))
            if 2.6<=abs(h)<=5.6 and np.linalg.norm(dd-h*nrm)<9.0:
                (A if h>0 else B).append(x)
        pc=np.stack([(C-c)@u,(C-c)@v],1)
        lo=pc.min(0)-VDW-0.5; hi=pc.max(0)+VDW+0.5
        gx=np.arange(lo[0],hi[0],GS); gy=np.arange(lo[1],hi[1],GS)
        GX,GY=np.meshgrid(gx,gy,indexing='ij'); P=np.stack([GX.ravel(),GY.ravel()],1)
        def m(pts):
            if len(pts)==0: return np.zeros(len(P),bool)
            a=np.array(pts); a2=np.stack([(a-c)@u,(a-c)@v],1)
            return (np.linalg.norm(P[:,None,:]-a2[None,:,:],axis=2)<VDW).any(1)
        L=m(C)
        cov=[(L&m(s)).sum()/max(1,L.sum()) for s in (A,B)]
        area=L.sum()*GS*GS
        res.append((float(p['MEMBER_ENERGY']),area,float(np.mean(cov)),area*float(np.mean(cov))))
    res.sort(key=lambda t:t[0]); return res

runs=[("NDI$^{2+}$","ndimet_zip","#0072B2"),("PDI$^{2+}$","pdi_zip","#D55E00")]
D={}
for lab,d,col in runs:
    r=analyse(f"{d}/files/receptor.pdb",f"{d}/result.dock4")
    D[lab]=(np.array([t[2] for t in r])*100, np.array([t[3] for t in r]),
            np.array([t[1] for t in r]), col)
    print(f"{lab:10s} n={len(r):4d}  core {np.median(D[lab][2]):5.1f} A2  "
          f"coverage {np.median(D[lab][0]):5.1f}%  stacked {np.median(D[lab][1]):5.1f} A2  "
          f"| top10 coverage {np.median(D[lab][0][:10]):5.1f}%")

plt.rcParams.update({'font.size':9,'font.family':'sans-serif','axes.linewidth':0.8})
fig,axes=plt.subplots(1,2,figsize=(6.6,3.3),constrained_layout=True)
for ax,idx,ttl,xl in ((axes[0],0,'(a)  fraction of the aromatic core\ncovered by the flanking base pairs','coverage / %'),
                      (axes[1],1,'(b)  absolute stacked area','stacked area / $\\AA^2$')):
    for i,(lab,(cov,st,area,col)) in enumerate(D.items()):
        v=(cov,st)[idx]
        parts=ax.violinplot([v],positions=[i],widths=0.72,showextrema=False,showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(col); b.set_alpha(0.25); b.set_edgecolor(col); b.set_linewidth(1.4)
        q1,med,q3=np.percentile(v,[25,50,75])
        ax.plot([i,i],[q1,q3],color=col,lw=5,solid_capstyle='round',alpha=0.85,zorder=3)
        ax.plot([i],[med],'o',ms=7,mfc='white',mec=col,mew=2,zorder=4)
        jitter=(np.random.RandomState(0).rand(min(10,len(v)))-0.5)*0.22
        ax.plot(i+jitter,v[:10],'o',ms=4.5,mfc=col,mec='black',mew=0.6,ls='none',alpha=0.95,zorder=5)
        ax.text(i,med,f'  {med:.0f}'+('%' if idx==0 else ''),fontsize=8,va='center',ha='left',color='0.25')
    ax.set_xticks(range(len(D))); ax.set_xticklabels(list(D.keys()))
    ax.set_ylabel(xl); ax.set_title(ttl,fontsize=8.6,pad=6)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='0.93',lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=3)
axes[0].set_ylim(0,100)
fig.text(0.5,-0.04,'violin = all intercalated poses;  bar = interquartile range;  white dot = median;  '
         'small dots = the 10 best-ranked poses',ha='center',fontsize=6.9,color='0.45')
fig.savefig('coverage.png',dpi=300,bbox_inches='tight',facecolor='white')
print('wrote coverage.png')
