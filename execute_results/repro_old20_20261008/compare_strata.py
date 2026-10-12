import importlib.util, numpy as np, sys
spec=importlib.util.spec_from_file_location('c','/workspace/EVMALOCALUPDATE/execute_results/repro_old20_20261008/compare_old_new.py'); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
P2E=c.P2E
for name,paths,kind in (('old',c.OLD,'old'),('new',c.NEW,'new')):
    A={}
    def add(k,v): A.setdefault(k,[]).append(v)
    strata={}
    for p in paths:
        s,e=c.unify(p,kind); steps=s['step'].astype(int); idx={t:i for i,t in enumerate(steps)}; n=len(steps)
        ii=np.array([idx[t] for t in e['step'].astype(int)])
        nominal=np.clip(e['action'],-1,1)*e['rated']; d=nominal-e['ex']
        add('EV-steps SoC>=99.9 %',100*(e['soc']>=99.9).mean()); add('EV-steps SoC<=0.1 %',100*(e['soc']<=0.1).mean())
        add('EV-steps at/above target %',100*(e['soc']>=e['target']-1e-6).mean())
        cu=np.zeros(n); cd=np.zeros(n)
        np.add.at(cu,ii,np.maximum(d,0)); np.add.at(cd,ii,np.maximum(-d,0))
        on=s['on']; band=s['band']; resid=s['resid']; req=s['req']
        add('clip: proposed charge not executed kW',cu[on].mean()); add('clip: proposed discharge not executed kW',cd[on].mean())
        # who gets clipped up: EVs full; down: EVs empty
        full=e['soc']>=99.9; empty=e['soc']<=0.1
        add('share of charge-clip from full EVs %',100*np.maximum(d,0)[full].sum()/max(np.maximum(d,0).sum(),1e-9))
        add('share of discharge-clip from empty EVs %',100*np.maximum(-d,0)[empty].sum()/max(np.maximum(-d,0).sum(),1e-9))
        fl=np.zeros(n); np.add.at(fl,ii,np.maximum(e['floor'],e['lo']))
        fmin=np.zeros(n); fmax=np.zeros(n); np.add.at(fmin,ii,e['lo']); np.add.at(fmax,ii,e['hi'])
        conflict=(req+band<fl); clipnet=np.abs(cu-cd); big=clipnet>band
        ok=np.abs(resid)<=band
        for kc in (False,True):
            for kb in (False,True):
                m=on&(conflict==kc)&(big==kb)
                st=strata.setdefault((kc,kb),[0,0]); st[0]+=m.sum(); st[1]+=(ok&m).sum()
        span=np.maximum(fmax-fmin,1e-6)
        add('pass if band = 6.98% of fleet range %',100*(np.abs(resid)<=0.0698*span)[on].mean())
    print('==',name)
    for k,v in A.items(): print(f'  {k:50s} {np.mean(v):9.2f}')
    tot=sum(v[0] for v in strata.values())
    for (kc,kb),(n_,ok_) in sorted(strata.items()):
        print(f'  conflict={kc!s:5s} clip>band={kb!s:5s}: share {100*n_/tot:5.1f}% of assessed steps, tracking {100*ok_/max(n_,1):5.1f}%')
