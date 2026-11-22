from pathlib import Path
import argparse,json,time,traceback,hashlib
import mph,numpy as np
REPO_ROOT=Path(__file__).resolve().parents[3]
BASE=REPO_ROOT/'working'/'comsol'/'giesekus'
OUT=BASE/'FINAL_OUTPUT'
SPECS={
'coarse':dict(hmax='1.8[mm]',hmin='0.06[mm]',ohmax='0.08[mm]',ohmin='0.03[mm]',bl='0.03[mm]',layers='3'),
'fine':dict(hmax='0.9[mm]',hmin='0.03[mm]',ohmax='0.04[mm]',ohmin='0.015[mm]',bl='0.015[mm]',layers='4')}
def scalar(m,e,u=None):
 a=np.asarray(m.evaluate(e,unit=u,inner='last'),float).ravel();a=a[np.isfinite(a)];return float(np.mean(a))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--mesh',choices=SPECS,required=True);ap.add_argument('--port',type=int,default=2036);a=ap.parse_args();spec=SPECS[a.mesh]
 c=mph.Client(port=a.port);m=c.load(str(BASE/'Model_1'/'models'/f'model1_{a.case}.mph'));j=m.java;me=j.component('comp1').mesh('mesh1');sg=me.feature('ftri1').feature('sg');so=me.feature('ftri1').feature('so');bp=me.feature('bl1').feature('bp1')
 for k,v in [('hmax',spec['hmax']),('hmin',spec['hmin'])]:sg.set(k,v)
 for k,v in [('hmax',spec['ohmax']),('hmin',spec['ohmin'])]:so.set(k,v)
 bp.set('blhmin',spec['bl']);bp.set('blnlayers',spec['layers']);me.run();j.sol('sol1').clearSolution();meta={'State_ID':a.case,'mesh_level':a.mesh,'status':'RUNNING'};t=time.time()
 try:
  j.sol('sol1').runAll();Qin=scalar(m,'Qin','m^3/s');Qout=scalar(m,'Qout','m^3/s');N1=scalar(m,'2*iexit(r*(vef.Te_1zz-vef.Te_1rr))/Ro^2','Pa');r=np.asarray(m.evaluate('r',unit='mm',inner='last'),float).ravel();mp=OUT/'models'/f'model1_{a.case}_{a.mesh}.mph';m.save(str(mp));meta.update(status='SOLVED',DeltaP_Pa=scalar(m,'Pin-Pout','Pa'),N1_exit_Pa=N1,Qin_m3_s=Qin,Qout_m3_s=Qout,mass_error=abs(Qin-Qout)/max(abs(Qin),abs(Qout),1e-30),n_field_nodes=int(np.isfinite(r).sum()),elapsed_s=time.time()-t,model=str(mp),sha256=sha(mp))
 except Exception as e:meta.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.time()-t)
 out=OUT/'data'/f'mesh_{a.case}_{a.mesh}.json';out.write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2),flush=True);c.clear()
if __name__=='__main__':main()
