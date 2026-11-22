from pathlib import Path
import argparse,time,traceback,json,hashlib
import mph,numpy as np,pandas as pd
REPO_ROOT=Path(__file__).resolve().parents[3]
ST=REPO_ROOT/'data'/'state_space'/'VISCOELASTIC_STATES.csv'
OUT=REPO_ROOT/'working'/'comsol'/'giesekus'/'Model_1'
for x in [OUT,OUT/'models',OUT/'fields',OUT/'logs']:x.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def box(comp,tag,dim,x0,x1,y0,y1,cond='intersects'):
 s=comp.selection().create(tag,'Box');s.geom('geom1',dim)
 for k,v in [('xmin',x0),('xmax',x1),('ymin',y0),('ymax',y1)]:s.set(k,v)
 s.set('condition',cond);return s
def scalar(a):
 x=np.asarray(a,float).ravel();x=x[np.isfinite(x)];return float(np.mean(x)) if len(x) else np.nan
def ev(m,e,u=None):return scalar(m.evaluate(e,unit=u,inner='last'))
def build(client,row):
 m=client.create('FIG4_GIESEKUS_NOZZLE');j=m.java;p=j.param()
 pars={'rho':'1000[kg/m^3]','Qpres':'19.33[mm^3/s]','Rs':'15[mm]','Rt':'3[mm]','Ro':'0.42[mm]','Ls':'25[mm]','Lt':'2[mm]','Ln':'34[mm]','mu_s':f'{row.mu_solvent_Pa_s:.16g}[Pa*s]','mu_e':f'{row.eta_polymer_Pa_s:.16g}[Pa*s]','lambda_e':f'{row.lambda_app_s:.16g}[s]','alpha_e':f'{row.giesekus_alpha:.16g}','tramp':'0.015[s]'}
 for k,v in pars.items():p.set(k,v)
 comp=j.component().create('comp1',True);g=comp.geom().create('geom1',2);g.axisymmetric(True);g.lengthUnit('mm');poly=g.feature().create('flow','Polygon');poly.set('x',['0','Rs','Rs','Rt','Ro','0']);poly.set('y',['0','0','Ls','Ls+Lt','Ls+Lt+Ln','Ls+Lt+Ln']);g.run()
 box(comp,'sel_in',1,'-1[um]','Rs+1[um]','-1[um]','1[um]','inside');box(comp,'sel_out',1,'-1[um]','Ro+1[um]','Ls+Lt+Ln-1[um]','Ls+Lt+Ln+1[um]','inside');box(comp,'sel_wall',1,'0.001[mm]','Rs+1[um]','0.001[mm]','Ls+Lt+Ln-0.001[mm]')
 for tag,sel in [('iin','sel_in'),('iout','sel_out'),('iexit','sel_out')]:op=comp.cpl().create(tag,'Integration');op.selection().named(sel)
 v=comp.variable().create('var1');
 for k,x in {'Uin':'Qpres/(pi*Rs^2)','rampQ':'1-exp(-t/tramp)','Qin':'2*pi*iin(r*w)','Qout':'2*pi*iout(r*w)','Pin':'2*iin(r*p)/Rs^2','Pout':'2*iout(r*p)/Ro^2','N1_exit':'2*iexit(r*(vef.Tzz-vef.Trr))/Ro^2'}.items():v.set(k,x)
 vef=comp.physics().create('vef','ViscoelasticFlow','geom1');fp=vef.feature('fp1');fp.set('rho_mat','userdef');fp.set('rho','rho');fp.set('Constitutiverelation','Viscoelastic');fp.set('MaterialModelViscoelastic','Giesekus');fp.set('mu_s_mat','userdef');fp.set('mu_s','mu_s');fp.set('mueGiesekus','mu_e');fp.set('lambdaeGiesekus','lambda_e');fp.set('alphaeGiesekus','alpha_e')
 wall=vef.feature('wallbc1');inc=vef.create('inl1','InletBoundary',1);inc.selection().named('sel_in');inc.set('BoundaryCondition','Velocity');inc.set('ComponentWise','NormalInflowVelocity');inc.set('U0in','Uin*rampQ');inc.set('Te0','0');out=vef.create('out1','OutletBoundary',1);out.selection().named('sel_out');out.set('BoundaryCondition','Pressure');out.set('p0','0[Pa]');out.set('SuppressBackflow','1')
 me=comp.mesh().create('mesh1','geom1');tri=me.feature().create('ftri1','FreeTri');sz=tri.create('sg','Size');sz.set('custom','on');sz.set('hmax','1.2[mm]');sz.set('hmin','0.04[mm]');sz.set('hgrad','1.25');so=tri.create('so','Size');so.selection().named('sel_out');so.set('custom','on');so.set('hmax','0.05[mm]');so.set('hmin','0.02[mm]');so.set('hgrad','1.15');bl=me.feature().create('bl1','BndLayer');bl.selection().geom('geom1',2);bl.selection().all();bp=bl.feature().create('bp1','BndLayerProp');bp.selection().named('sel_wall');bp.set('blnlayers','3');bp.set('blstretch','1.2');bp.set('blhmin','0.02[mm]');me.run()
 st=j.study().create('std1');tm=st.create('time','Transient');tm.set('tlist','0 0.02 0.05 0.1 0.15');tm.set('usertol',True);tm.set('rtol','2e-3');st.createAutoSequences('all');return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=2036);ap.add_argument('--case',default='LOW_ELASTICITY');ap.add_argument('--solve',action='store_true');a=ap.parse_args();row=pd.read_csv(ST).query('State_ID==@a.case').iloc[0];client=mph.Client(port=a.port);m=build(client,row);mp=OUT/'models'/f'model1_{a.case}_template.mph';m.save(str(mp));print('BUILT',mp,flush=True)
 if a.solve:
  j=m.java
  try:
   client.java.showProgress(str(OUT/'logs'/f'{a.case}.log'))
  except:pass
  t=time.time();meta={'State_ID':a.case,'Rheology_ID':row.Rheology_ID,'status':'RUNNING'}
  try:
   j.sol('sol1').runAll();qin=ev(m,'Qin','m^3/s');qout=ev(m,'Qout','m^3/s');meta.update(status='SOLVED',DeltaP_Pa=ev(m,'Pin-Pout','Pa'),Qin_m3_s=qin,Qout_m3_s=qout,mass_error=abs(qin-qout)/max(abs(qin),abs(qout),1e-30),N1_exit_Pa=ev(m,'N1_exit','Pa'),elapsed_s=time.time()-t);dat=np.asarray(m.evaluate(['r','z','u','w','p','vef.Trr','vef.Tzz','vef.Trz'],inner='last'),float);dat=dat.T if dat.shape[0]==8 else dat;pd.DataFrame(dat,columns=['r_mm','z_mm','u_m_s','w_m_s','p_Pa','Trr_Pa','Tzz_Pa','Trz_Pa']).to_csv(OUT/'fields'/f'{a.case}_field.csv.gz',index=False,compression='gzip');mp=OUT/'models'/f'model1_{a.case}.mph';m.save(str(mp));meta['model']=str(mp);meta['sha256']=sha(mp)
  except Exception as e:meta.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.time()-t)
  (OUT/f'{a.case}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2),flush=True)
 client.clear()
if __name__=='__main__':main()


