from pathlib import Path
import argparse,hashlib,json,time,traceback
import mph,numpy as np,pandas as pd
REPO_ROOT=Path(__file__).resolve().parents[3]
SRC=REPO_ROOT/'external_data'/'simulation'/'generalized_newtonian'
OUT=REPO_ROOT/'working'/'comsol'/'generalized_newtonian'
(OUT/'models').mkdir(parents=True,exist_ok=True);(OUT/'fields').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
ST=REPO_ROOT/'data'/'state_space'/'MODEL_A_STATES.csv'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def box(c,tag,dim,x0,x1,y0,y1,cond='intersects'):
 s=c.selection().create(tag,'Box');s.geom('geom1',dim)
 for k,v in [('xmin',x0),('xmax',x1),('ymin',y0),('ymax',y1)]:s.set(k,v)
 s.set('condition',cond);return s
def scalar(x):
 a=np.asarray(x,dtype=float).ravel();a=a[np.isfinite(a)];return float(np.mean(a)) if len(a) else np.nan
def ev(m,e,u=None):return scalar(m.evaluate(e,unit=u,inner='last'))
def build(c):
 m=c.create('FIG4_MODEL_A_NOZZLE_EXTRUSION');j=m.java;p=j.param()
 for k,v in {'rho':'1000[kg/m^3]','tau_y':'30[Pa]','K_hb':'40[Pa*s^n_hb]','n_hb':'0.45','Q_assumed_prescribed':'4.43647212844253e-8[m^3/s]','epsilon_reg':'0.01[1/s]','mu_floor':'1e-3[Pa*s]','mu_cap':'1e6[Pa*s]','Rs':'15[mm]','Rt':'3[mm]','Ro':'0.42[mm]','Ls':'25[mm]','Lt':'2[mm]','Ln':'34[mm]'}.items():p.set(k,v)
 c1=j.component().create('comp1',True);g=c1.geom().create('geom1',2);g.axisymmetric(True);g.lengthUnit('mm');poly=g.feature().create('flow','Polygon');poly.set('x',['0','Rs','Rs','Rt','Ro','0']);poly.set('y',['0','0','Ls','Ls+Lt','Ls+Lt+Ln','Ls+Lt+Ln']);g.run()
 box(c1,'sel_in',1,'-1[um]','Rs+1[um]','-1[um]','1[um]','inside');box(c1,'sel_out',1,'-1[um]','Ro+1[um]','Ls+Lt+Ln-1[um]','Ls+Lt+Ln+1[um]','inside');box(c1,'sel_wall',1,'0.01[mm]','Rs+1[um]','0.01[mm]','Ls+Lt+Ln-0.01[mm]')
 for tag,sel in [('iin','sel_in'),('iout','sel_out'),('iwall','sel_wall')]:op=c1.cpl().create(tag,'Integration');op.selection().named(sel)
 v=c1.variable().create('var1');
 for k,x in {'gamma_reg':'sqrt(spf.sr^2+epsilon_reg^2)','eta_hb':'min(mu_cap,max(mu_floor,tau_y/gamma_reg+K_hb*gamma_reg^(n_hb-1)))','Uin':'Q_assumed_prescribed/(pi*Rs^2)','Qin':'2*pi*iin(r*w)','Qout':'2*pi*iout(r*w)','Pin':'2*iin(r*p)/Rs^2','Pout':'2*iout(r*p)/Ro^2','Aout':'pi*Ro^2','Wout':'2*iout(r*w)/Ro^2','W2out':'2*iout(r*w^2)/Ro^2','OutletCV':'sqrt(max(0,W2out-Wout^2))/max(abs(Wout),1e-15[m/s])'}.items():v.set(k,x)
 spf=c1.physics().create('spf','LaminarFlow','geom1');fp=spf.feature('fp1');fp.set('rho_mat','userdef');fp.set('rho','rho');fp.set('mu_mat','userdef');fp.set('mu','eta_hb');inc=spf.create('in1','Inlet',1);inc.selection().named('sel_in');inc.set('BoundaryCondition','Velocity');inc.set('U0in','Uin');out=spf.create('out1','Outlet',1);out.selection().named('sel_out');out.set('BoundaryCondition','PressureNoViscousStress');out.set('p0','0[Pa]')
 me=c1.mesh().create('mesh1','geom1');tri=me.feature().create('ftri1','FreeTri');sz=tri.create('sg','Size');sz.set('custom','on');sz.set('hmax','0.5[mm]');sz.set('hmin','0.015[mm]');sz.set('hgrad','1.25');so=tri.create('so','Size');so.selection().named('sel_out');so.set('custom','on');so.set('hmax','0.018[mm]');so.set('hmin','0.008[mm]');so.set('hgrad','1.15');bl=me.feature().create('bl1','BndLayer');bl.selection().geom('geom1',2);bl.selection().all();bp=bl.feature().create('bp1','BndLayerProp');bp.selection().named('sel_wall');bp.set('blnlayers','5');bp.set('blstretch','1.2');bp.set('blhmin','0.01[mm]');me.run()
 st=j.study().create('std1');stat=st.create('stat','Stationary');st.createAutoSequences('all');j.sol('sol1').feature('s1').set('stol','1e-6');print('STATIONARY_SUBFEATURES',list(j.sol('sol1').feature('s1').feature().tags()),flush=True);return m

def wallq(m,q):
 lo,hi=0.,1.
 for _ in range(30):
  if ev(m,f'iwall(if(eta_hb*spf.sr<={hi}[Pa],1,0))/iwall(1)')>=q:break
  hi*=10
 for _ in range(22):
  md=(lo+hi)/2
  if ev(m,f'iwall(if(eta_hb*spf.sr<={md}[Pa],1,0))/iwall(1)')<q:lo=md
  else:hi=md
 return hi

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,required=True);a=ap.parse_args();states=pd.read_csv(ST);states=states[states.Case_ID.isin(['A_LOW','A_OPT','A_HIGH'])];cl=mph.Client(port=a.port);m=build(cl);j=m.java;m.save(str(OUT/'model_A_template.mph'));rows=[]
 for r in states.itertuples(index=False):
  t=time.time();p=j.param();p.set('tau_y',f'{r.tau_y_Pa:.16g}[Pa]');p.set('K_hb',f'{r.K_Pa_s_n:.16g}[Pa*s^n_hb]');p.set('n_hb',f'{r.n:.16g}');p.set('Q_assumed_prescribed',f'{r.Q_assumed_prescribed_m3_s:.16g}[m^3/s]')
  try:
   j.sol('sol1').runAll();qin=ev(m,'Qin','m^3/s');qout=ev(m,'Qout','m^3/s');pin=ev(m,'Pin','Pa');pout=ev(m,'Pout','Pa');vals={'Case_ID':r.Case_ID,'State_role':r.State_role,'Rheology_ID':r.Rheology_ID,'solver_status':'SOLVED','tau_y_Pa':r.tau_y_Pa,'K_Pa_s_n':r.K_Pa_s_n,'n':r.n,'Q_prescribed_m3_s':r.Q_assumed_prescribed_m3_s,'Q_in_m3_s':qin,'Q_out_m3_s':qout,'mass_balance_error':abs(qout-qin)/max(abs(qout),abs(qin),1e-30),'inlet_vs_prescribed_error':abs(qin-r.Q_assumed_prescribed_m3_s)/r.Q_assumed_prescribed_m3_s,'DeltaP_Pa':pin-pout,'Outlet_velocity_CV':ev(m,'OutletCV','1'),'Wall_stress_p95_Pa':wallq(m,.95),'Wall_stress_p99_Pa':wallq(m,.99),'shear_rate_max_s_1':float(np.nanmax(m.evaluate('spf.sr',inner='last'))),'stationary_relative_solution_tolerance':1e-6,'tolerance_property':'stol','fully_coupled_requested':True,'boundary_layers':5,'boundary_layer_growth':1.2,'outlet_hmax_mm':.018,'nominal_cells_across_outlet':47,'elapsed_s':time.time()-t}
   dat=np.asarray(m.evaluate(['r','z','u','w','p','spf.sr','eta_hb'],inner='last'),float);dat=dat.T if dat.shape[0]==7 else dat;pd.DataFrame(dat,columns=['r_mm','z_mm','u_r_m_s','u_z_m_s','p_Pa','shear_rate_s_1','eta_Pa_s']).to_csv(OUT/'fields'/f'{r.Case_ID}_model_A_field.csv.gz',index=False,compression='gzip');mp=OUT/'models'/f'model_A_{r.Case_ID}.mph';m.save(str(mp));vals['model_path']=str(mp);vals['model_sha256']=sha(mp)
  except Exception as e:vals={'Case_ID':r.Case_ID,'State_role':r.State_role,'solver_status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.time()-t}
  rows.append(vals);pd.DataFrame(rows).to_csv(OUT/'pressure.csv',index=False);print(json.dumps(vals,indent=2),flush=True)
 (OUT/'MODEL_A_SOLVER_SETTINGS.txt').write_text('Stationary\nStudy usertol=true\nRelative tolerance=1e-5\nNonlinear method=COMSOL auto sequence with fully coupled requested\nAbsolute tolerance=automatic\n',encoding='utf-8');cl.clear()
if __name__=='__main__':main()

