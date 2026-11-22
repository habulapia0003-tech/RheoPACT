from pathlib import Path
import argparse,json,mph,numpy as np,pandas as pd
REPO_ROOT=Path(__file__).resolve().parents[3]
BASE=REPO_ROOT/'working'/'comsol'/'giesekus'/'Model_1'
def arr(m,e,u=None):
 a=np.asarray(m.evaluate(e,unit=u,inner='last'),float).ravel();return a
def scalar(m,e,u=None):
 a=arr(m,e,u);a=a[np.isfinite(a)];return float(np.mean(a))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=2036);ap.add_argument('--case',required=True);a=ap.parse_args();c=mph.Client(port=a.port);m=c.load(str(BASE/'models'/f'model1_{a.case}.mph'))
 Qin,Qout=scalar(m,'Qin','m^3/s'),scalar(m,'Qout','m^3/s');Pin,Pout=scalar(m,'Pin','Pa'),scalar(m,'Pout','Pa');N1=scalar(m,'2*iexit(r*(vef.Te_1zz-vef.Te_1rr))/Ro^2','Pa')
 expr=[('r_mm','r','mm'),('z_mm','z','mm'),('u_m_s','u','m/s'),('w_m_s','w','m/s'),('p_Pa','p','Pa'),('Te_rr_Pa','vef.Te_1rr','Pa'),('Te_zz_Pa','vef.Te_1zz','Pa'),('Te_rz_Pa','vef.Te_1rz','Pa'),('Te_phiphi_Pa','vef.Te_1phiphi','Pa')];data={}
 for name,e,u in expr:data[name]=arr(m,e,u);print(name,len(data[name]),flush=True)
 df=pd.DataFrame(data);df['N1_Pa']=df.Te_zz_Pa-df.Te_rr_Pa;df.to_csv(BASE/'fields'/f'{a.case}_field.csv.gz',index=False,compression='gzip')
 summary={'State_ID':a.case,'Qin_m3_s':Qin,'Qout_m3_s':Qout,'mass_error':abs(Qin-Qout)/max(abs(Qin),abs(Qout),1e-30),'Pin_Pa':Pin,'Pout_Pa':Pout,'DeltaP_Pa':Pin-Pout,'N1_exit_area_mean_Pa':N1,'N1_field_max_Pa':float(np.nanmax(df.N1_Pa)),'velocity_max_m_s':float(np.nanmax(np.hypot(df.u_m_s,df.w_m_s))),'n_nodes':len(df)}
 (BASE/f'{a.case}_final.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True);c.clear()
if __name__=='__main__':main()
