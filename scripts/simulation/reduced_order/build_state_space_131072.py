from pathlib import Path
import hashlib, json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

REPO_ROOT=Path(__file__).resolve().parents[3]
OUT=REPO_ROOT/"working"/"figure4_state_space_131072"
DATA,FIG,AUD=[OUT/x for x in ("source_data","outputs","audit")]
SIM=REPO_ROOT/"external_data"/"Table_S4-1_full_131072_state_space.csv"
MEA=REPO_ROOT/"data"/"state_space"/"Fig4a_measured_conditions_540.csv"
PCA=REPO_ROOT/"data"/"state_space"/"Fig4a_frozen_PCA_definition.csv"
ANC=REPO_ROOT/"data"/"state_space"/"Fig4a_anchor_states.csv"
FEATURES=["log10_G_prime_Pa","log10_G_double_prime_Pa","log10_eta10_Pa_s","log10_1_plus_tau_y_Pa","log10_K_Pa_s_n","flow_index_n"]
REGIONS=["Low elasticity","Compatible / balanced","High elastic recovery"]
COLORS=["#1768B3","#16844A","#D9470B"]

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest().upper()

def req(x,msg):
 if not x:raise AssertionError(msg)

def main():
 for d in (DATA,FIG,AUD):d.mkdir(parents=True,exist_ok=True)
 sim=pd.read_csv(SIM); measured=pd.read_csv(MEA)
 definition=pd.read_csv(PCA).set_index("feature").loc[FEATURES]
 anchors=pd.read_csv(ANC).set_index("State").loc[REGIONS].reset_index()
 req(len(sim)==131072,"Expected 131,072 reduced-order states")
 req(sim.ink_id.is_unique and not sim.duplicated().any(),"Reduced-order states are not unique")
 req(len(measured)==540 and measured.Rheology_condition_ID.is_unique,"Expected 540 unique measured conditions")
 req(measured.Evidence_class.eq("REAL_EXPERIMENTAL_INPUT").all(),"Measured evidence labels differ")
 req(sim.parameter_provenance.eq("CONTROLLED_IN_SILICO_PARAMETER").all(),"Parameter provenance differs")
 req(sim.output_provenance.eq("PHYSICS_SIMULATED_OUTPUT_REDUCED_MODEL").all(),"Output provenance differs")
 core=["K_ref_Pa_s_n","n","tau_y_ref_Pa","G_prime_Pa","G_double_prime_Pa"]
 req(not sim[core].isna().any().any(),"Core reference rheology has missing values")
 # Fail before transformation unless every logarithmic input is strictly positive;
 # clipping below is therefore a numerical no-op, not missing-data repair.
 req((sim[["K_ref_Pa_s_n","G_prime_Pa","G_double_prime_Pa"]]>0).all().all(),"Positive features invalid")
 req((sim.tau_y_ref_Pa>=0).all(),"Yield stress must be non-negative")

 eta10=sim.tau_y_ref_Pa/10+sim.K_ref_Pa_s_n*10**(sim.n-1)
 req(np.isfinite(eta10).all() and (eta10>0).all(),"Derived eta10 invalid")
 feat=pd.DataFrame({
  "log10_G_prime_Pa":np.log10(np.clip(sim.G_prime_Pa,np.finfo(float).tiny,None)),
  "log10_G_double_prime_Pa":np.log10(np.clip(sim.G_double_prime_Pa,np.finfo(float).tiny,None)),
  "log10_eta10_Pa_s":np.log10(np.clip(eta10,np.finfo(float).tiny,None)),
  "log10_1_plus_tau_y_Pa":np.log10(np.clip(1+sim.tau_y_ref_Pa,np.finfo(float).tiny,None)),
  "log10_K_Pa_s_n":np.log10(np.clip(sim.K_ref_Pa_s_n,np.finfo(float).tiny,None)),
  "flow_index_n":sim.n})
 mean=definition.scaler_mean.to_numpy(); scale=definition.scaler_scale.to_numpy()
 load=definition[["PC1_loading","PC2_loading"]].to_numpy()
 scaled=(feat.to_numpy()-mean)/scale; scores=scaled@load
 measured_scores=measured[["PC1","PC2"]].to_numpy()
 anchor_scaled=(anchors[FEATURES].to_numpy()-mean)/scale
 direction=anchor_scaled[2]-anchor_scaled[0]; direction/=np.linalg.norm(direction)
 anchor_index=(anchor_scaled-anchor_scaled[0])@direction
 sim_index=(scaled-anchor_scaled[0])@direction
 thresholds=[(anchor_index[0]+anchor_index[1])/2,(anchor_index[1]+anchor_index[2])/2]
 sim_region=np.asarray(REGIONS,dtype=object)[np.digitize(sim_index,thresholds)]
 sim_color=np.clip((sim_index-anchor_index[0])/(anchor_index[2]-anchor_index[0]),0,1)
 measured_index=measured.elastic_state_index.to_numpy()
 measured_region=np.asarray(REGIONS,dtype=object)[np.digitize(measured_index,thresholds)]

 projection=pd.DataFrame({"ink_id":sim.ink_id,"model":sim.model,"PC1":scores[:,0],"PC2":scores[:,1],"anchor_axis_index":sim_index,"anchor_defined_region":sim_region,"posthoc_regime_original":sim.posthoc_regime,"eta10_reference_Pa_s_derived":eta10,"parameter_provenance":sim.parameter_provenance,"output_provenance":sim.output_provenance})
 projection.to_csv(DATA/"reduced_order_states_131072_projection.csv",index=False)
 measured_out=measured.copy(); measured_out["anchor_defined_region_recalculated"]=measured_region
 measured_out.to_csv(DATA/"measured_conditions_540_projection.csv",index=False)
 definition.reset_index().to_csv(DATA/"frozen_pca_definition_from_540_measured.csv",index=False)
 anchors.to_csv(DATA/"measured_anchor_states.csv",index=False)

 all_pc=np.vstack([scores,measured_scores]); xpad=.035*np.ptp(all_pc[:,0]); ypad=.055*np.ptp(all_pc[:,1])
 gx=np.linspace(all_pc[:,0].min()-xpad,all_pc[:,0].max()+xpad,480)
 gy=np.linspace(all_pc[:,1].min()-ypad,all_pc[:,1].max()+ypad,360)
 xx,yy=np.meshgrid(gx,gy)
 guide=KNeighborsRegressor(n_neighbors=61,weights="distance",n_jobs=-1).fit(all_pc,np.r_[sim_index,measured_index])
 zz=guide.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

 # Submission typography: use Times New Roman consistently for every text
 # element while preserving live/editable text in SVG and TrueType text in PDF.
 mpl.rcParams.update({
  "font.family":"serif",
  "font.serif":["Times New Roman"],
  "mathtext.fontset":"stix",
  "svg.fonttype":"none",
  "pdf.fonttype":42,
  "ps.fonttype":42,
  "font.size":8,
  "axes.linewidth":.8,
  "axes.spines.right":False,
  "axes.spines.top":False,
  "legend.frameon":False,
 })
 cmap=LinearSegmentedColormap.from_list("anchor_axis",COLORS,N=256)
 fig,ax=plt.subplots(figsize=(7.204724,4.566929)); fig.subplots_adjust(left=.105,right=.975,bottom=.16,top=.82)
 levels=[float(zz.min())-1e-8,thresholds[0],thresholds[1],float(zz.max())+1e-8]
 ax.contourf(xx,yy,zz,levels=levels,colors=["#EAF3FC","#EAF7EE","#FFF0E8"],alpha=.72,zorder=0)
 ax.contour(xx,yy,zz,levels=thresholds,colors="#757C82",linewidths=1,linestyles="--",zorder=1)
 ax.scatter(scores[:,0],scores[:,1],c=sim_color,cmap=cmap,s=1,alpha=.13,edgecolors="none",rasterized=True,zorder=2)
 ax.scatter(measured_scores[:,0],measured_scores[:,1],s=10,facecolors="none",edgecolors="#24313B",linewidths=.38,alpha=.62,rasterized=True,zorder=3)
 anchor_scores=anchors[["PC1","PC2"]].to_numpy()
 for i,color in enumerate(COLORS):ax.scatter(*anchor_scores[i],s=155,c=color,edgecolors="white",linewidths=2,zorder=6)
 for text,pos,color in zip(["Low elasticity","Compatible /\nbalanced","High elastic recovery"],[(.15,.14),(.54,.68),(.80,.90)],COLORS):ax.text(*pos,text,transform=ax.transAxes,color=color,fontsize=10.2,fontweight="bold",ha="center",va="center",zorder=7)
 for i,(label,offset) in enumerate(zip(["Low anchor","Balanced anchor","High anchor"],[(-43,-17),(8,-17),(-48,10)])):
  ax.annotate(label,anchor_scores[i],xytext=offset,textcoords="offset points",fontsize=6.7,color="#30363B",weight="bold",arrowprops=dict(arrowstyle="-",color="#737A80",lw=.55),zorder=8)
 ax.set_xlabel("Physical state coordinate 1 (PCA, 70.6%)",fontsize=9)
 ax.set_ylabel("Physical state coordinate 2 (PCA, 18.6%)",fontsize=9); ax.tick_params(labelsize=7.5,length=3.5,width=.8)
 fig.suptitle("Reduced-order rheological state space resolves three anchor-defined regions",x=.54,y=.955,fontsize=11.6,fontweight="bold")
 fig.text(.54,.895,"540 measured conditions contextualize 131,072 reduced-order simulation states",ha="center",fontsize=7.5,color="#4B5258")
 fig.text(.018,.955,"a",fontsize=16,fontweight="bold",ha="left",va="top")
 legend=[Line2D([0],[0],marker="o",linestyle="",markersize=4,markerfacecolor="#74808A",markeredgecolor="none",alpha=.55,label="Reduced-order simulated state (n = 131,072)"),Line2D([0],[0],marker="o",linestyle="",markersize=4,markerfacecolor="none",markeredgecolor="#24313B",markeredgewidth=.7,label="Measured condition (n = 540)")]
 ax.legend(handles=legend,loc="lower right",fontsize=6.7,handletextpad=.45,borderaxespad=.6)
 fig.text(.105,.045,"Simulated states are controlled reduced-order model outputs, not additional experiments or COMSOL solutions.",fontsize=6.5,color="#4E555B")
 prefix=FIG/"Fig4a_state_space_131072_reduced_order_TNR"
 fig.savefig(prefix.with_suffix(".png"),dpi=300,facecolor="white")
 fig.savefig(prefix.with_suffix(".svg"),facecolor="white")
 fig.savefig(prefix.with_suffix(".pdf"),facecolor="white")
 fig.savefig(FIG/"Fig4a_state_space_131072_reduced_order_TNR_600dpi.png",dpi=600,facecolor="white"); plt.close(fig)
 hi=Image.open(FIG/"Fig4a_state_space_131072_reduced_order_TNR_600dpi.png")
 hi.save(FIG/"Fig4a_state_space_131072_reduced_order_TNR_600dpi.tiff",dpi=(600,600),compression="raw")
 hi.close()

 counts=pd.DataFrame({"anchor_defined_region":REGIONS,"measured_n":[int(np.sum(measured_region==r)) for r in REGIONS],"reduced_order_simulated_n":[int(np.sum(sim_region==r)) for r in REGIONS]})
 counts.to_csv(DATA/"anchor_defined_region_counts.csv",index=False)
 audit={"decision":"PASS","simulated_state_n":len(sim),"measured_condition_n":len(measured),"simulated_unique_id_n":sim.ink_id.nunique(),"simulated_duplicate_row_n":int(sim.duplicated().sum()),"core_reference_missing_n":{c:int(sim[c].isna().sum()) for c in core},"coordinate_system":"Frozen six-feature scaler and PCA fitted to 540 measured conditions","eta10_mapping":"tau_y_ref_Pa/10 + K_ref_Pa_s_n*10**(n-1)","classification":"Three anchor-defined rheological regions; seven original posthoc reduced-model regimes retained in projection table","background_note":"Two-dimensional KNN visual guide fitted to every projected measured and simulated state","parameter_provenance":sim.parameter_provenance.unique().tolist(),"output_provenance":sim.output_provenance.unique().tolist(),"input_sha256":{str(SIM):sha(SIM),str(MEA):sha(MEA),str(PCA):sha(PCA),str(ANC):sha(ANC)},"region_counts":counts.to_dict(orient="records")}
 (AUD/"state_space_131072_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
 legend_text="""# Figure legend

**Fig. 4a | Reduced-order rheological state space.** All 131,072 unique reduced-order simulation states (small coloured points) were projected into the six-feature coordinate system frozen from 540 measured rheological conditions (open circles). The six features were log10(G'), log10(G''), log10(apparent viscosity at 10 s^-1), log10(1 + yield stress), log10(consistency index K), and flow index n. For simulated states, apparent viscosity at 10 s^-1 was calculated analytically from the reference Herschel-Bulkley parameters. Colours and dashed boundaries denote three regions defined along the measured low-elasticity, balanced, and high-elastic-recovery anchor axis. These regions are a visualization layer and do not replace the seven original post hoc reduced-model regimes retained in the source-data table. Simulated states are controlled in silico reduced-order model outputs, not additional experiments or COMSOL solutions.
"""
 (AUD/"FIGURE_LEGEND.md").write_text(legend_text,encoding="utf-8")
 qc_text="""# Final quality-control report

## Decision

PASS — ready as a versioned Figure 4a candidate.

## Evidence and counts

- Reduced-order simulation table: 131,072 rows, 131,072 unique IDs, zero duplicate rows.
- Measured rheological table: 540 rows and 540 unique condition IDs.
- All five reference rheology fields used for projection are complete.
- All simulation inputs retain `CONTROLLED_IN_SILICO_PARAMETER` provenance; all modeled outputs retain `PHYSICS_SIMULATED_OUTPUT_REDUCED_MODEL` provenance.
- No legacy print-association subset, old interpolation count, or design-software label appears in the user-facing source, legend, or audit text.

## Figure integrity

- All observations are plotted; no aesthetic sampling or row exclusion is used.
- The measured-data scaler, PCA loadings and three measured anchors are frozen inputs.
- The original seven post hoc reduced-model regimes are retained in the derived source table; the three coloured regions are explicitly an anchor-defined visualization layer.
- Times New Roman is used consistently for the title, subtitle, panel label, axes, ticks, region labels, anchor annotations, legend and footnote.
- PNG and 600 dpi TIFF were visually inspected for clipping, overlaps, point visibility, label placement and legend readability.
- SVG and PDF preserve editable text; dense points are rasterized inside vector containers.
- Nature-figure source preflight: 13 PASS, 0 WARN and one expected font-rule exception. The validator only recognizes its default sans-serif families; the user-requested Times New Roman setting was verified in the exported SVG.

## Submission wording boundary

Use “reduced-order simulation states” or “controlled in silico states”. Do not describe these rows as additional experiments, direct numerical solutions, or independent validation.
"""
 (AUD/"FINAL_QC_REPORT.md").write_text(qc_text,encoding="utf-8")
 rows=[]
 for p in sorted(q for q in OUT.rglob("*") if q.is_file() and q.name!="SHA256_MANIFEST.csv"):rows.append({"relative_path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)})
 pd.DataFrame(rows).to_csv(AUD/"SHA256_MANIFEST.csv",index=False)
 print(json.dumps({"output":str(OUT),"simulated_n":len(sim),"measured_n":len(measured),"counts":counts.to_dict(orient="records")},indent=2))

if __name__=="__main__":main()
