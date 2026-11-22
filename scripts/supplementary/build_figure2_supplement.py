from pathlib import Path
import json, shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image

REPO_ROOT=Path(__file__).resolve().parents[2]
ROOT=REPO_ROOT/"external_data"/"source_tree"
MASTER=ROOT/"fig02_supplementary_master"; OUT=MASTER/"figures"; SRC=MASTER/"source_data"
OLD=ROOT/"fig02_submission_ready"/"supplementary_figures"
PIPE=ROOT/"fig02_pipeline"/"outputs"/"supplementary"
ELIG=ROOT/"fig02_pipeline"/"fig2f_transferability"/"asset_eligibility_audit.csv"
SCORES=REPO_ROOT/"external_data"/"figure2"/"digital_phenotype_scores.csv"
MASTER590=REPO_ROOT/"external_data"/"figure2"/"printing_analysis_590.csv"
SELECTED=ROOT/"fig02_pipeline"/"fig2f_horizontal_v2"/"Fig2f_selected_samples.csv"
OUT.mkdir(exist_ok=True); SRC.mkdir(exist_ok=True)

BLACK="#1A1A1A"; DARK="#555555"; GRAY="#8C939B"; LIGHT="#D7DDE2"; PALE="#EEF2F4"; NAVY="#285E8E"; STEEL="#6E96B3"; ACCENT="#A85C45"
mpl.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":7.5,"axes.titlesize":8.5,"axes.titleweight":"bold","axes.labelsize":8,"axes.labelweight":"bold","xtick.labelsize":6.8,"ytick.labelsize":6.8,"legend.fontsize":6.6,"axes.linewidth":.65,"pdf.fonttype":42,"svg.fonttype":"none","figure.facecolor":"white","axes.facecolor":"white"})


def style(ax):
    ax.spines[["top","right"]].set_visible(False); ax.spines[["left","bottom"]].set_linewidth(.65); ax.tick_params(width=.6,length=3)


def panel(ax,label,title): ax.set_title(f"{label}  {title}",loc="left",pad=5,fontweight="bold")


def save(fig,stem):
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor="white")
    p=stem.with_suffix(".tiff"); fig.savefig(p,dpi=600,bbox_inches="tight",facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    with Image.open(p) as im: im.convert("RGB").save(p,dpi=(600,600),compression="tiff_lzw")
    plt.close(fig)


def copy_existing():
    mapping={
        "Supplementary_Figure_1_repeatability":"FigS_Fig2_01_scanning_repeatability",
        "Supplementary_Figure_4_density":"FigS_Fig2_03_point_density_sensitivity",
        "Supplementary_Figure_5_tolerance":"FigS_Fig2_04_tolerance_sensitivity",
        "Supplementary_Figure_2_denoising":"FigS_Fig2_08_denoising_sensitivity",
        "Supplementary_Figure_6_all_geometries":"FigS_Fig2_09_multigeometry_signed_deviation_maps",
    }
    for old,new in mapping.items():
        for ext in ("pdf","svg","png","tiff"): shutil.copy2(OLD/f"{old}.{ext}",OUT/f"{new}.{ext}")


def registration_robustness():
    cand=pd.read_csv(PIPE/"registration_candidate_robustness.csv"); methods=pd.read_csv(PIPE/"registration_method_comparison_turtle.csv")
    names={"Point-to-point ICP":"P2P ICP","Point-to-plane ICP":"P2L ICP","Multiscale point-to-plane ICP":"Multiscale P2L","FPFH+RANSAC + multiscale ICP":"FPFH–RANSAC + ICP","Multi-start OBB/PCA + multiscale ICP":"OBB/PCA + ICP"}
    fig,axs=plt.subplots(1,3,figsize=(7.12,2.85)); fig.subplots_adjust(left=.08,right=.985,bottom=.31,top=.78,wspace=.38)
    geoms=list(dict.fromkeys(cand.case_id)); x=np.arange(1,5)
    for g in geoms:
        q=cand[cand.case_id==g].sort_values("candidate_rank"); axs[0].plot(x,q.fitness,color=LIGHT,lw=.8); axs[0].scatter(x,q.fitness,s=12,facecolor="white",edgecolor=GRAY,lw=.6)
    axs[0].set_xlabel("Initialization candidate rank"); axs[0].set_ylabel("Registration fitness"); axs[0].set_xticks(x); style(axs[0]); panel(axs[0],"a","Initialization robustness")
    best=cand[cand.candidate_rank==1].sort_values("case_id"); xx=np.arange(len(best)); axs[1].scatter(xx,best.inlier_rmse_mm,s=24,color=NAVY)
    axs[1].set_xticks(xx,[x.title() for x in best.case_id],rotation=40,ha="right"); axs[1].set_ylabel("Best-candidate inlier RMSE (mm)"); style(axs[1]); panel(axs[1],"b","Across-geometry residuals")
    m=methods.copy(); xx=np.arange(len(m)); axs[2].scatter(xx,m.surface_mae_mm,s=30,facecolor="white",edgecolor=GRAY,lw=.9); axs[2].scatter(len(m)-1,m.surface_mae_mm.iloc[-1],s=38,color=NAVY,edgecolor="white",lw=.5)
    axs[2].set_xticks(xx,[names[x] for x in m.method],rotation=40,ha="right"); axs[2].set_ylim(0,.30); axs[2].set_ylabel("Surface MAE (mm)"); style(axs[2]); panel(axs[2],"c","Registration-method agreement")
    fig.suptitle("Registration robustness without iteration-level convergence claims",x=.08,y=.97,ha="left",fontsize=9.2,fontweight="bold")
    save(fig,OUT/"FigS_Fig2_02_registration_robustness_and_convergence")
    cand.to_csv(SRC/"FigS_Fig2_02_initialization_candidates.csv",index=False,encoding="utf-8-sig"); methods.to_csv(SRC/"FigS_Fig2_02_registration_methods.csv",index=False,encoding="utf-8-sig")


def redundancy():
    d=pd.read_csv(SCORES); cols=[f"D{i}_score" for i in range(1,6)]; x=d[cols].astype(float); corr=x.corr(method="spearman")
    z=(x-x.mean())/x.std(ddof=1); u,s,vt=np.linalg.svd(z,full_matrices=False); ev=s*s/(len(z)-1); ratio=ev/ev.sum(); load=vt.T*np.sqrt(ev)
    fig,axs=plt.subplots(1,3,figsize=(7.12,2.75)); fig.subplots_adjust(left=.07,right=.985,bottom=.20,top=.79,wspace=.42)
    im=axs[0].imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1); axs[0].set_xticks(range(5),[f"d{i}" for i in range(1,6)]); axs[0].set_yticks(range(5),[f"d{i}" for i in range(1,6)]); axs[0].tick_params(length=0)
    for i in range(5):
        for j in range(5): axs[0].text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=5.8,color="white" if abs(corr.iloc[i,j])>.55 else BLACK)
    panel(axs[0],"a","Spearman correlations"); cb=fig.colorbar(im,ax=axs[0],fraction=.047,pad=.04,ticks=[-1,0,1]); cb.set_label("Spearman ρ",fontweight="bold")
    axs[1].bar(np.arange(1,6),ratio*100,color=[NAVY]+[LIGHT]*4,edgecolor="none"); axs[1].plot(np.arange(1,6),np.cumsum(ratio)*100,color=DARK,lw=1,marker="o",ms=3); axs[1].set_xlabel("Principal component"); axs[1].set_ylabel("Explained variance (%)"); axs[1].set_xticks(range(1,6)); axs[1].set_ylim(0,105); style(axs[1]); panel(axs[1],"b","Variance structure")
    yy=np.arange(5); axs[2].axvline(0,color=LIGHT,lw=.8)
    axs[2].scatter(load[:,0],yy,s=30,color=NAVY,label="PC1"); axs[2].scatter(load[:,1],yy,s=30,facecolor="white",edgecolor=STEEL,label="PC2"); axs[2].set_yticks(yy,[f"d{i}" for i in range(1,6)]); axs[2].invert_yaxis(); axs[2].set_xlabel("Loading"); axs[2].legend(frameon=False); style(axs[2]); panel(axs[2],"c","PC1–PC2 loadings")
    fig.suptitle("Descriptive redundancy assessment of the five-domain phenotype (n = 590)",x=.07,y=.97,ha="left",fontsize=9.2,fontweight="bold")
    save(fig,OUT/"FigS_Fig2_05_descriptor_definition_and_redundancy")
    corr.rename_axis("dimension").reset_index().to_csv(SRC/"FigS_Fig2_05_spearman_correlations.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"PC":np.arange(1,6),"explained_variance_ratio":ratio,"cumulative_ratio":np.cumsum(ratio)}).to_csv(SRC/"FigS_Fig2_05_pca_variance.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(load,columns=[f"PC{i}_loading" for i in range(1,6)]).assign(dimension=[f"d{i}" for i in range(1,6)]).to_csv(SRC/"FigS_Fig2_05_pca_loadings.csv",index=False,encoding="utf-8-sig")


def crop_white(im,pad=4):
    a=np.asarray(im.convert("RGB")); m=np.any(a<245,axis=2)
    if not m.any(): return im
    y,x=np.where(m); return im.crop((max(0,x.min()-pad),max(0,y.min()-pad),min(a.shape[1],x.max()+pad+1),min(a.shape[0],y.max()+pad+1)))


def ranked_assets():
    d=pd.read_csv(ELIG); d=d.sort_values(["OQI","specimen_id"],kind="stable").reset_index(drop=True); d["rank_ascending"]=np.arange(1,len(d)+1); d["percentile_midpoint"]=(d.rank_ascending-.5)/len(d)
    return d


def additional_examples():
    d=ranked_assets(); current=set(pd.read_csv(SELECTED).sample_id); targets=[.15,.30,.40,.60,.75,.85]; rows=[]
    for t in targets:
        q=d[d.eligible & ~d.sample_id.isin(current)].copy(); q["distance"]=(q.percentile_midpoint-t).abs(); r=q.sort_values(["distance","OQI","specimen_id"],kind="stable").iloc[0].copy(); r["target_percentile"]=t; rows.append(r)
    sel=pd.DataFrame(rows); master=pd.read_csv(MASTER590); master["sample_id"]=master.canonical_sample_id.astype(str).str.lower(); sel=sel.merge(master[["sample_id","Starch_type","Concentration_pct","Preparation_T_C","Printing_T_C","Printing_speed_mm_s","Nozzle_diameter_mm","Model_height_mm"]],on="sample_id",how="left")
    fig=plt.figure(figsize=(7.12,5.35),facecolor="white"); fig.text(.035,.98,"Additional automatically selected starch-print examples",ha="left",va="top",fontsize=9.2,fontweight="bold")
    plotted=[]; last=None
    for i,r in sel.iterrows():
        row=i//3; col=i%3; x0=.035+col*.325; ytop=.85-row*.44; p=Path(r.photo_path); proc=Path(r.processed_dir); signed=pd.read_csv(next(proc.glob("*SignedDeviation.csv.gz")))
        q=signed if len(signed)<=8000 else signed.sample(8000,random_state=20260901+i).sort_index(); q=q.assign(sample_id=r.sample_id,target_percentile=r.target_percentile,cohort_percentile=r.percentile_midpoint,OQI=r.OQI); plotted.append(q)
        ax=fig.add_axes([x0,ytop-.22,.13,.22]); ax.imshow(crop_white(Image.open(p).convert("RGB"))); clean=ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(f"{r.sample_id.upper()} | cohort {r.percentile_midpoint*100:.1f}%\nOQI {r.OQI:.1f}",fontsize=6.0,fontweight="bold",pad=1.5)
        ov=fig.add_axes([x0+.145,ytop-.22,.075,.20]); ov.fill([-15,15,15,-15],[-15,-15,15,15],facecolor=PALE,edgecolor=GRAY,lw=.5,ls="--"); ov.scatter(q.X_mm,q.Y_mm,s=.10,color=STEEL,alpha=.5,linewidths=0,rasterized=True); ov.set_aspect("equal"); ov.set_xticks([]); ov.set_yticks([]); [s.set_visible(False) for s in ov.spines.values()]
        dm=fig.add_axes([x0+.230,ytop-.22,.075,.20]); last=dm.scatter(q.X_mm,q.Y_mm,c=np.clip(q.Signed_deviation_mm,-3,3),s=.10,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=3),linewidths=0,rasterized=True); dm.set_aspect("equal"); dm.set_xticks([]); dm.set_yticks([]); [s.set_visible(False) for s in dm.spines.values()]
        if i==0: ov.set_title("CAD–scan",fontsize=5.5,fontweight="bold",pad=1); dm.set_title("Signed δ",fontsize=5.5,fontweight="bold",pad=1)
    cax=fig.add_axes([.35,.035,.30,.018]); cb=fig.colorbar(last,cax=cax,orientation="horizontal",ticks=[-3,0,3],extend="both"); cb.set_label("Signed surface deviation, δ (mm)",fontweight="bold",fontsize=6.2,labelpad=1); cb.ax.tick_params(labelsize=5.8,length=2)
    save(fig,OUT/"FigS_Fig2_06_additional_starch_print_examples")
    keep=["specimen_id","sample_id","Starch_type","Concentration_pct","Preparation_T_C","Printing_T_C","Printing_speed_mm_s","Nozzle_diameter_mm","Model_height_mm","OQI","D1_score","D2_score","D3_score","D4_score","D5_score","percentile_midpoint","target_percentile","photo_path","processed_dir"]
    sel[keep].to_csv(SRC/"FigS_Fig2_06_selected_examples.csv",index=False,encoding="utf-8-sig"); pd.concat(plotted,ignore_index=True).to_csv(SRC/"FigS_Fig2_06_plot_points.csv",index=False,encoding="utf-8-sig")


def selection_workflow():
    d=ranked_assets(); sel=pd.read_csv(SELECTED); fig,axs=plt.subplots(1,2,figsize=(7.12,2.80)); fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.78,wspace=.30)
    bins=np.linspace(d.OQI.min(),d.OQI.max(),28); axs[0].hist(d.OQI,bins=bins,color=LIGHT,edgecolor="white",lw=.4)
    quant=d.OQI.quantile([.10,.45,.55,.90]); axs[0].axvspan(d.OQI.min(),quant.loc[.10],color=GRAY,alpha=.16); axs[0].axvspan(quant.loc[.45],quant.loc[.55],color=STEEL,alpha=.13); axs[0].axvspan(quant.loc[.90],d.OQI.max(),color=NAVY,alpha=.10)
    marks={"High fidelity":NAVY,"Intermediate fidelity":STEEL,"Low fidelity":DARK}
    for _,r in sel.iterrows(): axs[0].axvline(r.OQI,color=marks[r.level],lw=1.2,ls="--",label=f"{r.sample_id.upper()} ({r.percentile_midpoint*100:.1f}%)")
    axs[0].set_xlabel("Overall quality index (OQI)"); axs[0].set_ylabel("Specimen count"); axs[0].legend(frameon=False); style(axs[0]); panel(axs[0],"a","Full-cohort distribution")
    axs[1].scatter(d.percentile_midpoint*100,d.OQI,s=7,color=LIGHT,edgecolor="none",label="All specimens")
    e=d[d.eligible]; axs[1].scatter(e.percentile_midpoint*100,e.OQI,s=11,facecolor="white",edgecolor=GRAY,lw=.5,label=f"Complete assets (n = {len(e)})")
    for _,r in sel.iterrows(): axs[1].scatter(r.percentile_midpoint*100,r.OQI,s=38,color=marks[r.level],edgecolor="white",lw=.5,zorder=4); axs[1].annotate(r.sample_id.upper(),(r.percentile_midpoint*100,r.OQI),xytext=(3,4),textcoords="offset points",fontsize=6,fontweight="bold")
    axs[1].set_xlabel("Cohort percentile (%)"); axs[1].set_ylabel("OQI"); axs[1].set_xlim(0,100); axs[1].legend(frameon=False,loc="lower right"); style(axs[1]); panel(axs[1],"b","Eligibility and selected anchors")
    fig.suptitle("Objective Figure 2f sample-selection workflow (n = 590)",x=.075,y=.97,ha="left",fontsize=9.2,fontweight="bold")
    save(fig,OUT/"FigS_Fig2_07_Fig2f_sample_selection_workflow")
    d.to_csv(SRC/"FigS_Fig2_07_full_cohort_ranking.csv",index=False,encoding="utf-8-sig"); sel.to_csv(SRC/"FigS_Fig2_07_selected_samples.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"level":"Low fidelity","band":"0–10%","target_percentile":5},{"level":"Intermediate fidelity","band":"45–55%","target_percentile":50},{"level":"High fidelity","band":"90–100%","target_percentile":95}],).to_csv(SRC/"FigS_Fig2_07_selection_rules.csv",index=False,encoding="utf-8-sig")


def main():
    copy_existing(); registration_robustness(); redundancy(); additional_examples(); selection_workflow(); print(OUT)


if __name__=="__main__": main()
