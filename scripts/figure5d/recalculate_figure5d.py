from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics as st

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import binomtest, wilcoxon


work = Path(__file__).parent
source = work / 'Figure5b_grouped_oof_predictions_469.csv'
with source.open(encoding='utf-8-sig', newline='') as f:
    attempts = list(csv.DictReader(f))
assert len(attempts) == 469
target = 'OQI_current_domain_strict'
predictions = {'process': 'pred_process_only_grouped_oof', 'combined': 'pred_rheology_process_grouped_oof'}
grouped = defaultdict(lambda: defaultdict(list))
for r in attempts:
    grouped[r['Rheology_ID']][r['Candidate_group_ID']].append(r)

detail = []
for state, groups in sorted(grouped.items()):
    if len(groups) < 2:
        continue
    records = [dict(Candidate_group_ID=g, attempts=len(rs), **{
        c: st.mean(float(r[c]) for r in rs) for c in (target, *predictions.values())
    }) for g, rs in sorted(groups.items())]
    optimum = max(r[target] for r in records)
    for method, predictor in predictions.items():
        order = sorted(records, key=lambda r: (-r[predictor], r['Candidate_group_ID']))
        best_rank = next(i for i, r in enumerate(order, 1) if r[target] == optimum)
        detail.append(dict(Rheology_ID=state, method=method, candidate_n=len(records),
                           attempt_n=sum(r['attempts'] for r in records),
                           best_observed_rank_in_predicted_order=best_rank,
                           top1=int(best_rank <= 1), top2=int(best_rank <= 2),
                           regret=optimum-order[0][target],
                           selected_candidate=order[0]['Candidate_group_ID'],
                           best_observed_OQI=optimum, selected_observed_OQI=order[0][target],
                           selected_predicted_OQI=order[0][predictor]))

rng = np.random.Generator(np.random.PCG64(20260902))
summaries = []
scope_tests = {}
for min_n in (2, 3):
    pairs = [[r for r in detail if r['method'] == m and r['candidate_n'] >= min_n] for m in predictions]
    assert [r['Rheology_ID'] for r in pairs[0]] == [r['Rheology_ID'] for r in pairs[1]]
    n = len(pairs[0]); assert n == (132 if min_n == 2 else 42)
    resample = rng.integers(0, n, size=(5000, n))
    for method, rows in zip(predictions, pairs):
        for k in (1, 2):
            flags = np.array([r[f'top{k}'] for r in rows])
            low, high = np.quantile(flags[resample].mean(axis=1), [.025, .975])
            summaries.append(dict(min_distinct_candidates=min_n, method=method, k=k, n=n, hits=int(flags.sum()),
                                  rate=float(flags.mean()), ci_low=float(low), ci_high=float(high),
                                  mean_top1_regret=st.mean(r['regret'] for r in rows)))
    ranks = [[r['best_observed_rank_in_predicted_order'] for r in rr] for rr in pairs]
    p = {'wilcoxon_rank_p':float(wilcoxon(*ranks, alternative='two-sided').pvalue)}
    for k in (1, 2):
        discordance = [a[f'top{k}']-b[f'top{k}'] for a,b in zip(*pairs)]
        pos, neg = discordance.count(1), discordance.count(-1)
        p[f'top{k}_exact_p'] = float(binomtest(pos, pos+neg).pvalue) if pos+neg else 1.
    scope_tests[min_n] = p

for name, rows in [('Figure5d_distinct_candidate_ranks.csv', detail), ('Figure5d_distinct_candidate_summary.csv', summaries)]:
    with (work/name).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(work/'Figure5d_distinct_candidate_tests.json').write_text(json.dumps(scope_tests,indent=2),encoding='utf-8')
for row in summaries:
    print(row['min_distinct_candidates'],row['method'],row['k'],row['hits'],row['n'],round(row['rate']*100,2), round(row['mean_top1_regret'],2))
print(scope_tests)

# Same-scale retrieval comparisons for all informative states and the nontrivial Top-2 subset.
mpl.rcParams.update({'font.family':['Times New Roman','Arial'],'font.size':8,'axes.labelsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                     'svg.fonttype':'none','axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8})
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7.1, 2.45))
colors={'process':'#58788f','combined':'#196582'}
markers={'process':'o','combined':'D'}
for ax, min_n, title in zip(axs,(2,3),('At least two candidates (n = 132)','At least three candidates (n = 42)')):
    ax.set_title(title,pad=8,fontweight='bold'); ax.set_ylim(24,108); ax.set_xlim(.7,2.3)
    ax.set_xticks([1,2],['Top 1','Top 2']); ax.grid(axis='y',color='#dce3e8',linewidth=.6)
    for method, offset in [('process',-.11),('combined',.11)]:
        rows=[r for r in summaries if r['min_distinct_candidates']==min_n and r['method']==method]
        x=np.array([r['k'] for r in rows])+offset; y=np.array([r['rate']*100 for r in rows]);
        err=np.array([[v-r['ci_low']*100 for r,v in zip(rows,y)],[r['ci_high']*100-v for r,v in zip(rows,y)]])
        ax.errorbar(x,y,yerr=err,marker=markers[method],ms=4,ls='-',lw=.8,elinewidth=.75,capsize=2,
                    color=colors[method],label={'process':'Process only','combined':'Rheology + process'}[method])
        for a, v in zip(x,y):
            ax.annotate(f'{v:.1f}',(a,v),xytext=(-5 if method=='process' else 5, 4 if method=='process' else -7),
                        textcoords='offset points',ha='right' if method=='process' else 'left',fontsize=7,color=colors[method])
axs[0].set_ylabel('Observed-best retrieval (%)',fontweight='bold')
fig.text(.5,.98,'Retrieval of the observed-best process among predicted candidates',ha='center',va='top',fontweight='bold')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,ncol=2)
fig.subplots_adjust(left=.1,right=.98,bottom=.22,top=.76,wspace=.16)
fig.savefig(work/'Figure5d_candidate_retrieval.png',dpi=300)
fig.text(.01,.98,'d',ha='left',va='top',fontweight='bold',fontsize=11)
fig.savefig(work/'Figure5d_distinct_candidates_TNR.pdf',dpi=600)
fig.savefig(work/'Figure5d_distinct_candidates_TNR.svg',dpi=600)
png = work/'Figure5d_distinct_candidates_TNR.png'
fig.savefig(png,dpi=600)
with Image.open(png) as image:
    image.convert('RGB').save(work/'Figure5d_distinct_candidates_TNR.tiff',dpi=(600,600),compression='raw')
