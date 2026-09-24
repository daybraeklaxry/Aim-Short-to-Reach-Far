"""Publication artifacts from complete MAIN cells; pending groups never become plotted estimates."""
from pathlib import Path
import json,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from publication_tables import fmt,main_table,main_facts,full_appendix,mechanism_tables,diagnostic_table,first_block_table,first_block_full_table,learned_target_tables,ablation_table,transport_table
from publication_plots import compute_figure,horizon_figure,teaser_figure
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
for f in (ROOT/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'savefig.pad_inches':.04})
T=ROOT/'tables_v2';F=ROOT/'figures_v2';T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
TASKS=['cube','pusht','reacher','tworoom'];NAMES=dict(cube='Cube',pusht='PushT',reacher='Reacher',tworoom='TwoRoom')
DISPLAY={'cem_final':'CEM (final goal)','cem_learned':'CEM (learned target)','cem_observed':'AP-CEM','rank_final':'Rank (final goal)','rank_learned':'Rank (learned target)','rank_observed':'AP-rank','lewm':'LeWM planner','cem_transport':'CEM (transported)'}
COLORS={'final':'#8C8C8C','learned':'#D98C2B','observed':'#1B8A8A','lewm':'#111111'}
MAIN_ARMS=['LeWM planner','CEM (final goal)','CEM (learned target)','AP-CEM','Direct','Rank (final goal)','Rank (learned target)','AP-rank']
STATUS={};DATA={}
def load(name,default=None):
 p=ROOT/name;return json.loads(p.read_text()) if p.exists() else default
main=load('collected/main.json',{'rows':[]})['rows'];inter=load('collected/interventions.json',{'rows':[]})['rows']
def mm(task,arm,start):
 found=[r for r in main if r['task']==task and r['arm']==arm and r['start']==start]
 return found[0] if len(found)==1 and found[0]['complete'] else None
def ii(kind,task,arm,start,**cfg):
 found=[r for r in inter if r['kind']==kind and r['task']==task and r['config']['arm']==arm and r['start']==start and all(r['config'].get(k)==v for k,v in cfg.items())]
 return found[0] if len(found)==1 and found[0]['complete'] else None
def table(name,head,rows,align=None):
 n=len(head);text='\\begin{tabular}{'+(align or 'l'+'r'*(n-1))+'}\n\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
 text+='\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (T/name).write_text(text);STATUS['tables_v2/'+name]='generated from complete cells'
def wait(name,reason):STATUS[name]='pending: '+reason
def savefig(fig,name,data):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=160);plt.close(fig)
 DATA[name]=data;STATUS['figures_v2/'+name+'.pdf']='generated from complete cells'
def finish(ax):
 ax.set_ylim(-2,102);ax.set_yticks([0,50,100]);ax.grid(axis='y',color='#E9E9E9',linewidth=.5);ax.set_axisbelow(True)
# Main table: groups are action resources; no-memory singleton is not bolded.
needed=[mm(t,a,s) for t in TASKS for a in MAIN_ARMS for s in ['Standard','Perturbed']]
if all(needed):
 (T/'main.tex').write_text(main_table(main));STATUS['tables_v2/main.tex']='generated from complete cells'
 (ROOT/'manuscript_fragments').mkdir(exist_ok=True)
 (ROOT/'manuscript_fragments/main_facts.tex').write_text(main_facts(main))
else:wait('tables_v2/main.tex','all eight controllers, four tasks and three starts required')
# Keep all prescribed target-model candidates and held-out target-quality measures.
selections={t:load(f'training/{t}/selected.json') for t in TASKS}
qualities={t:load(f'target_quality/{t}/summary.json') for t in TASKS}
if all(selections.values()) and all(q and q.get('complete') for q in qualities.values()):
 for name,value in learned_target_tables(selections,qualities).items():
  (T/name).write_text(value);STATUS['tables_v2/'+name]='generated from complete validation results'
else:
 for name in ['target_models.tex','target_quality.tex']:wait('tables_v2/'+name,'all prescribed trained candidates and target-quality samples required')
# Compute sweep, all tasks and aggregated starts. Original 30-point is an explicit alias.
compute=[(t,s,a,I,ii('budget',t,a,s,iterations=I)) for t in TASKS for s in ['Standard','Perturbed'] for a in ['cem_final','cem_learned','cem_observed'] for I in [1,2,5,10,30]]
if all(r is not None for *_,r in compute) and all(mm(t,'LeWM planner',s) for t in TASKS for s in ['Standard','Perturbed']):
 data=[dict(task=t,start=s,arm=a,iterations=I,success=r['success_percent'],work=r['mean_pred_blocks']) for t,s,a,I,r in compute]
 refs=[dict(task=t,start=s,success=mm(t,'LeWM planner',s)['success_percent'],work=mm(t,'LeWM planner',s)['mean_pred_blocks']) for t in TASKS for s in ['Standard','Perturbed']]
 for start,name in [('Standard','compute'),('Perturbed','compute_perturbed')]:
  savefig(compute_figure(data,refs,[start]),name,[r for r in data if r['start']==start])
else:
 for name in ['compute','compute_perturbed']:
  wait('figures_v2/'+name+'.pdf','complete three-target budget sweep and LeWM references required')
# Horizon sweep: all prescribed controller points, with main-H aliases.
horizon_arms=['cem_final','cem_learned','cem_observed','lewm','rank_final','rank_learned','rank_observed']
HS={'cube':[25,50,100,150],'pusht':[25,50,100,140],'reacher':[25,50,100,150],'tworoom':[25,50,100]}
horizon=[(t,s,a,H,ii('horizon',t,a,s,H=H)) for t in TASKS for s in ['Standard','Perturbed'] for a in horizon_arms for H in HS[t]]
if all(r is not None for *_,r in horizon):
 data=[dict(task=t,start=s,arm=a,H=H,success=r['success_percent']) for t,s,a,H,r in horizon]
 for start,name in [('Standard','horizon'),('Perturbed','horizon_perturbed')]:
  savefig(horizon_figure(data,[start]),name,[r for r in data if r['start']==start])
else:
 for name in ['horizon','horizon_perturbed']:
  wait('figures_v2/'+name+'.pdf','complete all seven controllers at all goal offsets')
teaser=[(a,H,ii('horizon','pusht',a,'Standard',H=H)) for a in ['cem_final','cem_learned','cem_observed','lewm'] for H in HS['pusht']]
if all(r is not None for *_,r in teaser):
 data=[dict(arm=a,H=H,success=r['success_percent']) for a,H,r in teaser]
 savefig(teaser_figure(data),'teaser',data)
else:wait('figures_v2/teaser.pdf','all four PushT horizon curves required')
# Absolute endpoint vs transported displacement: same MAIN query set.
transport=[(t,s,a,ii('transport',t,a,s)) for t in TASKS for s in ['Standard','Perturbed'] for a in ['cem_observed','cem_transport']]
if all(r is not None for *_,r in transport):
 (T/'transport.tex').write_text(transport_table(inter));STATUS['tables_v2/transport.tex']='generated from complete cells'
else:wait('tables_v2/transport.tex','complete MAIN transport pair required')
# Diagnostic and first-choice tables keep actual eligible sample counts.
diagnostic=load('collected/diagnostic.json',{'complete':False})
if diagnostic['complete']:
 for name,starts in [('diagnostic.tex',('Standard','Perturbed')),('diagnostic_full.tex',('Standard','Perturbed 1','Perturbed 2','Perturbed'))]:
  (T/name).write_text(diagnostic_table(diagnostic['rows'],starts));STATUS['tables_v2/'+name]='generated from complete cells'
else:wait('tables_v2/diagnostic.tex','complete assigned simulator branches and eligible counts required')
first=load('collected/first_block.json',{'complete':False})
selectors=['Direct','Predictor-Final','Predictor-Observed','Predictor-Learned','Simulator-Final','Simulator-Observed','Simulator-Learned','Displacement-Observed']
if first['complete']:
 for name,fn in [('first_block.tex',first_block_table),('first_block_full.tex',first_block_full_table)]:
  (T/name).write_text(fn(first['rows']));STATUS['tables_v2/'+name]='generated from complete cells'
else:wait('tables_v2/first_block.tex','all eight selector continuations required')
# Ablations: rank-only choices use an em dash in the CEM column (not an unrun experiment).
variants=[('Default',{'default':True}),(r'Target $\ell=1$',{'target_L':1}),(r'Target $\ell=3$',{'target_L':3}),(r'Target $\ell=10$',{'target_L':10}),('Memory 10%',{'memory_fraction':.1}),('Memory 25%',{'memory_fraction':.25}),('Memory 50%',{'memory_fraction':.5}),('Key without far endpoint',{'key':'no_far'}),('Key without displacement',{'key':'no_delta'}),('Fixed retrieval span',{'span':'fixed'})]
rows=[];complete=True
for label,cfg in variants:
 values=[]
 for a in ['rank_observed','cem_observed']:
  for st in ['Standard','Perturbed']:
   if a=='cem_observed' and ('key' in cfg or 'span' in cfg):values.append(None);continue
   selected=[ii('ablation',t,a,st,**cfg) for t in TASKS]
   if all(selected):values.append(float(np.mean([x['success_percent'] for x in selected])))
   else:complete=False;values.append(None)
 rows.append([label,*values])
if complete:
 (T/'ablation.tex').write_text(ablation_table(rows));STATUS['tables_v2/ablation.tex']='generated from complete cells'
else:wait('tables_v2/ablation.tex','all prescribed task/variant cells required')
# Mechanism figure has a complete original-MAIN foundation; new controller table waits for them.
mechanism=load('mechanism/summary.json',{'complete':False});curves=load('mechanism/curves.json',{})
if mechanism.get('complete'):
 mr={(r['task'],r['arm'],r['start']):r for r in mechanism['rows']}
 fig=plt.figure(figsize=(5.5,2.5),layout='constrained');axs=fig.subplots(1,3,gridspec_kw={'width_ratios':[1.3,1,1]})
 drawn=[]
 for arm,success,name,color in [('gaussian_final',False,'CEM (final goal)',COLORS['final']),('gaussian_observed',True,'AP-CEM',COLORS['observed'])]:
  items=sorted((v for v in curves.values() if v['arm']==arm and v['success']==success and v['t']),key=lambda x:x['query_id'])
  for v in items[:3]:axs[0].plot(v['t']+[v['terminal_t']],v['error']+[v['terminal_error']],color=color,alpha=.30,linewidth=.6)
  grid=np.arange(0,max(v['terminal_t'] for v in items)+1,5)
  # Carry terminal error forward, so early successes remain represented in the cohort median.
  values=np.asarray([np.interp(grid,v['t']+[v['terminal_t']],v['error']+[v['terminal_error']]) for v in items])
  axs[0].plot(grid,np.median(values,axis=0),color=color,label=name,linewidth=1.4)
  drawn.append(dict(arm=arm,cohort_count=len(items),example_query_ids=[v['query_id'] for v in items[:3]],median_rule='All eligible traces in the outcome stratum; terminal error carried forward. Examples are first three sorted query IDs.'))
 axs[0].axhline(1,color='#C8C8C8',linewidth=.6,linestyle=':');axs[0].set_yscale('symlog',linthresh=1,linscale=.5);axs[0].set_ylim(0,None);axs[0].set_yticks([0,1,10,100]);axs[0].set_yticklabels(['0','1','10','100']);axs[0].set_xlabel('Primitive actions');axs[0].set_ylabel('Normalized error (symlog)');axs[0].set_title('(a) PushT / standard start');axs[0].legend(frameon=False,fontsize=8)
 for panel,metric,title in [(1,'stall_W10','(b) Stalling in failures'),(2,'physical_detour_0.5','(c) Detours in successes')]:
  x=np.arange(4)
  for offset,arm,color in [(-.18,'gaussian_final',COLORS['final']),(.18,'gaussian_observed',COLORS['observed'])]:
   values=[mr[(t,arm,'Standard')][metric]['percent'] for t in TASKS];axs[panel].bar(x+offset,values,width=.34,color=color)
  axs[panel].set_xticks(x,[NAMES[t] for t in TASKS],rotation=35,ha='right');axs[panel].set_ylim(0,105);axs[panel].set_yticks([0,50,100]);axs[panel].set_ylabel('Episodes (%)');axs[panel].set_title(title)
 savefig(fig,'mechanism',dict(scope='Complete original MAIN standard-start final vs observed CEM',physical_error_axis='symlog with linear threshold 1; full data retained without clipping',illustrations=drawn,conditional_denominators='Stalling among failed episodes with decisions; detours among policy-entered successful episodes. This comparison does not establish a unique cause of failure.'))
else:wait('figures_v2/mechanism.pdf','complete MAIN behavioral extraction required')
# Complete appendix rows are emitted only when every assigned intervention is available.
if inter and all(r['complete'] for r in inter) and all(needed):
 (T/'appendix_full.tex').write_text(full_appendix(inter,main));STATUS['tables_v2/appendix_full.tex']='generated from complete cells; paginated by study'
else:wait('tables_v2/appendix_full.tex','every registered intervention and main cell must be complete')
# Long-form mechanism table will add newly completed controllers in the next collection pass.
new_mechanism=load('mechanism_all/summary.json',{'rows':[]})
mechanism_outputs=mechanism_tables(mechanism,new_mechanism)
if mechanism_outputs is not None:
 for key,value in mechanism_outputs.items():
  name='mechanism.tex' if key=='mechanism' else 'mechanism_sensitivity.tex'
  (T/name).write_text(value);STATUS['tables_v2/'+name]='generated from complete cells with eligible denominators'
else:
 for name in ['mechanism.tex','mechanism_sensitivity.tex']:wait('tables_v2/'+name,'all main controllers and Transported behavioral cells required')
(ROOT/'artifact-status.json').write_text(json.dumps(dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outputs=STATUS,all_requested_outputs_ready=all(not v.startswith('pending') for v in STATUS.values()),no_partial_estimates=True,plot_data=DATA),indent=2))
print(json.dumps(STATUS))
