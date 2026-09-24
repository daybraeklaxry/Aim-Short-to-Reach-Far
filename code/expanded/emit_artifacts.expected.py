"""Publication artifacts from complete MAIN cells; pending groups never become plotted estimates."""
from pathlib import Path
import json,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
for f in (ROOT/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'savefig.pad_inches':.04})
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
def fmt(x):return f'{x:.1f}'
def savefig(fig,name,data):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=160);plt.close(fig)
 DATA[name]=data;STATUS['figures_v2/'+name+'.pdf']='generated from complete cells'
def finish(ax):
 ax.set_ylim(-2,102);ax.set_yticks([0,50,100]);ax.grid(axis='y',color='#E9E9E9',linewidth=.5);ax.set_axisbelow(True)
# Main table: groups are action resources; no-memory singleton is not bolded.
needed=[mm(t,a,s) for t in TASKS for a in MAIN_ARMS for s in ['Standard','Perturbed']]
if all(needed):
 rows=[]
 for t in TASKS+['Mean']:
  for st in ['Standard','Perturbed']:
   vals=[np.mean([mm(k,a,st)['success_percent'] for k in TASKS]) if t=='Mean' else mm(t,a,st)['success_percent'] for a in MAIN_ARMS]
   cells=[fmt(v) for v in vals]
   for group in [[1,2,3],[4,5,6,7]]:
    best=max(vals[i] for i in group)
    for i in group:
     if vals[i]==best:cells[i]=r'\textbf{'+cells[i]+'}'
   rows.append([NAMES.get(t,t),st,*cells])
 header=['Task','Start',r'\shortstack{LeWM\\planner}',r'\shortstack{CEM\\final}',r'\shortstack{CEM\\learned}','AP-CEM','Direct',r'\shortstack{Rank\\final}',r'\shortstack{Rank\\learned}','AP-rank']
 table('main.tex',header,rows,'ll'+'r'*8)
 path=T/'main.tex';text=path.read_text();needle='\\toprule\n';groups=r' & & No memory & \multicolumn{3}{c}{Observation-only} & \multicolumn{4}{c}{Recorded actions} \\'+'\n'+r'\cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(lr){7-10}'+'\n'
 path.write_text(text.replace(needle,needle+groups,1))
else:wait('tables_v2/main.tex','all eight controllers, four tasks and three starts required')
# Compute sweep, all tasks and aggregated starts. Original 30-point is an explicit alias.
compute=[(t,s,a,I,ii('budget',t,a,s,iterations=I)) for t in TASKS for s in ['Standard','Perturbed'] for a in ['cem_final','cem_learned','cem_observed'] for I in [1,2,5,10,30]]
if all(r is not None for *_,r in compute) and all(mm(t,'LeWM planner',s) for t in TASKS for s in ['Standard','Perturbed']):
 fig,axs=plt.subplots(2,4,figsize=(7.2,4.5),sharey=True,layout='constrained')
 for i,s in enumerate(['Standard','Perturbed']):
  for j,t in enumerate(TASKS):
   ax=axs[i,j]
   for a in ['cem_final','cem_learned','cem_observed']:
    rs=[ii('budget',t,a,s,iterations=I) for I in [1,2,5,10,30]];ax.plot([r['mean_pred_blocks'] for r in rs],[r['success_percent'] for r in rs],marker='o',markersize=3,color=COLORS[a.split('_')[1]],label=DISPLAY[a])
   ref=mm(t,'LeWM planner',s);ax.plot(ref['mean_pred_blocks'],ref['success_percent'],marker='*',markersize=7,color=COLORS['lewm'],linestyle='none',label='LeWM planner')
   ax.set_xscale('log');finish(ax);ax.set_title(NAMES[t]+' / '+s)
   if j==0:ax.set_ylabel('Success (%)')
   if i==1:ax.set_xlabel('Prediction blocks / episode')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
 savefig(fig,'compute',[dict(task=t,start=s,arm=a,iterations=I,success=r['success_percent'],work=r['mean_pred_blocks']) for t,s,a,I,r in compute])
else:wait('figures_v2/compute.pdf','complete three-target budget sweep and LeWM references required')
# Horizon sweep: all prescribed controller points, with main-H aliases.
horizon_arms=['cem_final','cem_learned','cem_observed','lewm','rank_final','rank_learned','rank_observed']
HS={'cube':[25,50,100,150],'pusht':[25,50,100,140],'reacher':[25,50,100,150],'tworoom':[25,50,100]}
horizon=[(t,s,a,H,ii('horizon',t,a,s,H=H)) for t in TASKS for s in ['Standard','Perturbed'] for a in horizon_arms for H in HS[t]]
if all(r is not None for *_,r in horizon):
 fig,axs=plt.subplots(2,4,figsize=(7.2,4.5),sharey=True,layout='constrained')
 for i,s in enumerate(['Standard','Perturbed']):
  for j,t in enumerate(TASKS):
   ax=axs[i,j]
   for a in horizon_arms:
    rs=[ii('horizon',t,a,s,H=H) for H in HS[t]];color=COLORS['lewm' if a=='lewm' else a.split('_')[1]]
    ax.plot(HS[t],[r['success_percent'] for r in rs],color=color,linestyle='--' if a.startswith('rank') else '-',marker='*' if a=='lewm' else 's' if a.startswith('rank') else 'o',markersize=5 if a=='lewm' else 3,label=DISPLAY[a])
   finish(ax);ax.set_xticks(HS[t]);ax.set_title(NAMES[t]+' / '+s)
   if j==0:ax.set_ylabel('Success (%)')
   if i==1:ax.set_xlabel('Goal offset H')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
 savefig(fig,'horizon',[dict(task=t,start=s,arm=a,H=H,success=r['success_percent']) for t,s,a,H,r in horizon])
else:wait('figures_v2/horizon.pdf','complete all seven controllers at all goal offsets')
teaser=[(a,H,ii('horizon','pusht',a,'Standard',H=H)) for a in ['cem_final','cem_learned','cem_observed','lewm'] for H in HS['pusht']]
if all(r is not None for *_,r in teaser):
 fig,ax=plt.subplots(figsize=(3.4,2.3),layout='constrained')
 for a in ['cem_final','cem_learned','cem_observed','lewm']:
  rs=[ii('horizon','pusht',a,'Standard',H=H) for H in HS['pusht']];ax.plot(HS['pusht'],[r['success_percent'] for r in rs],marker='*' if a=='lewm' else 'o',markersize=6 if a=='lewm' else 3,color=COLORS['lewm' if a=='lewm' else a.split('_')[1]],label=DISPLAY[a])
 finish(ax);ax.set_xlabel('Goal offset H');ax.set_ylabel('Success (%)');ax.set_xticks(HS['pusht']);ax.legend(frameon=False,fontsize=7,loc='best')
 savefig(fig,'teaser',[dict(arm=a,H=H,success=r['success_percent']) for a,H,r in teaser])
else:wait('figures_v2/teaser.pdf','all four PushT horizon curves required')
# Absolute endpoint vs transported displacement: same MAIN query set.
transport=[(t,s,a,ii('transport',t,a,s)) for t in TASKS for s in ['Standard','Perturbed'] for a in ['cem_observed','cem_transport']]
if all(r is not None for *_,r in transport):
 table('transport.tex',['Task','Start','Observed','Transported'],[[NAMES[t],s,fmt(ii('transport',t,'cem_observed',s)['success_percent']),fmt(ii('transport',t,'cem_transport',s)['success_percent'])] for t in TASKS for s in ['Standard','Perturbed']])
else:wait('tables_v2/transport.tex','complete MAIN transport pair required')
# Diagnostic and first-choice tables keep actual eligible sample counts.
diagnostic=load('collected/diagnostic.json',{'complete':False})
if diagnostic['complete']:
 table('diagnostic.tex',['Task','Start','n',r'$n_q$','Pred. error','Disp. error','Pred. regret','Disp. regret','Direct regret'],[[NAMES[r['task']],r['start'],r['eligible_starts'],r['eligible_queries'],*[f'{r[k]:.3f}' for k in ['predictor_error','displacement_error','predictor_regret','displacement_regret','direct_regret']]] for r in diagnostic['rows'] if r['start'] in ['Standard','Perturbed']])
else:wait('tables_v2/diagnostic.tex','complete assigned simulator branches and eligible counts required')
first=load('collected/first_block.json',{'complete':False})
selectors=['Direct','Predictor-Final','Predictor-Observed','Predictor-Learned','Simulator-Final','Simulator-Observed','Simulator-Learned','Displacement-Observed']
if first['complete']:
 index={(r['task'],r['start'],r['selector']):r for r in first['rows']}
 table('first_block.tex',['Task','Start',*selectors],[[NAMES[t],s,*[fmt(index[(t,s,a)]['success_percent']) for a in selectors]] for t in TASKS for s in ['Standard','Perturbed']],'ll'+'r'*8)
else:wait('tables_v2/first_block.tex','all eight selector continuations required')
# Ablations: rank-only choices use an em dash in the CEM column (not an unrun experiment).
variants=[('Default',{'default':True}),('Target L=1',{'target_L':1}),('Target L=3',{'target_L':3}),('Target L=10',{'target_L':10}),('Memory 10%',{'memory_fraction':.1}),('Memory 25%',{'memory_fraction':.25}),('Memory 50%',{'memory_fraction':.5}),('Key without far endpoint',{'key':'no_far'}),('Key without displacement',{'key':'no_delta'}),('Fixed retrieval span',{'span':'fixed'})]
rows=[];complete=True
for label,cfg in variants:
 values=[]
 for a in ['rank_observed','cem_observed']:
  for st in ['Standard','Perturbed']:
   if a=='cem_observed' and ('key' in cfg or 'span' in cfg):values.append('--');continue
   selected=[ii('ablation',t,a,st,**cfg) for t in TASKS]
   if all(selected):values.append(fmt(np.mean([x['success_percent'] for x in selected])))
   else:complete=False;values.append('pending')
 rows.append([label.replace('%',r'\%'),*values])
if complete:table('ablation.tex',['Variant','Rank Std.','Rank Pert.','CEM Std.','CEM Pert.'],rows)
else:wait('tables_v2/ablation.tex','all prescribed task/variant cells required')
# Mechanism figure has a complete original-MAIN foundation; new controller table waits for them.
mechanism=load('mechanism/summary.json',{'complete':False});curves=load('mechanism/curves.json',{})
if mechanism.get('complete'):
 mr={(r['task'],r['arm'],r['start']):r for r in mechanism['rows']}
 fig=plt.figure(figsize=(6.6,2.7),layout='constrained');axs=fig.subplots(1,3,gridspec_kw={'width_ratios':[1.3,1,1]})
 drawn=[]
 for arm,success,name,color in [('gaussian_final',False,'Final goal: failures',COLORS['final']),('gaussian_observed',True,'AP-CEM: successes',COLORS['observed'])]:
  items=sorted((v for v in curves.values() if v['arm']==arm and v['success']==success and v['t']),key=lambda x:x['query_id'])
  for v in items[:3]:axs[0].plot(v['t']+[v['terminal_t']],v['error']+[v['terminal_error']],color=color,alpha=.30,linewidth=.6)
  grid=np.arange(0,max(v['terminal_t'] for v in items)+1,5)
  # Carry terminal error forward, so early successes remain represented in the cohort median.
  values=np.asarray([np.interp(grid,v['t']+[v['terminal_t']],v['error']+[v['terminal_error']]) for v in items])
  axs[0].plot(grid,np.median(values,axis=0),color=color,label=name,linewidth=1.4)
  drawn.append(dict(arm=arm,cohort_count=len(items),example_query_ids=[v['query_id'] for v in items[:3]],median_rule='All eligible traces in the outcome stratum; terminal error carried forward. Examples are first three sorted query IDs.'))
 axs[0].axhline(1,color='#C8C8C8',linewidth=.6,linestyle=':');axs[0].set_xlabel('Primitive actions');axs[0].set_ylabel('Normalized physical error');axs[0].set_title('(a) PushT / standard start');axs[0].legend(frameon=False,fontsize=7)
 for panel,metric,title in [(1,'stall_W10','(b) Stalling among failures'),(2,'physical_detour_0.5','(c) Detours among successes')]:
  x=np.arange(4)
  for offset,arm,color in [(-.18,'gaussian_final',COLORS['final']),(.18,'gaussian_observed',COLORS['observed'])]:
   values=[mr[(t,arm,'Standard')][metric]['percent'] for t in TASKS];axs[panel].bar(x+offset,values,width=.34,color=color)
  axs[panel].set_xticks(x,[NAMES[t] for t in TASKS],rotation=35,ha='right');axs[panel].set_ylim(0,105);axs[panel].set_yticks([0,50,100]);axs[panel].set_ylabel('Episodes (%)');axs[panel].set_title(title)
 savefig(fig,'mechanism',dict(scope='Complete original MAIN standard-start final vs observed CEM',illustrations=drawn,conditional_denominators='Stalling among failed episodes with decisions; detours among policy-entered successful episodes. This comparison does not establish a unique cause of failure.'))
else:wait('figures_v2/mechanism.pdf','complete MAIN behavioral extraction required')
# Complete appendix rows are emitted only when every assigned intervention is available.
if inter and all(r['complete'] for r in inter) and all(needed):
 full=[]
 for r in inter:
  c=r['config'];desc=', '.join(f'{k}={v}' for k,v in c.items() if k!='arm').replace('_',r'\_').replace('%',r'\%')
  full.append([NAMES[r['task']],r['kind'].replace('_',r'\_'),DISPLAY.get(c['arm'],c['arm']),r['start'],desc,fmt(r['success_percent']),f"{r['mean_pred_blocks']:.1f}"])
 table('appendix_full.tex',['Task','Study','Controller','Start','Configuration','Success','Blocks'],full,'lllllrr')
else:wait('tables_v2/appendix_full.tex','every registered intervention and main cell must be complete')
# Long-form mechanism table will add newly completed controllers in the next collection pass.
new_mechanism=load('mechanism_all/summary.json',{'rows':[]})
required=[]
for t in TASKS:
 for arm,kind in [('cem_learned','main'),('rank_learned','main'),('lewm_25','main'),('cem_transport','transport')]:
  for st in ['Standard','Perturbed']:
   found=[r for r in new_mechanism['rows'] if r['group']['kind']==kind and r['group']['task']==t and r['group']['config']['arm']==arm and r['start']==st]
   required.append(found[0] if len(found)==1 and found[0]['complete'] else None)
if all(required):
 mrows=[]
 for r in mechanism['rows']:
  if r['start'] in ['Standard','Perturbed']:
   mrows.append([NAMES[r['task']],r['arm'].replace('_',r'\_'),r['start'],*[fmt(r[k]['percent']) if r[k]['percent'] is not None else '--' for k in ['stall_W10','physical_detour_0.5','latent_detour']]])
 for r in required:
  mrows.append([NAMES[r['group']['task']],r['group']['config']['arm'].replace('_',r'\_'),r['start'],*[fmt(r[k]['percent']) if r[k]['percent'] is not None else '--' for k in ['stall_W10','physical_detour_0.5','latent_detour']]])
 table('mechanism.tex',['Task','Controller','Start','Stalling','Physical detour','Latent detour'],mrows,'lllrrr')
else:wait('tables_v2/mechanism.tex','all main controllers and Transported behavioral cells required')
(ROOT/'artifact-status.json').write_text(json.dumps(dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outputs=STATUS,all_requested_outputs_ready=all(not v.startswith('pending') for v in STATUS.values()),no_partial_estimates=True,plot_data=DATA),indent=2))
print(json.dumps(STATUS))
