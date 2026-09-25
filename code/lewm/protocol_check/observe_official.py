"""Run an unchanged official evaluation script with read-only Python call profiling."""
from pathlib import Path
import argparse,json,os,runpy,sys,time
import numpy as np
import torch
from collections import deque
from types import SimpleNamespace
R=Path(__file__).parent
sys.path.insert(0,str(R/'deps'))
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,default=50);p.add_argument('--attempt',default='');p.add_argument('--mode',choices=['official','wrapper'],default='official')
a=p.parse_args();tag=f'{a.task}_seed{a.seed}_n{a.episodes}'+a.attempt
out=R/(a.mode+'_runs')/tag;out.mkdir(parents=True,exist_ok=False)
os.environ['STABLEWM_HOME']=str(R/'cache')
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
source=R/'official_source';sys.path.insert(0,str(source))
policy_name=f'audit_runs/{a.mode}_{tag}/lewm'
object_path=R/'cache'/(policy_name+'_object.ckpt')
object_path.parent.mkdir(parents=True,exist_ok=True)
if not object_path.exists():object_path.symlink_to(R/'cache'/a.task/'lewm_object.ckpt')
report={'task':a.task,'seed':a.seed,'episodes':a.episodes,'mode':a.mode,'status':'running','script':'official_source/eval.py','source_modified':False,'started_unix':time.time(),'solver_calls':0,'environment_steps':0}
report['render_backend']='EGL'
report['render_device']=os.environ.get('MUJOCO_EGL_DEVICE_ID','driver default')
report['egl_vendor_library']=os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES','driver default')
report['software_rendering']=os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
report['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
arrays={}
baseline=native=None
if a.mode=='wrapper':
    sys.path.insert(0,str(R.parent/'baseline'))
    import official_baseline as baseline
    _,_,_,_,_,_,_,native,_=baseline.runtime.load_runtime(a.task,'confirm')
    report['wrapper_entrypoints']=['official_baseline.policy_for','official_baseline.raw_info','official_baseline.released_block']
    report['audit_protocol_adaptations']=['Identical released environment construction and dataset initial/goal images','Per-query paper policies share the released vectorized CEM random stream in the same query order','Continue the released 50-step loop; success is the first hitting event, with episode work truncated at that event for comparisons']

class PaperPolicies:
    """Actual paper action code, scheduled in the reference environment order.

    This changes neither CEM nor its action buffer. Sharing the reference RNG
    avoids confounding sequential per-query seeding with wrapper equivalence.
    The reference dataset evaluator supplies precisely matched starts and goals.
    """
    def __init__(self,reference):
        self.reference=reference;self.queue=deque()
    def set_env(self,env):
        self.env=env
        model=self.reference.solver.model
        report['wrapper_checkpoint_tensors_equal']=all(torch.equal(v,native.model.state_dict()[k]) for k,v in model.state_dict().items())
        assert report['wrapper_checkpoint_tensors_equal']
        report['wrapper_scaler_equal']=all(np.array_equal(getattr(self.reference.process['action'],k),getattr(native.scaler,k)) for k in ['mean_','scale_'])
        assert report['wrapper_scaler_equal']
        self.policies=[baseline.policy_for(native,e,a.seed) for e in env.unwrapped.envs]
        for policy in self.policies:policy.solver.torch_gen=self.reference.solver.torch_gen
    @torch.inference_mode()
    def get_action(self,info):
        if not self.queue:
            blocks=[]
            for i,policy in enumerate(self.policies):
                raw=baseline.raw_info(native,info['pixels'][i,-1],info['goal'][i,-1])
                blocks.append(baseline.released_block(policy,raw))
            self.queue.extend(np.stack(blocks).transpose(1,0,2))
        return self.queue.popleft()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def profile(frame,event,arg):
    fn=frame.f_code.co_name
    file=frame.f_code.co_filename
    if a.mode=='wrapper' and event=='call' and fn=='evaluate_from_dataset' and file.endswith('/stable_worldmodel/world.py'):
        world=frame.f_locals['self'];world.set_policy(PaperPolicies(world.policy))
    elif event=='call' and fn=='solve' and file.endswith('/solver/cem.py'):
        solver=frame.f_locals['self'];i=report['solver_calls']
        arrays[f'solver_rng_{i}']=solver.torch_gen.get_state().cpu().numpy().copy()
        report['solver_calls']+=1
    elif event=='call' and fn=='step' and file.endswith('/stable_worldmodel/world.py') and report['environment_steps']%25==0:
        world=frame.f_locals['self'];i=report['environment_steps']
        for key,val in world.infos.items():
            if isinstance(val,np.ndarray) and val.dtype.kind in 'biuf':
                arrays[f'input_{key}_{i}']=val.copy()
    elif event=='return' and fn=='step' and file.endswith('/stable_worldmodel/world.py'):
        world=frame.f_locals['self'];i=report['environment_steps']
        arrays[f'actions_{i}']=np.asarray(frame.f_locals['actions']).copy()
        for key in ['terminateds','truncateds','rewards']:
            val=getattr(world,key,None)
            if val is not None:arrays[f'{key}_{i}']=np.asarray(val).copy()
        states=world.states
        if isinstance(states,dict):
            for key,val in states.items():
                if isinstance(val,np.ndarray) and val.dtype.kind in 'biuf' and val.ndim<=3:arrays[f'state_{key}_{i}']=val.copy()
        elif isinstance(states,np.ndarray) and states.dtype.kind in 'biuf' and states.ndim<=3:arrays[f'states_{i}']=states.copy()
        for key,val in world.infos.items():
            if isinstance(val,np.ndarray) and val.dtype.kind in 'biuf' and val.ndim<=3:
                arrays[f'info_{key}_{i}']=val.copy()
        report['environment_steps']+=1
        if report['environment_steps']%25==0:
            (out/'run.json').write_text(json.dumps(report,indent=2))
    elif event=='return' and fn=='run' and Path(file)==source/'eval.py':
        loc=frame.f_locals
        if 'metrics' in loc:
            report['metrics']=plain(loc['metrics'])
            for key in ['random_episode_indices','eval_episodes','eval_start_idx']:
                report[key]=plain(loc[key])
            report['seconds']=float(loc['end_time']-loc['start_time'])
            report['status']='complete'
    return profile
sys.argv=[str(source/'eval.py'),f'--config-name={a.task}',f'policy={policy_name}',f'seed={a.seed}',f'eval.num_eval={a.episodes}',f'output.filename={out}/official_output.txt',f'hydra.run.dir={out}/hydra']
(out/'run.json').write_text(json.dumps(report,indent=2))
try:
    sys.setprofile(profile)
    runpy.run_path(str(source/'eval.py'),run_name='__main__')
except BaseException as e:
    if report['status']!='complete':report.update(status='error',error=repr(e))
    raise
finally:
    sys.setprofile(None)
    report['wall_seconds']=time.time()-report['started_unix']
    (out/'run.json').write_text(json.dumps(report,indent=2))
    if arrays:np.savez_compressed(out/'observed_runtime.npz',**arrays)
    print('AUDIT_RESULT '+json.dumps(report),flush=True)
