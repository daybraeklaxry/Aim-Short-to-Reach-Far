"""Released LeWM planning: history_len=1, rolling history=3, CEM horizon=5.

Two prescribed execution arms: native_official and native_matched5. No simulator reset,
success test, action clipping, training, or checkpoint replacement lives here.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms
from stable_worldmodel.solver.cem import CEMSolver


ARMS = {'native_official': 25, 'native_matched5': 5}
HORIZON = 5
ACTION_BLOCK = 5


def make_native_cem(cost_model, action_dim, device):
    """Use the installed released solver, normalized mean=0 and initial std=1.

    CEMSolver reads only this space's shape; it does not clip to its bounds.
    Subsequent warm-start means are supplied explicitly to solve_normalized.
    """
    solver = CEMSolver(model=cost_model, batch_size=1, num_samples=300,
                       var_scale=1.0, n_steps=30, topk=30,
                       device=str(device), seed=42)
    shape_space = Box(low=-1.0, high=1.0, shape=(1, int(action_dim)), dtype=np.float32)
    solver.configure(action_space=shape_space, n_envs=1,
                     config=SimpleNamespace(horizon=HORIZON, action_block=ACTION_BLOCK))
    return solver


def solve_normalized(solver, info, init_action=None):
    """Return the actual official CEM result, including [1,5,5*A] actions."""
    return solver.solve(info, init_action=init_action)


class _DirectCost:
    """Call official get_cost unchanged; retain its last encoded inputs for logs."""
    def __init__(self, official):
        self.official = official

    def get_cost(self, info, candidates):
        cost = self.official.get_cost(info, candidates)
        self.current = info['emb'][:, 0, -1].detach()
        self.goal = info['goal_emb'][:, -1].detach()
        return cost


class _CachedEncodingView:
    """Replace only encode with supplied embeddings; reuse official rollout code."""
    def __init__(self, official):
        self.action_encoder = official.action_encoder
        self.predict = official.predict

    def encode(self, info):
        info['emb'] = info['current_latent']
        return info


class LatentGoalCost:
    """Official rolling-history F1 and criterion, with an F2 latent target.

    current_latent is [B,1,192] before CEM expansion. If supplied it must be
    produced by the same fp32 projected encoder. Pixels retain the true initial
    history length; no artificial history frames or actions are added.
    """
    def __init__(self, official):
        self.official = official
        self.cached_view = _CachedEncodingView(official)

    def get_cost(self, info, candidates):
        target = info.pop('subgoal_latent')
        if 'current_latent' in info:
            # This executes JEPA.rollout itself, not a reimplemented recurrence.
            rollout = self.official.rollout.__func__(self.cached_view, info, candidates)
        else:
            rollout = self.official.rollout(info, candidates)
        rollout['goal_emb'] = target[:, :, None, :]
        return self.official.criterion(rollout)


def scaler_from_record(record):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(record['mean'], dtype=np.float64)
    scaler.scale_ = np.asarray(record['std'], dtype=np.float64)
    scaler.var_ = np.asarray(record['var'], dtype=np.float64)
    scaler.n_features_in_ = scaler.mean_.size
    scaler.n_samples_seen_ = int(record['count'])
    return scaler


class NativeAdapter:
    def __init__(self, ew, context, task, protocol, root, arm='native_matched5'):
        if arm not in ARMS:
            raise ValueError('Prescribed arms are native_official and native_matched5')
        self.ew, self.context, self.task, self.arm = ew, context, task, arm
        self.device = torch.device(context['device'])
        self.execute_steps = ARMS[arm]
        self.model = context['model'].official.eval().requires_grad_(False)
        if next(self.model.parameters()).dtype != torch.float32:
            raise ValueError('Corrected Native requires the original fp32 LeWM model')
        records = protocol.get('native_action_stats')
        if records is None:
            records = json.loads((Path(root) / 'native_action_stats.json').read_text())
        record = records[task]
        if record['scope'] != 'full original dataset' or record['statistics_dtype'] != 'float64':
            raise ValueError('Use the corrected full-dataset StandardScaler records')
        self.scaler = scaler_from_record(record)
        self.mean, self.std = self.scaler.mean_.copy(), self.scaler.scale_.copy()
        self.normalization_source = record['source']
        self.action_dim = self.mean.size
        self.transform = transforms.Compose([
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.Resize(size=224),
        ])
        self.direct_cost = _DirectCost(self.model)
        self.solver = make_native_cem(self.direct_cost, self.action_dim, self.device)
        self.latent_cost = LatentGoalCost(self.model)
        self.latent_solver = make_native_cem(self.latent_cost, self.action_dim, self.device)
        self._next_init = self._latent_next_init = None
        self.goal_batch_size = 1

    def prepare_rows(self, rows):
        # Released get_cost encodes each query goal within its CEM call.
        pass

    def pixels(self, rgb):
        frame = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        return self.transform(frame)[None, None].to(self.device)

    @torch.inference_mode()
    def encode_rgb(self, rgb):
        """Official fp32 projected E; return a device tensor of shape [1,192]."""
        return self.model.encode({'pixels': self.pixels(rgb)})['emb'][:, -1].float()

    def begin(self, row, cached):
        self.row = row
        self.current_rgb = np.ascontiguousarray(cached['initial_rgb']).copy()
        self.goal_rgb = np.ascontiguousarray(cached['goal_rgb']).copy()
        self.goal = np.asarray(cached['goal'], dtype=np.float32).reshape(1, -1).copy()
        self.initial_history = self.current_rgb[None].copy()
        self.initial_past = np.empty((0, self.action_dim * ACTION_BLOCK), dtype=np.float32)
        self._next_init = self._latent_next_init = None

    def _raw(self, normalized):
        values = normalized.detach().cpu().numpy().reshape(-1, self.action_dim)
        return self.scaler.inverse_transform(values).astype(np.float32, copy=False)

    def _metadata(self, result, initialized_blocks):
        return dict(method=f'corrected_{self.arm}', history_len=1,
                    rolling_prediction_history=3, macro_horizon=5, action_block=5,
                    planned_primitive_steps=25, replan_steps=self.execute_steps,
                    cem_samples=300, cem_iterations=30, cem_topk=30,
                    cem_seed=42, rng='continuous per-solver stream; not reset per decision',
                    normalized_initial_std=1.0, warm_start=True,
                    warm_start_macro_blocks=initialized_blocks,
                    candidate_clipping=False, action_clipping=False,
                    output='final elite mean; StandardScaler.inverse_transform',
                    objective='terminal squared L2 sum', precision='fp32 without autocast',
                    predictor_calls=150, candidate_prediction_evaluations=45000,
                    last_elite_mean_cost=float(result['costs'][0]))

    @torch.inference_mode()
    def choose(self, current, step):
        info = dict(pixels=self.pixels(self.current_rgb), goal=self.pixels(self.goal_rgb),
                    action=torch.zeros((1, 1, self.action_dim * ACTION_BLOCK), device=self.device))
        initialized = 0 if self._next_init is None else self._next_init.shape[1]
        result = solve_normalized(self.solver, info, self._next_init)
        normalized = result['actions']
        self.last_normalized_plan = normalized.detach().cpu().numpy().copy()
        self.last_raw_plan = self._raw(normalized).copy()
        self._next_init = normalized[:, self.execute_steps // ACTION_BLOCK:].clone()
        raw = self.last_raw_plan[:self.execute_steps].copy()
        actual_current = self.direct_cost.current.float().cpu().numpy()
        self.goal = self.direct_cost.goal.float().cpu().numpy()
        trace = self._metadata(result, initialized)
        trace['encoding_calls_in_cost'] = 60
        return raw, trace, dict(current=actual_current[0], history=actual_current,
                               past=self.initial_past.copy())

    @torch.inference_mode()
    def plan_to_latent(self, current_rgb, subgoal_latent, current_latent=None):
        """HWM low layer: full raw plan [25,A], same official solver and L2.

        Execute the first self.execute_steps actions. begin clears warm starts
        at a new query. The high-layer optimizer must own a separate RNG stream.
        """
        initial = self.encode_rgb(current_rgb) if current_latent is None else current_latent
        initial = torch.as_tensor(initial, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        target = torch.as_tensor(subgoal_latent, dtype=torch.float32, device=self.device).reshape(1, -1)
        info = dict(pixels=self.pixels(current_rgb), current_latent=initial, subgoal_latent=target)
        initialized = 0 if self._latent_next_init is None else self._latent_next_init.shape[1]
        result = solve_normalized(self.latent_solver, info, self._latent_next_init)
        normalized = result['actions']
        self._latent_next_init = normalized[:, self.execute_steps // ACTION_BLOCK:].clone()
        trace = self._metadata(result, initialized)
        trace.update(method='corrected_hwm_native_low', target='F2 first latent subgoal',
                     encoding_calls_in_cost=0, current_encoder_calls=int(current_latent is None))
        return self._raw(normalized), trace

    def completed_primitive(self, env, plan_index, continuing):
        if continuing and plan_index + 1 == self.execute_steps:
            self.current_rgb = np.ascontiguousarray(env.render()).copy()
            return 1
        return 0

    def rng_state(self):
        return self.solver.torch_gen.get_state().cpu().numpy().copy()

    def restore_rng(self, state):
        self.solver.torch_gen.set_state(torch.from_numpy(np.asarray(state, np.uint8).copy()))
