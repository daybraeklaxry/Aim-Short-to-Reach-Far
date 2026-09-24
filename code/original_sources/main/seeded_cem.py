"""Source-reference CEM in a fixed legal recorded-action neighborhood."""
import numpy as np
import torch

BLOCK, SAMPLES, ITERATIONS, ELITES = 5, 300, 30, 30
SIGMA, STD_FLOOR = 1.0 / 3.0, 1e-5
TRUST_RADIUS = SIGMA

def cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds):
    """The old CEM update and prior/mean choice, with an explicitly supplied per-decision prior."""
    prior = torch.as_tensor(normalized_prior, dtype=torch.float32, device=native.device).reshape(1, 1, BLOCK * native.action_dim).clone()
    lower = torch.as_tensor(np.tile(normalized_bounds[0], BLOCK), dtype=torch.float32, device=native.device)
    upper = torch.as_tensor(np.tile(normalized_bounds[1], BLOCK), dtype=torch.float32, device=native.device)
    # Keep the existing sample scale, and explicitly bound each correction by it.
    lower = torch.maximum(lower, prior[0, 0] - TRUST_RADIUS)
    upper = torch.minimum(upper, prior[0, 0] + TRUST_RADIUS)
    projected_components = torch.zeros((), dtype=torch.int64, device=native.device)
    projected_blocks = torch.zeros((), dtype=torch.int64, device=native.device)
    scored_min = torch.full_like(lower, float('inf'))
    scored_max = torch.full_like(upper, float('-inf'))

    def project(value):
        return torch.minimum(torch.maximum(value, lower), upper)

    assert torch.equal(project(prior), prior), 'Recorded prior is outside the action domain.'
    mean = prior.clone()
    std = torch.full_like(mean, SIGMA)
    pixels = native.pixels(rgb)
    work_calls = work_transitions = 0

    def cost(actions):
        nonlocal work_calls, work_transitions
        bounded = project(actions)
        changed = bounded != actions
        projected_components.add_(changed.sum())
        projected_blocks.add_(changed.any(dim=-1).sum())
        actions = bounded
        scored_min.copy_(torch.minimum(scored_min, actions.amin(dim=(0, 1, 2))))
        scored_max.copy_(torch.maximum(scored_max, actions.amax(dim=(0, 1, 2))))
        count = actions.shape[1]
        info = dict(pixels=pixels[:, None].expand(-1, count, -1, -1, -1, -1),
            current_latent=current[:, None, None, :].expand(-1, count, -1, -1),
            subgoal_latent=target[:, None, :].expand(-1, count, -1))
        values = native.latent_cost.get_cost(info, actions)
        assert values.shape == (1, count) and torch.isfinite(values).all()
        work_calls += 1
        work_transitions += count
        return values, bounded

    first_noise = None
    for iteration in range(ITERATIONS):
        noise = torch.randn(1, SAMPLES, 1, BLOCK * native.action_dim,
                            generator=generator, device=native.device)
        if iteration == 0:
            first_noise = noise[0, 1, 0].cpu().tolist()
        candidates = noise * std[:, None] + mean[:, None]
        candidates[:, 0] = mean
        values, _ = cost(candidates)
        elite = torch.topk(values, ELITES, dim=1, largest=False).indices
        elites = candidates[torch.arange(1, device=native.device)[:, None], elite]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1).clamp_min(STD_FLOOR)
    values, effective = cost(torch.stack((prior, mean), dim=1))
    improved = bool((values[0, 1] < values[0, 0]).item())
    chosen = effective[:, 1] if improved else effective[:, 0]
    assert work_calls == 31 and work_transitions == 9002
    normalized = chosen[0, 0].cpu().numpy().reshape(BLOCK, native.action_dim)
    raw = native.scaler.inverse_transform(normalized).astype(np.float32)
    # Account for the last float32 scaler round trip at a box boundary.
    raw = np.clip(raw, raw_bounds[0], raw_bounds[1]).astype(np.float32)
    executed_normalized = native.scaler.transform(raw).astype(np.float32)
    roundtrip_error = float(np.max(np.abs(executed_normalized - normalized)))
    local_lower = lower.cpu().numpy().reshape(BLOCK, native.action_dim)
    local_upper = upper.cpu().numpy().reshape(BLOCK, native.action_dim)
    local_violation = float(max(0.0, np.max(local_lower-executed_normalized),
                                np.max(executed_normalized-local_upper)))
    assert local_violation <= 1e-6, local_violation
    assert raw.shape == (BLOCK, native.action_dim) and np.isfinite(raw).all()
    return raw, dict(normalized_prior=prior[0, 0].cpu().tolist(),
        normalized_selected=normalized.tolist(), refined_mean_selected=improved,
        prior_cost=float(values[0, 0]), refined_cost=float(values[0, 1]),
        selected_cost=float(values.min()), first_standard_noise=first_noise,
        predictor_batch_calls=work_calls, predictor_candidate_transitions=work_transitions,
        projected_scored_blocks=int(projected_blocks.item()),
        projected_scored_components=int(projected_components.item()),
        scored_normalized_min=scored_min.cpu().tolist(),
        scored_normalized_max=scored_max.cpu().tolist(),
        normalized_local_lower=local_lower.tolist(), normalized_local_upper=local_upper.tolist(),
        normalized_trust_radius=TRUST_RADIUS,
        executed_normalized_local_violation=local_violation,
        normalized_execution_roundtrip_error=roundtrip_error)


def query_seed(task, ordinal):
    if task == 'pusht':
        return 26091331 + 1000 * ordinal
    return 26091351 + 1000000 * ['cube', 'reacher', 'tworoom'].index(task) + 1000 * ordinal


class LocalTrustCEM:
    def __init__(self, native, handle):
        self.native, self.handle = native, handle
        self.references = {}

    def set_action_bounds(self, low, high):
        self.raw_bounds = np.stack((low, high)).astype(np.float32)
        assert self.raw_bounds.shape == (2, self.native.action_dim)
        assert np.isfinite(self.raw_bounds).all()
        assert np.all(self.raw_bounds[0] < self.raw_bounds[1])
        self.normalized_bounds = self.native.scaler.transform(self.raw_bounds).astype(np.float32)

    def reference(self, source, normalized_prior):
        index = int(source)
        missed = index not in self.references
        if missed:
            rgb = np.ascontiguousarray(self.handle['pixels'][index])
            current = self.native.encode_rgb(rgb)
            pixels = self.native.pixels(rgb)[:, None]
            actions = torch.as_tensor(normalized_prior.reshape(1,1,1,-1),
                                      dtype=torch.float32,device=self.native.device)
            info = dict(pixels=pixels, current_latent=current[:,None,None,:],
                        subgoal_latent=current[:,None,:])
            # NativeLatentCost removes the scoring target before rollout.
            # The target here is only a placeholder for obtaining F(source, prior).
            self.native.latent_cost.get_cost(info, actions)
            prediction = info['predicted_emb'][:,0,-1,:]
            assert prediction.shape == current.shape and torch.isfinite(prediction).all()
            self.references[index] = (current.detach().cpu(), prediction.detach().cpu())
        current, prediction = self.references[index]
        return current.to(self.native.device), prediction.to(self.native.device), int(missed)

    @torch.inference_mode()
    def choose(self, rgb, sources, raw, normalized, generator):
        effective_raw_prior = np.clip(raw[0], self.raw_bounds[0], self.raw_bounds[1]).astype(np.float32)
        prior_projected = not np.array_equal(effective_raw_prior, raw[0])
        # P is the identity for a legal record, including its supplied normalization.
        effective_normalized_prior = self.native.scaler.transform(effective_raw_prior).astype(np.float32) \
            if prior_projected else normalized[0]
        current = self.native.encode_rgb(rgb)
        source_current, target, misses = self.reference(sources[0], effective_normalized_prior)
        chosen, detail = cem(self.native, rgb, current, target, generator, effective_normalized_prior,
                             self.normalized_bounds, self.raw_bounds)
        if not detail['refined_mean_selected']:
            # Reuse the exact effective prior; legal records stay literally unchanged.
            chosen = effective_raw_prior.copy()
        detail.update(prior_source=int(sources[0]), reference_source=int(sources[0]),
                      recorded_prior_projected=prior_projected,
                      recorded_prior_projection_max_abs=float(np.max(np.abs(effective_raw_prior-raw[0]))),
                      target_index=None, source_prediction_misses=misses,
                      predictor_total_calls=detail['predictor_batch_calls']+misses,
                      predictor_total_candidate_blocks=detail['predictor_candidate_transitions']+misses)
        arrays = dict(score_current=current[0].cpu().numpy(), score_target=target[0].cpu().numpy(),
                      proposed_raw_actions=chosen.copy(), normalized_prior=effective_normalized_prior.copy(), effective_raw_prior=effective_raw_prior.copy(),
                      normalized_selected=np.asarray(detail['normalized_selected'],np.float32),
                      reference_source_embedding=source_current[0].cpu().numpy(),
                      reference_prediction=target[0].cpu().numpy(),
                      raw_action_bounds=self.raw_bounds.copy(),
                      normalized_action_bounds=self.normalized_bounds.copy(),
                      normalized_local_lower=np.asarray(detail['normalized_local_lower'],np.float32),
                      normalized_local_upper=np.asarray(detail['normalized_local_upper'],np.float32))
        return chosen, detail, arrays

    @torch.inference_mode()
    def verify_source_consistency(self, source, raw_prior, normalized_prior, seed):
        # A separate scorer leaves the actual policy cache and generator untouched.
        verifier = LocalTrustCEM(self.native, self.handle)
        verifier.set_action_bounds(*self.raw_bounds)
        rgb = np.ascontiguousarray(self.handle['pixels'][int(source)])
        generator = torch.Generator(device=self.native.device).manual_seed(seed)
        chosen, detail, arrays = verifier.choose(rgb, np.asarray([source]),
            raw_prior[None], normalized_prior[None], generator)
        np.testing.assert_array_equal(arrays['score_current'], arrays['reference_source_embedding'])
        assert detail['prior_cost'] <= 1e-6, detail['prior_cost']
        effective_raw_prior = np.clip(raw_prior, self.raw_bounds[0], self.raw_bounds[1]).astype(np.float32)
        effective_normalized_prior = self.native.scaler.transform(effective_raw_prior).astype(np.float32) \
            if not np.array_equal(effective_raw_prior, raw_prior) else normalized_prior
        np.testing.assert_allclose(chosen, effective_raw_prior, rtol=0, atol=1e-6)
        np.testing.assert_allclose(arrays['normalized_selected'], effective_normalized_prior, rtol=0, atol=1e-6)
        return dict(passed=True, source=int(source), prior_cost=detail['prior_cost'],
                    refined_cost=detail['refined_cost'], selected_cost=detail['selected_cost'],
                    refined_mean_selected=detail['refined_mean_selected'],
                    recorded_prior_projected=detail['recorded_prior_projected'],
                    action_reference='P(recorded raw prior) using the environment Box and original scaler',
                    raw_action_bitwise_equal=bool(np.array_equal(chosen, effective_raw_prior)),
                    normalized_action_bitwise_equal=bool(np.array_equal(arrays['normalized_selected'], effective_normalized_prior)),
                    raw_action_max_abs_difference=float(np.max(np.abs(chosen-effective_raw_prior))),
                    normalized_action_max_abs_difference=float(np.max(np.abs(arrays['normalized_selected']-effective_normalized_prior))),
                    action_equivalence_atol=1e-6, action_equivalence_rtol=0,
                    semantics='FP32 effective source-action equivalence; the actual strict selection branch is reported without alteration.',
                    verification_candidate_blocks=detail['predictor_total_candidate_blocks'],
                    verification_predictor_calls=detail['predictor_total_calls'],
                    included_in_policy_work=False)
