import numpy as np
import torch
BLOCK,SAMPLES,ITERATIONS,ELITES=5,300,30,30
SIGMA,STD_FLOOR=1.0/3.0,1e-5

def cem(native, rgb, current, target, generator, normalized_prior, normalized_bounds, raw_bounds):
    """The old CEM update and prior/mean choice, with an explicitly supplied per-decision prior."""
    prior = torch.as_tensor(normalized_prior, dtype=torch.float32, device=native.device).reshape(1, 1, BLOCK * native.action_dim).clone()
    lower = torch.as_tensor(np.tile(normalized_bounds[0], BLOCK), dtype=torch.float32, device=native.device)
    upper = torch.as_tensor(np.tile(normalized_bounds[1], BLOCK), dtype=torch.float32, device=native.device)
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
            if getattr(native,'capture_initial_noise',False):
                native.initial_standard_noise=noise.detach().cpu().numpy().copy()
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
    roundtrip_error = float(np.max(np.abs(native.scaler.transform(raw) - normalized)))
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
        normalized_execution_roundtrip_error=roundtrip_error)

def query_seed(task, ordinal):
    if task == 'pusht':
        return 26091331 + 1000 * ordinal
    return 26091351 + 1000000 * ['cube', 'reacher', 'tworoom'].index(task) + 1000 * ordinal
