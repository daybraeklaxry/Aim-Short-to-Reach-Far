"""The two main-study searches and the local anchor/transport comparison.

Retrieval selects the support rows before these scorers are called. Gaussian
planners receive observation pixels only, never the recorded action array.
"""
import numpy as np
import torch

from .gaussian_cem import BLOCK, cem


class TargetRank:
    """One frozen-dynamics call over the same eight legal record actions."""
    def __init__(self, native, pixels):
        self.native, self.pixels = native, pixels
        self.encodings = {}

    def encoding(self, index):
        index = int(index)
        if index not in self.encodings:
            rgb = np.ascontiguousarray(self.pixels[index])
            self.encodings[index] = self.native.encode_rgb(rgb)[0].detach().cpu()
        return self.encodings[index].to(self.native.device)

    @torch.inference_mode()
    def choose(self, arm, rgb, sources, normalized, goal_record):
        if arm == "direct":
            return 0, dict(source_prediction_misses=0)
        assert arm in ("ap_observed", "ap_final") and len(sources) == 8
        index = int(sources[0]) + BLOCK if arm == "ap_observed" else int(goal_record)
        target = self.encoding(index)
        current = self.native.encode_rgb(rgb)[0]
        pixels = torch.stack([self.native.pixels(rgb)[0] for _ in range(8)], dim=0)[None]
        candidates = torch.as_tensor(normalized.reshape(1, 8, 1, -1), device=self.native.device)
        info = dict(pixels=pixels, current_latent=current[None, None, None, :].expand(1, 8, 1, -1),
                    subgoal_latent=target[None, None, :].expand(1, 8, -1))
        self.native.latent_cost.get_cost(info, candidates)
        live = info["predicted_emb"][0, :, -1, :]
        costs = (live - target[None]).square().sum(-1)
        rank = int(costs.argmin().item())
        return rank, dict(source_prediction_misses=0, target_index=index,
            target_kind="observed" if arm == "ap_observed" else "final",
            candidate_scores=costs.cpu().tolist())


class TargetGaussian:
    """Fixed standardized-zero prior, one-block CEM, explicit scoring target."""
    def __init__(self, native, pixels):
        self.native, self.pixels = native, pixels
        self.targets = {}

    def set_action_bounds(self, low, high):
        self.raw_bounds = np.stack((low, high)).astype(np.float32)
        self.normalized_bounds = self.native.scaler.transform(self.raw_bounds).astype(np.float32)

    def encoding(self, index):
        index = int(index)
        missed = index not in self.targets
        if missed:
            rgb = np.ascontiguousarray(self.pixels[index])
            self.targets[index] = self.native.encode_rgb(rgb).detach().cpu()
        return self.targets[index].to(self.native.device), int(missed)

    @torch.inference_mode()
    def choose(self, arm, rgb, sources, generator, goal_record):
        prior = np.zeros((BLOCK, self.native.action_dim), np.float32)
        raw_prior = self.native.scaler.inverse_transform(prior).astype(np.float32)
        effective = np.clip(raw_prior, self.raw_bounds[0], self.raw_bounds[1]).astype(np.float32)
        changed = not np.array_equal(effective, raw_prior)
        if changed:
            prior = self.native.scaler.transform(effective).astype(np.float32)
        current = self.native.encode_rgb(rgb)
        if arm in ("anchor", "transport"):
            source, source_miss = self.encoding(sources[0])
            successor, successor_miss = self.encoding(int(sources[0]) + BLOCK)
            target = successor if arm == "anchor" else current + successor - source
            index = int(sources[0]) + BLOCK
        else:
            assert arm in ("gaussian_observed", "gaussian_final")
            index = int(sources[0]) + BLOCK if arm == "gaussian_observed" else int(goal_record)
            target, successor_miss = self.encoding(index)
            source_miss = 0
        assert current.dtype == target.dtype == torch.float32
        chosen, detail = cem(self.native, rgb, current, target, generator, prior,
                             self.normalized_bounds, self.raw_bounds)
        if not detail["refined_mean_selected"]:
            chosen = effective.copy()
        detail.update(prior_kind="fixed_standardized_zero", prior_source=None,
            prior_projected=changed, target_index=index, target_kind=arm,
            source_index=int(sources[0]), source_prediction_misses=0,
            native_live_encodings=1, native_source_encodings=source_miss,
            native_successor_encodings=successor_miss,
            predictor_total_calls=detail["predictor_batch_calls"],
            predictor_total_candidate_blocks=detail["predictor_candidate_transitions"])
        return chosen, detail
