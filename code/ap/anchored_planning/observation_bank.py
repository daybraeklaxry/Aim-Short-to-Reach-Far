from typing import Any
import numpy as np
import torch

def _pair_features(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    relative = goal - current
    return np.concatenate((current, goal, relative), axis=-1).astype(np.float32, copy=False)

class ObservationBank:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        latents: np.ndarray,
        train_episodes: np.ndarray,
        option_steps: int,
        weights: np.ndarray,
        device: torch.device,
    ) -> None:
        self.lengths = arrays["lengths"]
        self.offsets = arrays["offsets"]
        self.episode_idx = np.repeat(np.arange(len(self.lengths)), self.lengths)
        self.latents = np.asarray(latents, dtype=np.float32)
        self.latents_device = torch.from_numpy(self.latents).to(device)
        self.train_mask = np.zeros(len(self.lengths), dtype=bool)
        self.train_mask[np.asarray(train_episodes, dtype=np.int64)] = True
        self.option_steps = int(option_steps)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.device = device
        self.cached_delta: int | None = None
        self.cached_bank: tuple[
            np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self.cached_squared_norm: torch.Tensor | None = None

    def for_delta(self, delta: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = int(delta)
        if self.cached_delta != delta:
            # Only one remaining-horizon bank is live at a time. Dropping the
            # previous tensor first removes a 2x peak without changing values.
            self.cached_bank = None
            self.cached_squared_norm = None
            positions = np.arange(len(self.latents), dtype=np.int64)
            valid = positions[
                self.train_mask[self.episode_idx]
                & (positions + delta < self.offsets[self.episode_idx] + self.lengths[self.episode_idx])
                & (positions + self.option_steps < self.offsets[self.episode_idx] + self.lengths[self.episode_idx])
            ]
            if len(valid) == 0:
                raise ValueError(f"no train bank rows for delta={delta}")
            valid_tensor = torch.from_numpy(valid).long().to(self.device)
            latent_dim = self.latents_device.shape[1]
            features = torch.empty(
                (len(valid), 3 * latent_dim),
                dtype=self.latents_device.dtype,
                device=self.device,
            )
            features[:, :latent_dim] = self.latents_device.index_select(0, valid_tensor)
            features[:, latent_dim : 2 * latent_dim] = self.latents_device.index_select(
                0, valid_tensor + delta
            )
            features[:, 2 * latent_dim :] = (
                features[:, latent_dim : 2 * latent_dim] - features[:, :latent_dim]
            )
            del valid_tensor
            mean = features.mean(dim=0)
            std = features.std(dim=0, correction=0).clamp_min(1e-4)
            features.sub_(mean).div_(std)
            normalized = features
            normalized = normalized.reshape(len(valid), 3, -1)
            normalized.mul_(
                torch.from_numpy(np.sqrt(self.weights)).to(self.device).view(1, 3, 1)
            )
            normalized = normalized.reshape(len(valid), -1)
            norm_chunk = 131_072
            squared_norm = torch.cat(
                [
                    normalized[start : start + norm_chunk].square().sum(dim=1)
                    for start in range(0, len(normalized), norm_chunk)
                ]
            )
            self.cached_delta = delta
            self.cached_bank = (valid, normalized, mean, std)
            self.cached_squared_norm = squared_norm
        if self.cached_bank is None:
            raise RuntimeError("latent retrieval bank was not initialized")
        return self.cached_bank

    def nearest_topk(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        delta: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k support search for a fixed local verifier.

        The bank remains train-only and returns source indices and distances.
        No source action values are loaded or returned.  Chunked merging keeps the query-by-bank
        matrix bounded while preserving exact global top-k ordering.
        """
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        valid, bank, mean, std = self.for_delta(delta)
        if self.cached_squared_norm is None:
            raise RuntimeError("retrieval bank norms were not initialized")
        features = _pair_features(current, goal)
        query = (torch.from_numpy(features).float().to(self.device) - mean) / std
        query = query.reshape(len(query), 3, -1)
        query = query * torch.from_numpy(np.sqrt(self.weights)).to(self.device).view(1, 3, 1)
        query = query.reshape(len(query), -1)
        k = min(top_k, len(bank))
        best_values = torch.full((len(query), k), torch.inf, device=self.device)
        best_positions = torch.zeros((len(query), k), dtype=torch.long, device=self.device)
        query_norm = query.square().sum(dim=1, keepdim=True)
        # The EXP4 X3 mechanism runner evaluates long horizons with a fixed
        # 30 GB per-process cap. A larger exact chunk reduces launch overhead
        # without changing the distance or top-k merge semantics.
        chunk_size = 1_048_576
        for start in range(0, len(bank), chunk_size):
            chunk = bank[start : start + chunk_size]
            distances = (
                query_norm
                + self.cached_squared_norm[start : start + chunk.size(0)].unsqueeze(0)
                - 2.0 * (query @ chunk.transpose(0, 1))
            ).clamp_min_(0.0)
            chunk_k = min(k, chunk.size(0))
            chunk_values, chunk_positions = distances.topk(chunk_k, dim=1, largest=False, sorted=True)
            merged_values = torch.cat((best_values, chunk_values), dim=1)
            merged_positions = torch.cat((best_positions, chunk_positions + start), dim=1)
            best_values, order = merged_values.topk(k, dim=1, largest=False, sorted=True)
            best_positions = merged_positions.gather(1, order)
        indices = valid[best_positions.cpu().numpy()]
        return indices, best_values.float().cpu().numpy()

