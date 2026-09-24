"""Record-calibrated ranking with the frozen FP32 LeWM scoring interface."""
import numpy as np
import torch


class RecordCalibration:
    def __init__(self, native, handle, verify_source_batch=False):
        self.native = native
        self.handle = handle
        self.encodings = {}
        self.source_predictions = {}
        self.verify_source_batch = verify_source_batch

    def encoding(self, index):
        index = int(index)
        if index not in self.encodings:
            rgb = np.ascontiguousarray(self.handle['pixels'][index])
            self.encodings[index] = self.native.encode_rgb(rgb)[0].detach().cpu()
        return self.encodings[index].to(self.native.device)

    def predict(self, rgbs, current, normalized, target):
        # Preserve the original [batch, candidates, time, feature] F1 path.
        pixels = torch.stack([self.native.pixels(rgb)[0] for rgb in rgbs], dim=0)[None]
        candidates = torch.as_tensor(normalized.reshape(1, len(rgbs), 1, -1),
                                     device=self.native.device)
        info = dict(pixels=pixels, current_latent=current[None, :, None, :],
                    subgoal_latent=target[None, None, :].expand(1, len(rgbs), -1))
        self.native.latent_cost.get_cost(info, candidates)
        return info['predicted_emb'][0, :, -1, :]

    def predict_records(self, rgbs, current, normalized, target):
        # Distinct record states are separate queries (B), not candidates (N).
        pixels = torch.stack([self.native.pixels(rgb)[0] for rgb in rgbs], dim=0)[:, None]
        candidates = torch.as_tensor(normalized.reshape(len(rgbs), 1, 1, -1), device=self.native.device)
        info = dict(pixels=pixels, current_latent=current[:, None, None, :],
                    subgoal_latent=target[None, None, :].expand(len(rgbs), 1, -1))
        self.native.latent_cost.get_cost(info, candidates)
        return info['predicted_emb'][:, 0, -1, :]

    @torch.inference_mode()
    def choose(self, arm, rgb, sources, normalized):
        if arm == 'direct':
            return 0, {'source_prediction_misses': 0}, {}
        target = self.encoding(int(sources[0]) + 5)
        current = self.native.encode_rgb(rgb)[0]
        source_current = endpoint = live = source_pred = corrected = None
        if arm in ('local', 'calibrated'):
            live = self.predict([rgb] * len(sources), current[None].expand(len(sources), -1),
                                normalized, target)
        if arm in ('delta', 'calibrated'):
            source_current = torch.stack([self.encoding(index) for index in sources])
            endpoint = torch.stack([self.encoding(int(index) + 5) for index in sources])
            delta = current[None] + (endpoint - source_current)
        misses = []
        if arm == 'calibrated':
            misses = [i for i, index in enumerate(sources) if int(index) not in self.source_predictions]
            if misses:
                source_rgbs = [np.ascontiguousarray(self.handle['pixels'][int(sources[i])]) for i in misses]
                predicted = self.predict_records(source_rgbs, source_current[misses], normalized[misses], target)
                if self.verify_source_batch:
                    # Check source-axis semantics independently of cuDNN's
                    # batch-dependent TF32 kernels; policy predictions retain
                    # the original experiment's cuDNN TF32 setting.
                    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                    try:
                        torch.backends.cudnn.allow_tf32 = False
                        checked = self.predict_records(source_rgbs, source_current[misses], normalized[misses], target)
                        for position in range(min(2, len(misses))):
                            i = misses[position]
                            single = self.predict_records([source_rgbs[position]], source_current[[i]], normalized[[i]], target)[0]
                            torch.testing.assert_close(checked[position], single, rtol=1e-5, atol=1e-6)
                    finally:
                        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
                    self.verify_source_batch = False
                for i, value in zip(misses, predicted):
                    self.source_predictions[int(sources[i])] = value.detach().cpu()
            source_pred = torch.stack([self.source_predictions[int(index)] for index in sources]).to(current.device)
            corrected = endpoint + (live - source_pred)
            selected_effect = corrected
        elif arm == 'delta':
            selected_effect = delta
        else:
            selected_effect = live
        costs = (selected_effect - target[None]).square().sum(-1)
        rank = int(costs.argmin().item())
        arrays = {'score_target': target.cpu().numpy(), 'score_current': current.cpu().numpy(),
                  'selected_effects': selected_effect.cpu().numpy(), 'candidate_scores': costs.cpu().numpy()}
        for name, value in [('live_predictions', live), ('record_source_embeddings', source_current),
                            ('record_observed_endpoints', endpoint), ('record_source_predictions', source_pred),
                            ('calibrated_predictions', corrected)]:
            if value is not None:
                arrays[name] = value.cpu().numpy()
        if arm == 'calibrated':
            arrays['raw_scores'] = (live - target[None]).square().sum(-1).cpu().numpy()
            arrays['displacement_scores'] = (delta - target[None]).square().sum(-1).cpu().numpy()
        metadata = {'candidate_scores': costs.cpu().tolist(), 'source_prediction_misses': len(misses),
                    'target_index': int(sources[0]) + 5}
        return rank, metadata, arrays
