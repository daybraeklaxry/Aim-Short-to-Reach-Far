"""Two frozen search rules; only observed versus final scoring target differs."""
import numpy as np
import torch
from observed_target import ValidObservedCEM, cem, BLOCK
from rank_reference import RecordCalibration

ARMS=('direct','ap_observed','ap_final','gaussian_observed','gaussian_final')

class TargetRank(RecordCalibration):
    def set_goal_record(self,index):
        self.goal_record=int(index)

    @torch.inference_mode()
    def choose(self,arm,rgb,sources,normalized):
        if arm=='direct':
            return 0,dict(source_prediction_misses=0),{}
        assert arm in ('ap_observed','ap_final') and len(sources)==8
        index=int(sources[0])+BLOCK if arm=='ap_observed' else self.goal_record
        target=self.encoding(index)
        current=self.native.encode_rgb(rgb)[0]
        live=self.predict([rgb]*8,current[None].expand(8,-1),normalized,target)
        costs=(live-target[None]).square().sum(-1)
        rank=int(costs.argmin().item())
        detail=dict(source_prediction_misses=0,target_index=index,
                    target_kind='observed' if arm=='ap_observed' else 'final',
                    candidate_scores=costs.cpu().tolist())
        arrays=dict(score_current=current.cpu().numpy(),score_target=target.cpu().numpy(),
                    live_predictions=live.cpu().numpy(),candidate_scores=costs.cpu().numpy())
        return rank,detail,arrays

class TargetGaussian(ValidObservedCEM):
    def __init__(self,native,handle,target_kind):
        super().__init__(native,handle)
        assert target_kind in ('observed','final')
        self.target_kind=target_kind

    def set_goal_record(self,index):
        self.goal_record=int(index)

    def target(self,source):
        index=int(source)+BLOCK if self.target_kind=='observed' else self.goal_record
        if index not in self.targets:
            self.targets[index]=self.native.encode_rgb(np.ascontiguousarray(self.handle['pixels'][index])).detach().cpu()
        return self.targets[index].to(self.native.device)

    @torch.inference_mode()
    def choose(self,rgb,sources,raw,normalized,generator):
        # No recorded action is used in the initial mean or incumbent.
        prior=np.zeros((BLOCK,self.native.action_dim),np.float32)
        raw_prior=self.native.scaler.inverse_transform(prior).astype(np.float32)
        effective=np.clip(raw_prior,self.raw_bounds[0],self.raw_bounds[1]).astype(np.float32)
        changed=not np.array_equal(effective,raw_prior)
        if changed:prior=self.native.scaler.transform(effective).astype(np.float32)
        current=self.native.encode_rgb(rgb);target=self.target(sources[0])
        chosen,detail=cem(self.native,rgb,current,target,generator,prior,self.normalized_bounds,self.raw_bounds)
        if not detail['refined_mean_selected']:chosen=effective.copy()
        index=int(sources[0])+BLOCK if self.target_kind=='observed' else self.goal_record
        detail.update(prior_kind='fixed_standardized_zero',prior_source=None,
            prior_projected=changed,prior_projection_max_abs=float(np.max(np.abs(effective-raw_prior))),
            target_index=index,target_kind=self.target_kind,source_prediction_misses=0,
            predictor_total_calls=detail['predictor_batch_calls'],
            predictor_total_candidate_blocks=detail['predictor_candidate_transitions'])
        arrays=dict(score_current=current[0].cpu().numpy(),score_target=target[0].cpu().numpy(),
            proposed_raw_actions=chosen.copy(),normalized_prior=prior.copy(),effective_raw_prior=effective.copy(),
            normalized_selected=np.asarray(detail['normalized_selected'],np.float32),
            raw_action_bounds=self.raw_bounds.copy(),normalized_action_bounds=self.normalized_bounds.copy())
        return chosen,detail,arrays
