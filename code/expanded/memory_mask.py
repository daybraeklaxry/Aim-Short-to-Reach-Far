"""Apply the reviewed episode exclusion before membership-dependent lookup."""
import numpy as np

def apply(context,protocol,task):
    bank=context['bank'];removed=np.asarray(protocol['removed_memory_episodes'][task],np.int64)
    expected=np.asarray(protocol['bank_train_episodes'][task],np.int64)
    original_count=int(bank.train_mask.sum())
    if task=='pusht':
        # Fresh process: native _push_context constructs a bank but never calls for_delta.
        assert bank.cached_delta is None and bank.cached_bank is None and bank.cached_squared_norm is None
        assert len(removed)==128 and bank.train_mask[removed].all()
        bank.train_mask[removed]=False
    else:
        assert len(removed)==0
    np.testing.assert_array_equal(np.flatnonzero(bank.train_mask),expected)
    queries=np.asarray([r['episode'] for r in protocol['studies'][task]['confirm']],np.int64)
    assert not bank.train_mask[queries].any()
    return dict(original_train_episodes=original_count,active_train_episodes=int(bank.train_mask.sum()),
        removed_episodes=removed.tolist(),query_episode_overlap=0,
        applied_before_first_membership_cache=task=='pusht',
        feature_statistics='native for_delta mean/std computed from active members using unchanged formula, weights, precision and floor',
        native_action_scaler='unchanged frozen values',predictor='unchanged frozen checkpoint')
