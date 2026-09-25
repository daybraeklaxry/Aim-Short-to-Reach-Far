    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info
