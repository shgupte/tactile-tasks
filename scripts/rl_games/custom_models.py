import torch
import torch.nn as nn

from rl_games.algos_torch.models import ModelA2C
from rl_games.algos_torch.torch_ext import register_network

# PointNet encoder
from tactile_tasks.tasks.manager_based.tactile_tasks.pointnet import PointNet


@register_network('pointnet_ac_policy')
class PointNetPolicy(ModelA2C):
    """PointNet-augmented policy that preserves rl-games' actor_critic heads.

    How it works:
      - The environment provides a flat observation vector.
      - The first pc_points * pc_channels entries are interpreted as a fixed-size
        point cloud (row-major). We encode it with PointNet to a global latent.
      - The remaining entries ("rest") are passed through a small MLP (optional).
      - We concatenate [pointnet_latent, rest_latent] and feed that into the
        default rl-games actor_critic network built via the builder.
    """

    def __init__(self, obs_shape, actions_num, **kwargs):
        # kwargs may include 'mlp', 'space', 'cnn', etc. from YAML
        super().__init__(obs_shape, actions_num, **kwargs)

        # --- PointNet + rest-MLP front-end ---
        pc_cfg = kwargs.get('pointnet', {}) or {}
        self.pc_points = int(pc_cfg.get('pc_points', 4096))
        self.pc_channels = int(pc_cfg.get('pc_channels', 3))  # 3=xyz, 6=xyzrgb
        self.global_feat_dim = int(pc_cfg.get('global_feat_dim', 1024))
        self.use_feature_transform = bool(pc_cfg.get('use_feature_transform', True))

        self.pointnet = PointNet(
            in_channels=self.pc_channels,
            global_feat_dim=self.global_feat_dim,
            use_feature_transform=self.use_feature_transform,
        )

        obs_dim_total = int(obs_shape[0])
        pc_dim = self.pc_points * self.pc_channels
        self.rest_dim = max(0, obs_dim_total - pc_dim)

        # Standardize: feed raw rest scalars directly (no extra projection)
        # Keeping an Identity ensures the head input size remains deterministic
        # keep an eye on this code!!!
        if self.rest_dim > 0:
            self.rest_mlp = nn.Identity()
            rest_out = self.rest_dim
        else:
            self.rest_mlp = nn.Identity()
            rest_out = 0

        combined_dim = self.global_feat_dim + rest_out

        # --- Build the rl-games actor_critic network via builder ---
        from rl_games.algos_torch import network_builder as nb  # type: ignore
        factory = nb.NetworkBuilderFactory({'name': 'actor_critic', **kwargs})
        self._builder = factory.build_builder()
        # Build the default A2C model with our combined input size
        self.base = self._builder.build('a2c', input_shape=[combined_dim], actions_num=actions_num)

    def forward(self, input_dict):
        # Expect a flat observation
        obs = input_dict['obs']
        b = obs.size(0)

        pc_dim = self.pc_points * self.pc_channels
        pc_flat = obs[:, :pc_dim] if pc_dim > 0 else None
        rest = obs[:, pc_dim:] if self.rest_dim > 0 else None

        if pc_flat is not None and pc_dim > 0:
            # (B, P*C) -> (B, P, C) -> (B, C, P)
            x = pc_flat.view(b, self.pc_points, self.pc_channels).transpose(1, 2)
            pc_latent = self.pointnet(x)
        else:
            pc_latent = torch.zeros((b, self.global_feat_dim), device=obs.device, dtype=obs.dtype)

        if self.rest_dim > 0:
            rest_latent = self.rest_mlp(rest)
            h = torch.cat([pc_latent, rest_latent], dim=1)
        else:
            h = pc_latent

        # Delegate to rl-games built model (returns dict with 'actions','values','log_std')
        feed = {'obs': h}
        # Pass through any optional keys rl-games might provide
        for k in ('is_train', 'prev_actions', 'rnn_states'):
            if k in input_dict:
                feed[k] = input_dict[k]
        return self.base(feed)