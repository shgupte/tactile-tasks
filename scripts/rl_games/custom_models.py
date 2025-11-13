import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import model_builder as mb

# PointNet encoder (user-provided)
from tactile_tasks.tasks.manager_based.tactile_tasks.pointnet import PointNet


class PointNetACBuilder(NetworkBuilder):
    """
    A NetworkBuilder that prepends a PointNet encoder to an A2C-style MLP.

    Assumes observations are flat unless use_central_value produces dicts.
    Layout (flat): first pc_points * pc_channels entries encode a point cloud (row-major),
    remaining entries are concatenated "rest" features.

    Returns (mu, logstd, value, states) for use with ModelA2CContinuous(LogStd).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    def load(self, params):
        # Store rl-games network params from YAML
        self.params = params
        return self

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            value_size = kwargs.pop("value_size", 1)

            # IMPORTANT: pass params to BaseNetwork so initializers/activations are configured
            super().__init__()

            # Parse YAML params
            mlp_cfg = params.get("mlp", {}) or {}
            self.units = list(mlp_cfg.get("units", [256, 256]))
            self.activation = mlp_cfg.get("activation", "relu")
            self.value_activation = params.get("value_activation", "None")

            # Space config (continuous)
            self.has_space = "space" in params
            if self.has_space and "continuous" in params["space"]:
                self.space_config = params["space"]["continuous"]
            else:
                self.space_config = {"mu_activation": "None"}
            # Note: if algorithm uses fixed_sigma: True, the framework may ignore network logstd.

            # PointNet configuration
            pc_cfg = params.get("pointnet", {}) or {}
            self.pc_points = int(pc_cfg.get("pc_points", 4096))
            self.pc_channels = int(pc_cfg.get("pc_channels", 3))
            self.global_feat_dim = int(pc_cfg.get("global_feat_dim", 1024))
            self.use_feature_transform = bool(pc_cfg.get("use_feature_transform", True))

            # Observation shape checks (expect flat shape [N])
            assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 1, \
                "Expected flat observation; provide a wrapper to flatten dict/stacked obs"
            obs_dim_total = int(input_shape[0])
            pc_dim = self.pc_points * self.pc_channels
            assert obs_dim_total >= pc_dim, \
                f"Observation smaller than point cloud slice: obs_dim={obs_dim_total}, pc_dim={pc_dim}"
            self.rest_dim = obs_dim_total - pc_dim

            # Modules
            self.pointnet = PointNet(
                in_channels=self.pc_channels,
                global_feat_dim=self.global_feat_dim,
                use_feature_transform=self.use_feature_transform,
            )

            # Optional MLP for rest features (identity by default)
            self.rest_mlp = nn.Identity()

            combined_dim = self.global_feat_dim + (self.rest_dim if self.rest_dim > 0 else 0)

            # Actor MLP trunk
            self.actor_mlp = self._build_mlp(
                input_size=combined_dim,
                units=self.units,
                activation=self.activation,
                dense_func=nn.Linear,
                norm_only_first_layer=mlp_cfg.get("norm_only_first_layer", False),
                norm_func_name=mlp_cfg.get("norm_func_name", None),
                d2rl=mlp_cfg.get("d2rl", False),
            )

            # Output sizes
            out_size = combined_dim if len(self.units) == 0 else self.units[-1]

            # Policy head
            self.mu = nn.Linear(out_size, actions_num)
            # Trainable logstd parameter (per action); cast to obs dtype/device in forward
            self.logstd = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32), requires_grad=True)

            # Value head
            self.value = self._build_value_layer(out_size, value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.mu_act = self.activations_factory.create(self.space_config.get("mu_activation", "None"))

        def _extract_obs_tensor(self, input_dict):
            """
            rl-games may pass either a tensor under 'obs' or a dict with keys {'obs', 'state'}
            when use_central_value is enabled. This normalizes to a 2D tensor (B, N).
            """
            obs = input_dict.get("obs")
            if isinstance(obs, dict):
                # Prefer primary 'obs' key; adjust if your wrapper uses different keys
                obs = obs.get("obs", None)
                assert obs is not None, "Dict observation missing key 'obs'"
            assert torch.is_tensor(obs) and obs.dim() == 2, "Expected obs as (B, N) tensor"
            return obs

        def forward(self, input_dict):
            obs = self._extract_obs_tensor(input_dict)  # (B, N)
            batch_size = obs.size(0)

            pc_dim = self.pc_points * self.pc_channels
            assert obs.size(1) >= pc_dim, \
                f"Runtime obs size smaller than expected pc slice: got {obs.size(1)}, need >= {pc_dim}"

            # Split point cloud and rest
            pc_flat = obs[:, :pc_dim] if pc_dim > 0 else None
            rest = obs[:, pc_dim:] if self.rest_dim > 0 else None
            # Sanity: rest shape
            if self.rest_dim > 0:
                assert rest.size(1) == self.rest_dim, \
                    f"Rest feature length mismatch: got {rest.size(1)}, expected {self.rest_dim}"

            # PointNet expects (B, C, P)
            if pc_flat is not None and pc_dim > 0:
                x = pc_flat.view(batch_size, self.pc_points, self.pc_channels).transpose(1, 2)
                pc_latent = self.pointnet(x)
            else:
                pc_latent = torch.zeros((batch_size, self.global_feat_dim),
                                        device=obs.device, dtype=obs.dtype)

            # Concatenate with rest
            if self.rest_dim > 0:
                rest_latent = self.rest_mlp(rest)
                h = torch.cat([pc_latent, rest_latent], dim=1)
            else:
                h = pc_latent

            # Trunk
            h = self.actor_mlp(h)

            # Heads
            mu = self.mu_act(self.mu(h))
            # Cast/expand logstd on-the-fly to match obs device/dtype
            logstd = self.logstd.to(dtype=obs.dtype, device=obs.device).unsqueeze(0).expand(batch_size, -1)
            value = self.value_act(self.value(h))

            # Non-recurrent model: states=None
            return mu, logstd, value, None

    def build(self, name, **kwargs):
        # name like 'a2c' is ignored; return our network
        return PointNetACBuilder.Network(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


# Register the builder under the name referenced in YAML:
# cfg:
#   model:
#     name: pointnet_ac_policy
#   space:
#     continuous:
#       mu_activation: None
#   algorithm (ensure fixed_sigma: False if you want learnable logstd)
mb.register_network('pointnet_ac_policy', PointNetACBuilder)
