import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (
            grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(
            spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FastKANClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 28, 28),
        num_classes: int = 9,
        kan_hidden: List[int] = [128, 64],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 32,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]

        self.input_fc = nn.Linear(self.input_dim, kan_hidden[0])

        self.kan = FastKAN(
            layers_hidden=kan_hidden + [num_classes],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )

    def forward(self, x):
        x = self.flatten(x)     # (B, 3*28*28)
        x = self.input_fc(x)    # (B, kan_hidden[0])
        x = self.kan(x)         # (B, num_classes)
        return x
    

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 28, 28),
        num_classes: int = 9,
        hidden_layers: List[int] = [128, 64],
        activation_fn=nn.ReLU
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]

        layers = []
        prev_dim = self.input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.model(x)
