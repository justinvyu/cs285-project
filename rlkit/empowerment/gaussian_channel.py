import numpy as np
# import rlkit.misc.pytorch_util as ptu
import rlkit.torch.pytorch_util as ptu
import torch
from torch import nn
import torch.optim as optim

import cvxpy as cp
import multiprocessing

def water_filling(sing_vals, sum_p=1.0):
    p = cp.Variable(len(sing_vals))
    obj_fn = 0.5 * cp.sum(cp.log(1 + cp.multiply(sing_vals, p)))
    obj = cp.Maximize(obj_fn)
    constraints = [p >= 0, cp.sum(p) == sum_p]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return prob.value

class GaussianChannelModel(nn.Module):
    def __init__(
            self,
            obs_dim,
            ac_dim,
            N,
            learning_rate=3e-4,
            l2_lambda=0,
            **kwargs
    ):
        super().__init__()
        self.G_b_model = ptu.build_mlp(
            obs_dim,
            obs_dim * (N * ac_dim) + obs_dim,
            n_layers=3,
            size=512,
            activation='relu',
        )
        self.N = N
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.optimizer = optim.Adam(
            self.G_b_model.parameters(),
            self.learning_rate,
            weight_decay=l2_lambda,
        )

    def forward(self, s_t, action_sequence):
        batch_size = s_t.shape[0]

        if isinstance(s_t, np.ndarray):
            s_t = ptu.from_numpy(s_t, to_device=self.device)
        if isinstance(action_sequence, np.ndarray):
            action_sequence = ptu.from_numpy(action_sequence, to_device=self.device)

        G_b = self.G_b_model(s_t)
        G, b = G_b[:, :-self.obs_dim], G_b[:, -self.obs_dim:]
        G_matrix = G.view(batch_size, self.obs_dim, self.N * self.ac_dim)
        a_vector = action_sequence.view(batch_size, self.N * self.ac_dim, 1)

        # Batched matrix multiplication
        output = torch.matmul(G_matrix, a_vector).squeeze() + b

        return output, G_matrix.detach()

    def G(self, s_t, to_numpy=False):
        batch_size = s_t.shape[0]

        if isinstance(s_t, np.ndarray):
            s_t = ptu.from_numpy(s_t, to_device=self.device)

        G_b = self.G_b_model(s_t)
        G, b = G_b[:, :-self.obs_dim], G_b[:, -self.obs_dim:]
        G_matrix = G.view(batch_size, self.obs_dim, self.N * self.ac_dim)

        if to_numpy:
            return ptu.to_numpy(G_matrix)
        else:
            return G_matrix

    @property
    def device(self):
        return next(self.parameters()).device

    def update(self, s_t, action_sequences, s_T):
        if isinstance(s_T, np.ndarray):
            s_T = ptu.from_numpy(s_T, to_device=self.device)

        pred_s_T, G_matrix = self(s_t, action_sequences)
        loss = self.loss(pred_s_T, s_T)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def empowerment(self, s_t, parallelize=True, max_workers=16):
        G_matrices = self.G(s_t)
        singular_values = ptu.to_numpy(torch.svd(G_matrices).S).squeeze()
        if parallelize:
            with multiprocessing.Pool(max_workers) as pool:
                empowerment_vals = pool.map(water_filling, singular_values)
        else:
            empowerment_vals = [
                water_filling(sing_vals) for sing_vals in singular_values
            ]

        empowerment_vals = np.array(empowerment_vals)
        return empowerment_vals
