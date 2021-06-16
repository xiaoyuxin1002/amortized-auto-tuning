import numpy as np
from collections import defaultdict
from itertools import combinations_with_replacement

import torch
import torch.nn as nn

from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, PolynomialKernel
from gpytorch.constraints import Positive, GreaterThan
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution


class MyGP(ApproximateGP):
    
    def __init__(self, info, meta, all_observations, if_observed, inducing_points):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.shape[0])
        variational_strategy = MyVariationalStrategy(meta, self, inducing_points, variational_distribution, True)
        super(MyGP, self).__init__(variational_strategy)
        
        self.info = info
        self.meta = meta
        
        self.mean_module = nn.ModuleList([ConstantMean() for _ in range(len(meta.TASK2ID))])
        self.covar_module_task = MyTaskKernel(info, meta, all_observations, if_observed)
        self.covar_module_hyper = MyHyperKernel(meta)
        self.covar_module_epoch = MyEpochKernel(meta)
        
        
    def forward(self, X):
        
        self.group(X)
        
        mean_X = torch.zeros(X.shape[0]).to(self.info.DEVICE)
        for task_id, group in self.groupby_task.items():
            mean_X[group] = self.mean_module[task_id](X[group, self.meta.INFO2ID['task']])
            
        covar_X_task = self.covar_module_task(X[:, self.meta.INFO2ID['task']])
        covar_X_hyper = self.covar_module_hyper(X)
        covar_X_epoch = self.covar_module_epoch(X[:, self.meta.INFO2ID['epoch']])
        covar_X = covar_X_epoch.mul(covar_X_task).mul(covar_X_hyper)
        
        return MultivariateNormal(mean_X, covar_X)
    
    
    def group(self, X):

        self.groupby_task = defaultdict(list)
        for item_idx, x in enumerate(X):
            self.groupby_task[int(x[self.meta.INFO2ID['task']].item())].append(item_idx)

        for task_id in self.groupby_task:
            self.groupby_task[task_id] = torch.Tensor(self.groupby_task[task_id]).long()

        self.covar_module_task.groupby_task = self.groupby_task
        

class MyVariationalStrategy(VariationalStrategy):
    
    def __init__(self, meta, model, inducing_points, variational_distribution, learn_inducing_locations):
        super(MyVariationalStrategy, self).__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        
        self.constrained_dim = meta.INFO2ID['epoch']
        self.unconstrained_dim = [dim for dim in range(inducing_points.shape[1]) if dim != self.constrained_dim]
        
        del self.inducing_points
        self.register_parameter(name="raw_inducing_points", parameter=nn.Parameter(torch.zeros(inducing_points.shape)))
        self.register_constraint("raw_inducing_points", Positive())
        self.inducing_points = inducing_points.clone()
        
    @property
    def inducing_points(self):
        
        inducing_points = torch.zeros_like(self.raw_inducing_points)
        inducing_points[:, self.unconstrained_dim] = self.raw_inducing_points[:, self.unconstrained_dim]        
        inducing_points[:, self.constrained_dim] = self.raw_inducing_points_constraint.transform(self.raw_inducing_points[:, self.constrained_dim])
        return inducing_points
                
    @inducing_points.setter
    def inducing_points(self, value):
        
        raw_value = torch.zeros_like(value)
        raw_value[:, self.unconstrained_dim] = value[:, self.unconstrained_dim]
        raw_value[:, self.constrained_dim] = self.raw_inducing_points_constraint.inverse_transform(value[:, self.constrained_dim])
        self.initialize(raw_inducing_points=raw_value)
        
        
class MyTaskKernel(Kernel):
    
    def __init__(self, info, meta, all_observations, if_observed):
        super(MyTaskKernel, self).__init__()
        self.register_parameter(name="raw_lengthscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_lengthscale", Positive())
        self.register_parameter(name="raw_outputscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", Positive())
        
        self.info = info
        self.meta = meta
        self.all_observations = all_observations
        self.if_observed = if_observed
                
        self.gammas = {}
        self.distances = {}
        self.groupby_task = None

        self.register_parameter(name="raw_U", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_U", GreaterThan(1))
        self.U = 1.5
        
    @property
    def U(self):
        
        return self.raw_U_constraint.transform(self.raw_U)

    @U.setter
    def U(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_U)
        self.initialize(raw_U=self.raw_U_constraint.inverse_transform(value))
        
    @property
    def lengthscale(self):
        
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)    

    @lengthscale.setter
    def lengthscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))
        
    @property
    def outputscale(self):
        
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @outputscale.setter
    def outputscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
        
    def get_gamma(self, num_matched):
        
        ratio = num_matched / (self.meta.NUM_BASE_CONFIG * self.meta.NUM_EPOCH)
        gamma = self.U / (1 + (self.U-1) * ratio)
        return gamma
        
    def get_distance(self, task_i, task_j, num_matched, matched_observations):
        
        if num_matched <= 1:
            return 0.5
        else:        
            matched_i = self.all_observations[task_i][matched_observations[0], matched_observations[1]]; matched_i -= np.mean(matched_i)
            matched_j = self.all_observations[task_j][matched_observations[0], matched_observations[1]]; matched_j -= np.mean(matched_j)
            return np.square(np.linalg.norm(matched_i - matched_j)) / num_matched
        
    def forward(self, X, _, **params):

        covar_X_task = torch.zeros((X.shape[0], X.shape[0])).to(self.info.DEVICE)

        for task_i, task_j in combinations_with_replacement(self.groupby_task, 2):
            matched_observations = np.where(self.if_observed[task_i] & self.if_observed[task_j])
            num_matched = len(matched_observations[0])

            gamma = self.get_gamma(num_matched)
            if (task_i, task_j) not in self.gammas or gamma != self.gammas[(task_i, task_j)]:
                self.distances[(task_i, task_j)] = self.get_distance(task_i, task_j, num_matched, matched_observations) if task_i != task_j else 0
            self.gammas[(task_i, task_j)] = gamma
            covar_task_ij = self.outputscale * torch.exp(- self.distances[(task_i, task_j)] / (self.lengthscale * self.gammas[(task_i, task_j)])**2)
    
            group_i, group_j = self.groupby_task[task_i], self.groupby_task[task_j]
            covar_X_task[group_i.unsqueeze(-1), group_j] = covar_task_ij            
            if task_i != task_j: covar_X_task[group_j.unsqueeze(-1), group_i] = covar_task_ij
        
        return covar_X_task
    
    
class MyHyperKernel(Kernel):
    
    def __init__(self, meta):
        super(MyHyperKernel, self).__init__()
        
        self.meta = meta
        
        self.numeric_indices = [iid for info, iid in self.meta.INFO2ID.items() if info not in self.meta.CONDITION2ID and info != 'task' and info != 'epoch']
        dim = len(self.numeric_indices) + len([BRANCH for CONDI, BRANCHES in self.meta.NESTED2ID.items() for BRANCH in BRANCHES])
                
        self.categorical_embeddings = nn.ModuleList([nn.Embedding(len(BRANCHES), len(BRANCHES)) for CONDI, BRANCHES in self.meta.NESTED2ID.items()])
        self.encoder = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.covar_module_hyper = PolynomialKernel(power=2)
        
    def forward(self, X, _, **params):

        inputs = X[:, self.numeric_indices]
        for i, CONDI in enumerate(self.meta.NESTED2ID):
            inputs = torch.hstack([inputs, self.categorical_embeddings[i](X[:, CONDI].long())])
        outputs = self.encoder(inputs)
        covar_X_hyper = self.covar_module_hyper(outputs)
        
        return covar_X_hyper
    
    
class MyEpochKernel(Kernel):
    
    def __init__(self, meta):
        super(MyEpochKernel, self).__init__()

        self.meta = meta
        
        self.register_parameter(name="raw_alpha", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_alpha", Positive())
        self.register_parameter(name="raw_beta", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_beta", Positive())
        self.register_parameter(name="raw_outputscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", Positive())
        
    @property
    def alpha(self):
        
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
        
    @property
    def beta(self):
        
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))
        
    @property
    def outputscale(self):
        
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @outputscale.setter
    def outputscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
        
    def forward(self, X, _, **params):

        X = X.expand(X.shape[0], -1)
        covar_X_epoch = 1 + (self.beta / (X + X.T + self.beta)).pow(self.alpha) - (self.beta / (X + self.beta)).pow(self.alpha) - (self.beta / (X.T + self.beta)).pow(self.alpha)
        covar_X_epoch *= self.outputscale

        return covar_X_epoch