import numpy as np
import scipy.stats as stats
import torch

from .optimizer import Optimizer


class CEMOptimizer(Optimizer):
    def __init__(self, sol_dim, popsize, upper_bound=None, lower_bound=None, max_iters=10, num_elites=100, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.mean, self.var = None, None
        self.cost_function = None

    def setup(self, cost_function):
        self.cost_function = cost_function

        def sample_truncated_normal(shape, mu, sigma, a, b):
            uniform = torch.rand(shape)
            normal = torch.distributions.normal.Normal(0, 1)

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            alpha_normal_cdf = normal.cdf(alpha)
            p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

            p = p.numpy()
            one = np.array(1, dtype=p.dtype)
            epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
            v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
            x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
            x = torch.clamp(x, a[0], b[0])
            return x
        self.sample_trunc_norm = sample_truncated_normal

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, use_pytorch=False, debug=False):
        """
        Optimizes the cost function using the provided initial candidate distribution parameters

        Parameters:
        ----------
            @param numpy array - init_mean, init_var: size should be (popsize x sol_dim)
            @param bool - use_pytorch: determine if use pytorch implementation
            @param bool - debug: if true, it will save some figures to help you find the best parameters

        Return:
        ----------
            @param numpy array - sol : size should be (sol_dim)
        """

        mean, var, t = init_mean, init_var, 0

        if use_pytorch:
            a, b = torch.tensor([self.lb]*self.sol_dim), torch.tensor([self.ub]*self.sol_dim)
            size = [self.popsize, self.sol_dim]
        else:
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        if debug:
            cost_list = []
            mean_list = []
            var_list = []

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            if use_pytorch:
                mu = torch.tensor(mean)
                sigma = torch.tensor(np.sqrt(var))
                samples = self.sample_trunc_norm(size, mu, sigma, a, b).numpy()
            else:
                samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
                samples = samples.astype(np.float32)

            costs = self.cost_function(samples)
            idx = np.argsort(costs)
            elites = samples[idx][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            if debug:
                min_cost = costs[idx][:self.num_elites]
                cost_list.append(np.mean(min_cost))
                mean_list.append(np.mean(new_mean[0]))
                var_list.append(np.mean(new_var))

            t += 1
            sol, solvar = mean, var
        return sol, solvar
