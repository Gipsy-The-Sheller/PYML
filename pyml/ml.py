from .likelihood import LikelihoodCalculator
import numpy as np

class BrLenOptimizer:
    def __init__(self, likelihood_calculator, epsilon=1e-5, min_branch_length=1e-6):
        self.calc = likelihood_calculator
        self.epsilon = epsilon
        self.min_branch_length = min_branch_length
    
    def _get_1d_loglik(self, params_dict, target_branch_key, t):
        temp_params = params_dict.copy()
        temp_params[target_branch_key] = t

        # compute local likelihood
        return self.calc.log_likelihood(temp_params)

    def optimize_single_branch(self, params_dict, target_branch_key, max_iter=10, tol=1e-4):
        t = params_dict[target_branch_key]
        h = self.epsilon

        for i in range(max_iter):
            L_curr = self._get_1d_loglik(params_dict, target_branch_key, t)
            L_plus = self._get_1d_loglik(params_dict, target_branch_key, t + h)
            L_minus = self._get_1d_loglik(params_dict, target_branch_key, t - h)

            grad = (L_plus - L_minus) / (2 * h)
            hess = (L_plus - 2 * L_curr + L_minus) / (h ** 2)

            if hess >= 0:
                step = 0.01 * grad if grad > 0 else -0.01
            else:
                step = - grad / hess

            t_new = t + step
            t_new = max(self.min_branch_length, t_new)

            if abs(t_new - t) < tol:
                t = t_new
                break
            
            t = t_new

        params_dict[target_branch_key] = t
        return params_dict
    
    def optimize_all_branches(self, params_dict, sweeps=3, verbose=0):
        for sweep in range(sweeps):
            if verbose in [2]:
                print(f"Starting global optimization sweep {sweep+1}...")
            for branch_key in params_dict.keys():
                params_dict = self.optimize_single_branch(params_dict, branch_key)
            
            current_loglik = self.calc.log_likelihood(params_dict)
            if verbose in [1, 2]:
                print(f"After sweep {sweep+1}, current Log-Likelihood: {current_loglik:.4f}")
        
        return params_dict