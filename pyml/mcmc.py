import numpy as np
import copy

def defaultPosterior(loglik, logprior):
    return loglik + logprior

class Operator:
    def __init__(self, name, param, propose_func: callable, weight=1.0):
        self.name = name
        self.param = param
        self.weight = weight
        self.propose_func = propose_func
    
    def set_param(self, param):
        self.param = param

class Prior:
    def __init__(self, name, param, prior_func:callable):
        self.name = name
        self.param = param
        self.prior_func = prior_func
    
    @property
    def log_prior(self):
        return np.log(self.prior_func(self.param))

class MetropolisHastings:
    def __init__(self, 
                 likelihoodCalculator, 
                 parameters:list, 
                 operators:list, 
                 priors:list,
                 trace:list,
                 posterior_func:callable=defaultPosterior):
        self.likelihoodCalculator = likelihoodCalculator
        self.operators = operators
        self.priors = priors
        self.parameters = parameters
        self.posterior_func = posterior_func
        self.posterior = None
        self.trace = trace
        self.init_posterior()
    
    def init_posterior(self):
        loglik = self.likelihoodCalculator.prune(self.parameters)
        logprior = sum(p.log_prior for p in self.priors)
        self.posterior = self.posterior_func(loglik, logprior)
    
    def run(self):
        # a single MCMC sampling

        # propose an operator to modify the current state
        operator = np.random.choice(self.operators, p=[op.weight for op in self.operators])
        params = copy.deepcopy(operator.param)
        operator.propose_func(self.parameters, operator.param)
        # calculate the likelihood and prior of the proposed state
        loglik = self.likelihoodCalculator.prune(self.parameters)
        logprior = sum(p.log_prior for p in self.priors)
        logpost = self.posterior_func(loglik, logprior)

        # calculate the acceptance ratio
        log_accept_ratio = logpost - self.posterior

        # accept or reject the proposed state
        if np.log(np.random.rand()) < log_accept_ratio:
            self.posterior = logpost  # accept the new state
        else:
            # reject the new state, revert the parameters
            operator.set_param(params)
    
    def print_state_title(self):
        names = [ele.name for ele in self.trace]
        print('\t'.join(names))

    def print_state(self):
        print('\t'.join(ele.value) for ele in self.trace)