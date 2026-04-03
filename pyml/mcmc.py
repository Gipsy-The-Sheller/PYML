import numpy as np
import copy
import random

def defaultPosterior(loglik, logprior):
    return loglik + logprior

class Operator:
    def __init__(self, name, param, propose_func: callable, weight=1.0):
        self.name = name
        self.param = param
        self.weight = weight
        self.propose_func = propose_func

    def resolve_param_key(self, parameters: dict):
        # If param is a string, treat it as the parameter key
        if isinstance(self.param, str):
            return self.param
        # Otherwise try to find a key in parameters whose value matches self.param
        for k, v in parameters.items():
            if v is self.param or v == self.param:
                return k
        # fallback: return original param (operator may handle non-key params)
        return self.param

    def set_param(self, param):
        self.param = param


class Prior:
    def __init__(self, name, param, prior_func:callable):
        # param is expected to be a parameter key (str) or a callable returning current value
        self.name = name
        self.param = param
        self.prior_func = prior_func

    def log_prior(self, parameters: dict):
        # resolve current parameter value
        if isinstance(self.param, str):
            val = parameters[self.param]
        elif callable(self.param):
            val = self.param()
        else:
            val = self.param
        p = self.prior_func(val)
        # guard against zero
        p = p if p > 0 else 1e-300
        return np.log(p)


class MetropolisHastings:
    def __init__(self, 
                 likelihoodCalculator, 
                 parameters:dict, 
                 operators:list, 
                 priors:list,
                 trace:list = None,
                 posterior_func:callable=defaultPosterior):
        self.likelihoodCalculator = likelihoodCalculator
        self.operators = operators
        self.priors = priors
        self.parameters = parameters
        self.posterior_func = posterior_func
        self.posterior = None
        self.trace = trace or []
        self.init_posterior()

    def init_posterior(self):
        loglik = self.likelihoodCalculator.log_likelihood(self.parameters)
        logprior = sum(p.log_prior(self.parameters) for p in self.priors)
        self.current_loglik = loglik
        self.current_logprior = logprior
        self.posterior = self.posterior_func(loglik, logprior)
        self.current_posterior = self.posterior

    def run(self):
        # single MCMC iteration: propose and accept/reject
        operator = random.choices(self.operators, weights=[op.weight for op in self.operators], k=1)[0]

        # snapshot current state (parameters and phyloData)
        params_snapshot = copy.deepcopy(self.parameters)
        phylo_snapshot = copy.deepcopy(self.likelihoodCalculator.phyloData)

        # resolve param identifier passed to propose_func
        op_param = operator.resolve_param_key(self.parameters)
        # apply proposal (operator modifies self.parameters and/or phyloData in-place)
        operator.propose_func(self.parameters, op_param)

        # compute posterior for proposed state
        loglik = self.likelihoodCalculator.log_likelihood(self.parameters)
        logprior = sum(p.log_prior(self.parameters) for p in self.priors)
        logpost = self.posterior_func(loglik, logprior)

        # store current proposed values for external access
        self.current_loglik = loglik
        self.current_logprior = logprior
        self.current_posterior = logpost

        log_accept_ratio = logpost - self.posterior

        if np.log(np.random.rand()) < log_accept_ratio:
            # accept
            self.posterior = logpost
        else:
            # reject: restore snapshots
            self.parameters.clear()
            self.parameters.update(params_snapshot)
            self.likelihoodCalculator.phyloData = phylo_snapshot

    def print_state_title(self):
        names = []
        for ele in self.trace:
            if isinstance(ele, str):
                names.append(ele)
            elif hasattr(ele, 'name'):
                names.append(ele.name)
            else:
                names.append(str(ele))
        print('\t'.join(names))

    def print_state(self):
        values = []
        for ele in self.trace:
            if isinstance(ele, str):
                # special keywords
                if ele == 'loglik':
                    values.append(str(getattr(self, 'current_loglik', '')))
                    continue
                if ele == 'logprior':
                    values.append(str(getattr(self, 'current_logprior', '')))
                    continue
                if ele == 'posterior':
                    values.append(str(getattr(self, 'current_posterior', '')))
                    continue

                val = self.parameters.get(ele)
                if callable(val):
                    values.append(str(val()))
                else:
                    values.append(str(val))
            elif callable(ele):
                try:
                    # allow callable that takes no args or takes parameters dict
                    values.append(str(ele()))
                except TypeError:
                    values.append(str(ele(self.parameters)))
            else:
                values.append(str(ele))
        print('\t'.join(values))

    def savetracelist(self):
        """Return a serializable list/dict of current trace values."""
        out = {}
        for ele in self.trace:
            if isinstance(ele, str):
                # special keywords
                if ele == 'loglik':
                    out['loglik'] = getattr(self, 'current_loglik', None)
                    continue
                if ele == 'logprior':
                    out['logprior'] = getattr(self, 'current_logprior', None)
                    continue
                if ele == 'posterior':
                    out['posterior'] = getattr(self, 'current_posterior', None)
                    continue

                val = self.parameters.get(ele)
                out[ele] = val() if callable(val) else val
            elif isinstance(ele, tuple) and len(ele) == 2:
                name, func = ele
                out[name] = func() if callable(func) else func
            elif callable(ele):
                # no name available, use stringified callable
                try:
                    out[getattr(ele, '__name__', str(ele))] = ele()
                except TypeError:
                    out[getattr(ele, '__name__', str(ele))] = ele(self.parameters)
            else:
                out[str(ele)] = ele
        return out

    def savetreelist(self):
        """Return a lightweight representation of the current tree (branch lengths list)."""
        tree = self.likelihoodCalculator.phyloData.tree
        brlens = [getattr(n, 'brlength', None) or getattr(n, 'branch_length', None) for n in tree.nodes if not n.is_root]
        return brlens