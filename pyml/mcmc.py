import numpy as np
import copy
import random
from dataclasses import dataclass

@dataclass
class MCMCParam:
    name: str
    value: any

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
    
    def update_posterior(self):
        # compute posterior for proposed state
        loglik = self.likelihoodCalculator.log_likelihood(self.parameters)
        logprior = sum(p.log_prior(self.parameters) for p in self.priors)
        logpost = self.posterior_func(loglik, logprior)

        self.current_loglik = loglik
        self.current_logprior = logprior
        self.current_posterior = logpost

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

        self.update_posterior()

        # store current proposed values for external access


        log_accept_ratio = self.current_posterior - self.posterior

        if np.log(np.random.rand()) < log_accept_ratio:
            # accept
            self.posterior = self.current_posterior
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

class MCMCMC: 
    def posterior_func(self, loglik, logprior, temperature):
        return loglik / temperature + logprior

    def __init__(self, 
                 likelihoodCalculator, 
                 parameters:dict, 
                 operators:list, 
                 priors:list,
                 trace:list = None,
                 chain_temperatures:list = [1.0],):
        # check temperatures (must at least 1 is 1.0 for the cold chain)
        if 1.0 not in chain_temperatures:
            raise ValueError("Chain temperatures must include 1.0 for the cold chain")
        self.chain_temperatures = chain_temperatures
        self.chains = [MetropolisHastings(
            likelihoodCalculator, 
            copy.deepcopy(parameters), 
            operators, 
            priors, 
            trace, 
            lambda x, y: self.posterior_func(x, y, temp)) 
            for temp in chain_temperatures]
    
    def run(self):

        for chain in self.chains:
            chain.run()
        
        to_swap = random.sample(range(len(self.chains)), 2)
        energies = [-chain.current_posterior for chain in self.chains]
        temperatures = self.chain_temperatures

        swap_accept_prob = min(1, np.exp((energies[to_swap[0]] - energies[to_swap[1]]) * (1/temperatures[to_swap[0]] - 1/temperatures[to_swap[1]])))

        if np.random.rand() < swap_accept_prob:
            # swap states between the two chains
            # NOTE: owing to that we want to visualize the swap of 2 chains (like MrBayes), we do not swap current states, but temperature
            # Therefore, the codes below should not be used:
            # self.chains[to_swap[0]].parameters, self.chains[to_swap[1]].parameters = self.chains[to_swap[1]].parameters, self.chains[to_swap[0]].parameters
            # self.chains[to_swap[0]].likelihoodCalculator.phyloData, self.chains[to_swap[1]].likelihoodCalculator.phyloData = self.chains[to_swap[1]].likelihoodCalculator.phyloData, self.chains[to_swap[0]].likelihoodCalculator.phyloData
            
            
            # after swapping, update posteriors for both chains
            self.chain_temperatures[to_swap[0]], self.chain_temperatures[to_swap[1]] = self.chain_temperatures[to_swap[1]], self.chain_temperatures[to_swap[0]]
            self.chains[to_swap[0]].posterior_func = lambda x, y: self.posterior_func(x, y, self.chain_temperatures[to_swap[0]])
            self.chains[to_swap[1]].posterior_func = lambda x, y: self.posterior_func(x, y, self.chain_temperatures[to_swap[1]])
            self.chains[to_swap[0]].update_posterior()
            self.chains[to_swap[1]].update_posterior()

    def print_states(self):
        print(self.states_log())

    def states_log(self):
        """
        single-line log resembling MrBayes output
        """
        posteriors = [chain.current_posterior for chain in self.chains]
        log = ''
        for i in range(len(self.chains)):
            if self.chain_temperatures[i] == 1.0:
                log = f"{log}[{posteriors[i]:.2f}]\t"  # highlight cold chain
            else:
                log = f"{log}({posteriors[i]:.2f})\t"
        
        return log
    
    def savetracelist(self):
        """Return the trace list from the cold chain"""
        cold_chain_index = self.chain_temperatures.index(1.0)
        return self.chains[cold_chain_index].savetracelist()
    
    def savetreelist(self):
        """Return the tree list from the cold chain"""
        cold_chain_index = self.chain_temperatures.index(1.0)
        return self.chains[cold_chain_index].savetreelist()

class SS(MCMCMC):
    """
    Stepping-stone sampler to estimate marginal likelihood
    """
    def __init__(self, 
                 likelihoodCalculator, 
                 parameters:dict, 
                 operators:list, 
                 priors:list,
                 trace:list = None,
                 temp_grids = 10):
        
        """
        temp_grids: number od divisions between 0 and 1 for beta (beta = 1 / T)
        """
        self.chain_temperatures = [i / temp_grids for i in range(temp_grids+1)]
        super().__init__(likelihoodCalculator, parameters, operators, priors, trace, self.chain_temperatures)
        self.logliks = [[] for _ in self.chain_temperatures]  # to store posterior samples for each chain

    def sample_ss(self):
        # obtain log-likelihood samples from each chain (required for stepping-stone)
        for index, chain in enumerate(self.chains):
            self.logliks[index].append(chain.current_loglik)

    def estimate_marginal_likelihood(self):
        """
        Estimate log marginal likelihood via the stepping-stone method.

        Returns an estimate of log Z = log Z(1) - log Z(beta_min), where beta=1/T.
        Requires that `self.logliks[k]` contains an array/list of log-likelihood
        samples collected from the chain at temperature `self.chain_temperatures[k]`.
        """
        # collect (beta, samples) pairs and filter empty sample sets
        pairs = []
        for i, T in enumerate(self.chain_temperatures):
            beta = 1.0 / T
            arr = np.asarray(self.logliks[i], dtype=float)
            if arr.size > 0:
                pairs.append((beta, arr))

        if len(pairs) < 2:
            raise ValueError("Need samples from at least two temperatures to estimate marginal likelihood")

        # sort by beta ascending (from 0 -> 1)
        pairs.sort(key=lambda x: x[0])
        betas = [p[0] for p in pairs]
        samples = [p[1] for p in pairs]

        def log_mean_exp(a):
            m = np.max(a)
            return m + np.log(np.mean(np.exp(a - m)))

        total = 0.0
        # stepping-stone product: sum_k log E_{beta_k}[ L^{delta_beta} ]
        for k in range(len(betas) - 1):
            delta = betas[k+1] - betas[k]
            vals = delta * samples[k]
            log_ratio = log_mean_exp(vals)
            total += log_ratio

        if betas[0] > 0:
            print(f"Warning: smallest beta={betas[0]:.4g} > 0; estimate is logZ(1)-logZ({betas[0]:.4g}). Consider including beta=0 for full marginal likelihood.")

        return total
    
    @property
    def temperature_gradients(self):
        return sorted(self.chain_temperatures)
