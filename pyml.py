# PYML: PYthon framework for phylogenetic Maximun Likelihood analyses 
# Salute: PAML (Phylogenetic Analysis by Maximum Likelihood) by Ziheng Yang.
# I want a more flexible framework (like RevBayes) directly available in Jupyter notebooks,
# and more flexible model supports, not only BaseML and CodeML.
# Also I want to know if Fortran and Numpy parallelization can catch up with (part of) PLL / BEAGLE, and faster than PAML's single-thread ML calculator.


import numpy as np
import Bio
import copy


# core data structure

class PhyloData:
    def __init__(self, seqs: Bio.SeqRecord, nstates=None, tree=None, qmatrix=None):
        self.seqs = seqs
        self.nstates = nstates
        self.tree = tree
        self.qmatrix = qmatrix
        if self.qmatrix is None:
            # use module default JC if not provided (created later)
            try:
                self.qmatrix = JC
            except NameError:
                self.qmatrix = None

        if self.nstates is None:
            self.nstates = len(set(str(seqs[0].seq)))  # infer nstates from the first sequence
    
# model parameters

class QMatrix:
    def __init__(self, nstates, attributes=None):
        self.nstates = nstates
        self.Rshape = np.zeros((nstates, nstates))
        self.Rmatrix = np.zeros((nstates, nstates))
        self.freqs = np.zeros(nstates)
        self.params = {}

        self.attributes = None
        if attributes is not None:
            self.set_attributes(attributes)

    def set_attributes(self, characters):
        self.attributes = characters

    def check_validity(self):
        # check if the Rmatrix is valid
        if self.Rmatrix.shape != (self.nstates, self.nstates):
            raise ValueError("Rmatrix must be a square matrix of size nstates x nstates")
        if np.any(self.Rmatrix < 0):
            raise ValueError("Rmatrix must have non-negative entries")
        if np.any(np.diag(self.Rmatrix) != 0):
            raise ValueError("Diagonal entries of Rmatrix must be zero")
        
        # check if the freqs are valid
        if self.freqs.shape != (self.nstates,):
            raise ValueError("freqs must be a vector of length nstates")
        if np.any(self.freqs < 0):
            raise ValueError("freqs must have non-negative entries")
        if not np.isclose(np.sum(self.freqs), 1.0):
            raise ValueError("freqs must sum to 1")
    
    def set_params(self, params):
        # example
        # RShape:
        [[-1, 'a', 'b', 'c']]
        [['a', -1, 'd', 'e']]
        [['b', 'd', -1, 'f']]
        [['c', 'e', 'f', -1]]

        # params:
        {{-1: -1, 'a': 0.1, 'b': 0.2, 'c': 0.7, 'd': 0.1, 'e': 0.2, 'f': 0.7},}

        # replace the symbolic entries in RShape with the corresponding values from params
        for i in range(self.nstates):   
            for j in range(self.nstates):
                if self.Rshape[i, j] in params:
                    self.Rmatrix[i, j] = params[self.Rshape[i, j]]
                else:
                    self.Rmatrix[i, j] = self.Rshape[i, j]

        # This design is to allow for flexible parameterization of the QMatrix (direct editing RMatrix) 
        # or empirical models (like JTT and WAG) with a fixed RMatrix.
        # && rate constraints for universal models (use Rshape).


# create simple default matrices after QMatrix is defined
JC, K2P, HKY, GTR = [QMatrix(4) for _ in range(4)]
JC.set_attributes(['A', 'C', 'G', 'T'])
JC.Rmatrix = np.array([[-1, 0.25, 0.25, 0.25],
                       [0.25, -1, 0.25, 0.25],
                       [0.25, 0.25, -1, 0.25],
                       [0.25, 0.25, 0.25, -1]])
JC.freqs = np.array([0.25, 0.25, 0.25, 0.25])

# binary tree topology structure

class Topology:
    def __init__(self, ntaxa):
        self.ntaxa = ntaxa

        # for a n_taxa rooted tree,
        # 3 -> 5, 4-> 7, n_taxa -> 2*n_taxa - 1
        self.nodes = [Node() for _ in range(2 * ntaxa - 1)]

class Node:
    def __init__(self, parents=None, children=None, **metadata):
        self.parents = parents
        self.children = children
        self.metadata = metadata

    def is_leaf(self):
        return self.children is None or len(self.children) == 0
    
    def is_root(self):
        return self.parents is None or len(self.parents) == 0

class ML_Estimator:
    def __init__(self, phylo_data):
        self.seqs = phylo_data.seqs
        self.qmatrix = phylo_data.qmatrix
        self.tree = phylo_data.tree
        self.nstates = phylo_data.nstates
        self.ntaxa = len(phylo_data.seqs)

        # cache for transition matrices by (id(qmatrix), t)
        self._P_cache = {}
        # prepare branch parameter indexing: assign an index to each non-root node's branch
        self.param_index = {}
        self.params = []
        # copy topology for indexing
        topo = copy.deepcopy(self.tree)
        pid = 0
        for node in topo.nodes:
            if not node.is_root():
                # branch leading to this node
                self.param_index[id(node)] = pid
                self.params.append(('branch', pid))
                pid += 1
        self.num_params = pid


        nodes = 2 * self.ntaxa - 1

        # initialize parallel numpy arrays for likelihood calculations
        # count: nodes
        # length: seq_length * nstates
        seq_length = len(phylo_data.seqs[0].seq)
        states = np.zeros((nodes, seq_length * self.nstates))


        # store one-hot encoding of the sequences at the leaves
        for i in range(self.ntaxa):
            seq = str(phylo_data.seqs[i].seq)
            for j, s in enumerate(seq):
                s_idx = self.state_to_index(s)
                states[i, j * self.nstates + s_idx] = 1.0
        
        # distribute remained arrays for phylogenetic tree internal nodes

        # 1. clone the phylogenetic tree topology
        self.topology = copy.deepcopy(phylo_data.tree)
        # 2. assign the states array to each nodes

        iter_node = 0
        for i in self.topology.nodes:
            if i.is_leaf():
                # assign the corresponding one-hot encoding
                seq_name = i.metadata['name']  # assuming the leaf node has a 'name' metadata that matches the sequence name
                seq_idx = self.get_sequence_index(seq_name)  # get the index of the sequence

                i.metadata['states'] = states[seq_idx]  # assign the one-hot encoding to the leaf node
            else:
                # assign an empty array for internal nodes (place after leaves)
                if iter_node == 0:
                    iter_node = self.ntaxa
                i.metadata['states'] = states[iter_node]  # assign the states array to the internal node
                iter_node += 1
        
        del iter_node

    def state_to_index(self, state):
        # map a single-character state to its index in the model
        if hasattr(self.qmatrix, 'attributes') and self.qmatrix.attributes:
            try:
                return self.qmatrix.attributes.index(state)
            except ValueError:
                pass

        if not hasattr(self, '_state_map'):
            chars = []
            for rec in self.seqs:
                chars.extend(list(str(rec.seq)))
            unique = sorted(set(chars))
            # ensure at least nstates entries (fallback to common bases)
            if len(unique) < self.nstates:
                for c in ['A', 'C', 'G', 'T']:
                    if c not in unique:
                        unique.append(c)
            self._state_map = {c: idx for idx, c in enumerate(unique[:self.nstates])}

        return self._state_map.get(state, 0)

    def get_sequence_index(self, seq_name):
        # find sequence index by common SeqRecord fields
        for idx, rec in enumerate(self.seqs):
            if hasattr(rec, 'id') and rec.id == seq_name:
                return idx
            if hasattr(rec, 'name') and rec.name == seq_name:
                return idx
            if hasattr(rec, 'description') and rec.description == seq_name:
                return idx
        # fallback: match by sequence string
        for idx, rec in enumerate(self.seqs):
            if str(rec.seq) == seq_name:
                return idx
        raise ValueError(f"Sequence name {seq_name} not found")

    def compute_likelihood(self):
        # compute likelihood using Felsenstein's pruning algorithm
        # Felsenstein (1985)

        for i in self.topology.nodes:
            if i.is_root():
                # compute the likelihood at the root node
                res = self.pruning(i.children[0], i.children[1])
                # res may be (L_root, log_scales) or (L_root, d1_root, d2_root, log_scales)
                if len(res) == 2:
                    L_root, log_scales = res
                    d1_root = None
                    d2_root = None
                else:
                    L_root, d1_root, d2_root, log_scales = res

                # compute per-site likelihoods by combining with stationary freqs
                pi = self.qmatrix.freqs
                if pi is None or len(pi) != self.nstates:
                    # fallback to uniform
                    pi = np.full(self.nstates, 1.0 / self.nstates)

                site_liks = (pi[:, None] * L_root).sum(axis=0)
                # protect zeros
                site_liks = np.where(site_liks <= 0, 1e-300, site_liks)
                log_sites = np.log(site_liks) + log_scales
                total_loglik = np.sum(log_sites)

                # if derivatives present, compute gradient and Hessian for branch lengths
                if d1_root is None:
                    return total_loglik

                P = self.num_params
                # gradient: g_p = sum_s (1/site_lik_s) * sum_i pi_i * d1_root[p,i,s]
                grads = np.zeros(P)
                for p in range(P):
                    num = (pi[:, None] * d1_root[p]).sum(axis=0)
                    grads[p] = np.sum(num / site_liks)

                # Hessian: H_{pq} = sum_s [ (sum_i pi_i d2_root[p,q,i,s]) / site_lik_s
                #                      - (sum_i pi_i d1_root[p,i,s])*(sum_j pi_j d1_root[q,j,s]) / site_lik_s^2 ]
                H = np.zeros((P, P))
                for p in range(P):
                    for q in range(P):
                        num2 = (pi[:, None] * d2_root[p, q]).sum(axis=0)
                        num1p = (pi[:, None] * d1_root[p]).sum(axis=0)
                        num1q = (pi[:, None] * d1_root[q]).sum(axis=0)
                        H[p, q] = np.sum(num2 / site_liks - (num1p * num1q) / (site_liks ** 2))

                return total_loglik, grads, H
    
    def pruning(self, node1, node2):
        """Vectorized pruning: returns (L_parent, log_scales).

        L_parent: ndarray (nstates, S)
        log_scales: ndarray (S,)
        """
        # helper to get leaf or recurse
        def get_L_d1_d2_scales(node):
            if node.is_leaf():
                arr = node.metadata['states']  # shape (seq_len * nstates,)
                seq_len = arr.shape[0] // self.nstates
                L = arr.reshape((seq_len, self.nstates)).T  # (nstates, S)
                scales = np.zeros(seq_len)
                # derivatives: d1 shape (P, n, S), d2 shape (P, P, n, S)
                P = getattr(self, 'num_params', 0)
                if P > 0:
                    d1 = np.zeros((P, self.nstates, L.shape[1]))
                    d2 = np.zeros((P, P, self.nstates, L.shape[1]))
                else:
                    d1 = np.zeros((0, self.nstates, L.shape[1]))
                    d2 = np.zeros((0, 0, self.nstates, L.shape[1]))
                return L, d1, d2, scales
            else:
                return self.pruning(node.children[0], node.children[1])

        L1, d1_1, d2_1, scales1 = get_L_d1_d2_scales(node1)
        L2, d1_2, d2_2, scales2 = get_L_d1_d2_scales(node2)

        # ensure same S
        if L1.shape[1] != L2.shape[1]:
            raise ValueError('Site count mismatch between child likelihoods')

        # get transition matrices for branches from parent to node1/node2
        t1 = node1.metadata.get('length', 0.0)
        t2 = node2.metadata.get('length', 0.0)
        P1 = self._transition_matrix(t1)
        P2 = self._transition_matrix(t2)
        # derivatives of P wrt its own branch length (dP/dt, d2P/dt2)
        dP1 = self._dP_dt(t1)
        dP2 = self._dP_dt(t2)
        d2P1 = self._d2P_dt2(t1)
        d2P2 = self._d2P_dt2(t2)

        # propagate: M = P @ L_child  (P: n x n, L_child: n x S) -> n x S
        M1 = P1.dot(L1)
        M2 = P2.dot(L2)

        # elementwise product across children
        L_parent = M1 * M2

        # scaling per site to avoid underflow
        scales = L_parent.sum(axis=0)
        # avoid zeros
        eps = 1e-300
        zero_mask = scales <= 0
        scales[zero_mask] = eps
        L_parent = L_parent / scales

        # now compute first and second derivatives for all parameters
        Pcount = self.num_params
        S = L_parent.shape[1]
        if Pcount == 0:
            d1_parent = np.zeros((0, self.nstates, S))
            d2_parent = np.zeros((0, 0, self.nstates, S))
        else:
            d1_parent = np.zeros((Pcount, self.nstates, S))
            d2_parent = np.zeros((Pcount, Pcount, self.nstates, S))

            # which parameter corresponds to node1/node2 branch (if any)
            idx1 = self.param_index.get(id(node1), None)
            idx2 = self.param_index.get(id(node2), None)

            # precompute M and children derivatives
            # For each parameter p, compute dM1/dp and dM2/dp
            dM1 = np.zeros((Pcount, self.nstates, S))
            dM2 = np.zeros((Pcount, self.nstates, S))
            d2M1 = np.zeros((Pcount, Pcount, self.nstates, S))
            d2M2 = np.zeros((Pcount, Pcount, self.nstates, S))

            # child 1 contributions
            for p in range(Pcount):
                # dP/dp is non-zero only if p == idx1
                term = np.zeros((self.nstates, S))
                if idx1 is not None and p == idx1:
                    term = dP1.dot(L1)
                # add contribution from derivative of L1
                term += P1.dot(d1_1[p])
                dM1[p] = term
            # child 2
            for p in range(Pcount):
                term = np.zeros((self.nstates, S))
                if idx2 is not None and p == idx2:
                    term = dP2.dot(L2)
                term += P2.dot(d1_2[p])
                dM2[p] = term

            # second derivatives
            for p in range(Pcount):
                for q in range(Pcount):
                    term1 = np.zeros((self.nstates, S))
                    term2 = np.zeros((self.nstates, S))
                    # for child1
                    if idx1 is not None:
                        if p == idx1 and q == idx1:
                            term1 += d2P1.dot(L1)
                        elif p == idx1:
                            term1 += dP1.dot(d1_1[q])
                        elif q == idx1:
                            term1 += dP1.dot(d1_1[p])
                    term1 += P1.dot(d2_1[p, q])
                    term1 += dP1.dot(d1_1[q]) if (idx1 is not None and q == idx1) else 0
                    # for child2
                    if idx2 is not None:
                        if p == idx2 and q == idx2:
                            term2 += d2P2.dot(L2)
                        elif p == idx2:
                            term2 += dP2.dot(d1_2[q])
                        elif q == idx2:
                            term2 += dP2.dot(d1_2[p])
                    term2 += P2.dot(d2_2[p, q])
                    term2 += dP2.dot(d1_2[q]) if (idx2 is not None and q == idx2) else 0

                    d2M1[p, q] = term1
                    d2M2[p, q] = term2

            # now form d1_parent and d2_parent using product rule
            for p in range(Pcount):
                d1_parent[p] = dM1[p] * M2 + M1 * dM2[p]
                for q in range(Pcount):
                    d2_parent[p, q] = d2M1[p, q] * M2 + dM1[p] * dM2[q] + dM1[q] * dM2[p] + M1 * d2M2[p, q]

        # accumulate log scales from children and this node
        log_scales = np.log(scales) + scales1 + scales2

        return L_parent, d1_parent, d2_parent, log_scales

    def _transition_matrix(self, t):
        # Return transition probability matrix P = exp(Q * t)
        n = self.nstates

        # cache key
        qid = id(self.qmatrix) if self.qmatrix is not None else 0
        key = (qid, float(t))
        if key in self._P_cache:
            return self._P_cache[key]

        # If JC-like simple case (4 states, uniform freqs, symmetric R), use analytic formula
        freqs = getattr(self.qmatrix, 'freqs', None)
        R = getattr(self.qmatrix, 'Rmatrix', None)
        use_jc = False
        if n == 4 and freqs is not None and np.allclose(freqs, np.full(4, 0.25)):
            # detect JC by Rmatrix having equal off-diagonals
            if R is not None:
                off = R[~np.eye(4, dtype=bool)]
                if np.allclose(off, off.flat[0]):
                    use_jc = True

        if use_jc:
            expf = np.exp(-4.0 / 3.0 * t)
            p_same = 0.25 + 0.75 * expf
            p_diff = 0.25 - 0.25 * expf
            P = np.full((4, 4), p_diff)
            np.fill_diagonal(P, p_same)
            self._P_cache[key] = P
            return P

        # General case: build rate matrix Q from Rmatrix and freqs (GTR-style)
        if R is None or freqs is None:
            raise NotImplementedError('General transition matrix requires Rmatrix and freqs')

        pi = np.asarray(freqs, dtype=float)
        # q_ij = r_ij * pi_j for i != j
        Q = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = R[i, j] * pi[j]
        # diagonal
        for i in range(n):
            Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]

        # matrix exponential via eigendecomposition (may produce small complex parts)
        vals, vecs = np.linalg.eig(Q)
        try:
            inv_vecs = np.linalg.inv(vecs)
        except np.linalg.LinAlgError:
            # fallback: use series approximation (very small t) or raise
            raise RuntimeError('Eigen-decomposition of Q failed; cannot compute expm')

        exp_diag = np.exp(vals * t)
        P = vecs.dot(np.diag(exp_diag)).dot(inv_vecs)
        P = np.real(P)

        # numerical fixes: clip tiny negatives, renormalize rows
        P[P < 0] = 0.0
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        P = P / row_sums[:, None]

        self._P_cache[key] = P
        return P

    def _rate_matrix(self):
        # build Q (generator) from qmatrix Rmatrix and freqs
        R = getattr(self.qmatrix, 'Rmatrix', None)
        freqs = getattr(self.qmatrix, 'freqs', None)
        if R is None or freqs is None:
            raise RuntimeError('QMatrix must have Rmatrix and freqs to build rate matrix')
        n = self.nstates
        pi = np.asarray(freqs, dtype=float)
        Q = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = R[i, j] * pi[j]
        for i in range(n):
            Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]
        return Q

    def _dP_dt(self, t):
        # dP/dt = Q P
        key = ('d', id(self.qmatrix), float(t))
        if key in self._P_cache:
            return self._P_cache[key]
        P = self._transition_matrix(t)
        Q = self._rate_matrix()
        dP = Q.dot(P)
        self._P_cache[key] = dP
        return dP

    def _d2P_dt2(self, t):
        # d2P/dt2 = Q^2 P
        key = ('d2', id(self.qmatrix), float(t))
        if key in self._P_cache:
            return self._P_cache[key]
        Q = self._rate_matrix()
        P = self._transition_matrix(t)
        Q2 = Q.dot(Q)
        d2P = Q2.dot(P)
        self._P_cache[key] = d2P
        return d2P
