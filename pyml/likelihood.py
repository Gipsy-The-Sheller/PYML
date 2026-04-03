# core vectorized likelihood calculator for Felsenstein (1981) pruning algorithm

import numpy as np

def dna_exceptions(trait):
    if trait in ['-', 'N', '?']:
        solution = [0.25, 0.25, 0.25, 0.25]  # equal likelihood for all states
    elif trait == 'R':
        solution = [0.5, 0, 0.5, 0]  # A or G
    elif trait == 'Y':
        solution = [0, 0.5, 0, 0.5]  # C or T
    else:
        raise ValueError(f"DNA trait exception processing: Unexpected trait '{trait}'")
    return {'A': solution[0], 'C': solution[1], 'G': solution[2], 'T': solution[3]}

class likelihoodCalculator:
    def __init__(self, phyloData, model, exceptions=None):
        """
        model: could be a QMatrix instance, or a customized class to deal with more complex models.
        exceptions: a function to process special traits (e.g. degenerate sites / gap sites / missings)
        """

        self.nstates = phyloData.nstates
        self.data_length = phyloData.alignment.shape[1]
        self.phyloData = phyloData
        self.model = model

        # initialize likelihood vector arrays
        # Create an array (rows: nodes [internal and leaf], columns: sites * states)
        # then distribute it to phyloData metadata for each node
        
        # NOTE: Why use sites * states for columns: spread out the likelihood vector,
        # so that we can use numpy to easily vectorize the likelihood calculation

        self.n_inodes = len(phyloData.tree.nodes)
        self.lik_array = np.zeros((self.n_inodes, self.data_length * self.nstates))

        self.exceptions = exceptions
        self.init_nodes()
    
    def default_model(self, t):
        # DEFAULT: directly use the phyloData's Qmatrix to generate a concatenated array of nsites * Pmatrix
        Pmatrix = self.phyloData.qmatrix.Pmatrix(t)
        return np.tile(Pmatrix, (self.data_length, 1))

    def set_branch_lengths_from_params(self, params: dict):
        # map parameters named 'brlens_{i}' to non-root nodes in the same order as tree.nodes
        nonroot_nodes = [n for n in self.phyloData.tree.nodes if not n.is_root]
        for i, node in enumerate(nonroot_nodes):
            key = f'brlens_{i}'
            if key in params:
                node.branch_length = params[key]

    def reset_internal_calculated_flags(self):
        for node in self.phyloData.tree.nodes:
            if not node.is_leaf:
                node.metadata['calculated'] = False

    def log_likelihood(self, params: dict):
        # apply branch lengths / topology from params then compute log likelihood
        # update branch lengths
        self.set_branch_lengths_from_params(params)

        # reset calculation flags for internal nodes
        self.reset_internal_calculated_flags()

        # find root
        root = None
        for n in self.phyloData.tree.nodes:
            if n.is_root:
                root = n
                break
        if root is None:
            raise ValueError('No root found in tree')

        return self.prune(root)

    def init_nodes(self):
        for i, node in enumerate(self.phyloData.tree.nodes):
            node.metadata['lik_index'] = i  # index in the likelihood array (for debugging)
            node.metadata['calculated'] = False  # track if likelihood has been computed
            
            if node.is_leaf:
                # initialize likelihood vector for leaf nodes based on observed traits
                seq_id = node.metadata['name']
                row = self.phyloData.alignment.loc[seq_id]

                for index, site in enumerate(row):
                    if site in self.phyloData.states:
                        state_index = self.phyloData.qmatrix.state_to_index(site)
                        self.lik_array[i, index*self.nstates + state_index] = 1
                    else:
                        solution = self.exceptions(site) if self.exceptions else [1/self.nstates] * self.nstates
                        for j in range(self.nstates):
                            self.lik_array[i, index*self.nstates + j] = solution[j]
                
                # Store the likelihood vector reference and mark as calculated
                node.metadata['lik_vector'] = self.lik_array[i].reshape(self.data_length, self.nstates).T
                node.metadata['calculated'] = True
    
    def get_pmatrix(self, t):
        """Get transition probability matrix for branch length t."""
        if self.model is not None:
            return self.model(t)
        elif self.phyloData.qmatrix is not None:
            return self.phyloData.qmatrix.Pmatrix(t)
        else:
            raise ValueError("No model or qmatrix available for computing transition matrix")
    
    def prune(self, node):
        """
        Implement Felsenstein's pruning algorithm to compute likelihoods at internal nodes.
        Uses post-order traversal with explicit stack.
        """
        stack_depth = 0
        while True:
            # Check if this node's likelihood has been calculated
            if node.metadata.get('calculated', False):
                break
            
            left, right = node.children[0], node.children[1]

            # If both children have calculated likelihoods, compute this node
            if left.metadata.get('calculated', False) and right.metadata.get('calculated', False):
                # Get branch lengths
                t_left = left.branch_length if left.branch_length is not None else left.metadata.get('length', 0.0)
                t_right = right.branch_length if right.branch_length is not None else right.metadata.get('length', 0.0)
                
                # Get transition probability matrices
                P_left = self.get_pmatrix(t_left)
                P_right = self.get_pmatrix(t_right)

                # Get child likelihood vectors (shape: nstates x nsites)
                L_left = left.metadata['lik_vector']
                L_right = right.metadata['lik_vector']

                # Compute likelihood vector for this node
                # L_parent[i] = sum_j P_ij * L_child[j] for each site
                likvec_left = np.dot(P_left, L_left)
                likvec_right = np.dot(P_right, L_right)
                L_parent = likvec_left * likvec_right  # element-wise multiplication

                # Store result
                node.metadata['lik_vector'] = L_parent
                node.metadata['calculated'] = True

                # Move up to parent node if exists, otherwise we're done
                if node.parent is not None:
                    node = node.parent
                    stack_depth = max(0, stack_depth - 1)
                    continue
                else:
                    break
            
            # Traverse down to uncalculated children
            elif left.metadata.get('calculated', False):
                node = right
                stack_depth += 1
            else:
                node = left
                stack_depth += 1

        # At root: combine with stationary frequencies
        freqs = self.phyloData.qmatrix.freqs if self.phyloData.qmatrix and self.phyloData.qmatrix.freqs is not None else np.array([1/self.nstates] * self.nstates)
        
        root_lik_vec = node.metadata['lik_vector']  # shape: nstates x nsites
        
        # Per-site likelihood: sum_i pi_i * L_i for each site
        site_liks = np.dot(freqs, root_lik_vec)  # shape: (nsites,)
        
        # Handle potential numerical issues
        site_liks = np.where(site_liks <= 0, 1e-300, site_liks)
        
        log_likelihood = np.sum(np.log(site_liks))
        return log_likelihood