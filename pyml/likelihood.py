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
        Pmatrix = self.phyloData.Pmatrix(t)
        return np.tile(Pmatrix, (self.data_length, 1))

    def init_nodes(self):
        for i, node in enumerate(self.phyloData.tree.nodes):
            node.metadata['lik_index'] = i  # index in the likelihood array
            if node.is_leaf:
                # initialize likelihood vector for leaf nodes based on observed traits

                # locate the row for this leaf
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


                # ABANDONED!
                # for site in range(self.data_length):
                #     # find the row in alignment for this leaf (with id corresponding to the node)
                #     trait = self.phyloData.alignment.loc[node.metadata['name'], site]
                #     if trait in self.phyloData.states:
                #         state_index = self.phyloData.qmatrix.state_to_index[trait]
                #         node.metadata['lik_index'] = [1 if j == state_index else 0 for j in range(self.nstates)]
                #     else:
                #         solution = self.exceptions(trait)
                #         node.metadata['lik_index'] = [solution[index] for index, state in enumerate(self.phyloData.states)]
    
    def prune(self, node):
        """
        implement Felsenstein's pruning algorithm to compute likelihoods at internal nodes.
        """
        
        # ABANDONED!
        # we preserve a vector cache to store all quotes to likelihood vectors to facilitate vectorized computation
        # likvec_left = np.array([])
        # likvec_right = np.array([])
        # p_left = np.array([])
        # p_right = np.array([])

        # Then traverse the subtree to add tasks for the cache
        stack_depth = 0
        while True:
            if np.any(node.metadata['lik_index']):
                # already calculated. break.
                break
            left, right = node.children[0], node.children[1]

            # if left and right nodes already have non-zero likelihood vectors, then we can prune this node
            if np.any(left.metadata['lik_index']) and np.any(right.metadata['lik_index']):
                # # compute transition probability matrices for left and right branches
                # p_left = self.phyloData.Pmatrix(left.metadata['length'])
                # p_right = self.phyloData.Pmatrix(right.metadata['length'])

                # # compute likelihood vector for this node by multiplying transition probabilities with child likelihood vectors
                # likvec_left = np.dot(p_left, left.metadata['lik_index'])
                # likvec_right = np.dot(p_right, right.metadata['lik_index'])
                # node.metadata['lik_index'] = likvec_left * likvec_right  # element-wise multiplication

                # generate pmatrix array with self.model
                p_left = self.model(left.metadata['length'])
                p_right = self.model(right.metadata['length'])

                # compute likelihood vector for this node by multiplying transition probabilities with child likelihood vectors
                likvec_left = np.dot(p_left, left.metadata['lik_index'])
                likvec_right = np.dot(p_right, right.metadata['lik_index'])
                node.metadata['lik_index'] = likvec_left * likvec_right  # element-wise multiplication

                # move up to parent node
                if stack_depth > 0:
                    node = node.parent
                    continue
                else:
                    break
            
            # if the likelihood vector for left node is already computed (but not for right node), then we can move down to right node

            elif np.any(left.metadata['lik_index']):
                node = right
                stack_depth += 1
            
            else:
                node = left
                stack_depth += 1

        # the root likelihood vector should be generated with frequencies
        # if we cannot obtain freqs, then use equal freqs
        freqs = self.phyloData.qmatrix.freqs if self.phyloData.qmatrix and self.phyloData.qmatrix.freqs is not None else np.array([1/self.nstates] * self.nstates)
        root_lik_vec = node.metadata['lik_index'] * freqs
        log_likelihood = np.sum(np.log(root_lik_vec))
        return log_likelihood