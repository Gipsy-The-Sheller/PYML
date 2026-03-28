"""Simple Metropolis-Hastings sampler for tree topology and branch lengths.

Default priors: branch lengths ~ Uniform(0, 10); topology uniform.
Proposals: NNI for topology (choose random internal node), log-scale normal for branch lengths.

Outputs:
- A TSV file with columns: `sample`, `prior_log`, `likelihood`, `posterior_log` for saved samples.
- A .trees file with one Newick tree per saved sample (one line each).

This is a minimal implementation for experimentation, not optimized.
"""
import copy
import math
import random
import time
from typing import Optional

from pyml import PhyloData, ML_Estimator, Topology, Node


def _newick_from_node(node: Node) -> str:
    """Serialize node (and subtree) to Newick. Branch length shown for child edge.
    Assumes binary tree.
    """
    def rec(n: Node):
        if n.is_leaf():
            name = n.metadata.get('name', n.name if hasattr(n, 'name') else '')
            s = name if name is not None else ''
        else:
            children = n.children
            s = '(' + ','.join(rec(c) for c in children) + ')'
        # append branch length (length to parent) if present and not root
        if not n.is_root():
            l = n.length if hasattr(n, 'length') else n.metadata.get('length', None)
            if l is None:
                return s
            return f"{s}:{l:.8f}"
        return s

    return rec(node) + ';'


def _find_root(topology: Topology) -> Optional[Node]:
    for n in topology.nodes:
        if n.is_root():
            return n
    return None


def _nni_neighbors(tree: Topology, node: Node):
    """Return two neighbor trees by NNI at `node`. Operates on deepcopy clones and
    returns list of Topology copies (may be fewer than 2 if not applicable).
    """
    # validate
    if node.parent is None:
        return []
    if len(node.children) != 2:
        return []
    parent = node.parent
    sisters = [c for c in parent.children if c is not node]
    if not sisters:
        return []
    sister = sisters[0]
    if len(sister.children) != 2:
        return []

    a, b = node.children
    c, d = sister.children
    neighbors = []
    for x, y in ((a, c), (a, d)):
        tcopy = copy.deepcopy(tree)
        # locate corresponding nodes by uid
        node_copy = next(n for n in tcopy.nodes if n.uid == node.uid)
        sister_copy = next(n for n in tcopy.nodes if n.uid == sister.uid)
        x_copy = next(n for n in tcopy.nodes if n.uid == x.uid)
        y_copy = next(n for n in tcopy.nodes if n.uid == y.uid)
        # swap: replace x with y in node_copy.children, and replace y with x in sister_copy.children
        i_node = node_copy.children.index(x_copy)
        i_sis = sister_copy.children.index(y_copy)
        node_copy.children[i_node] = y_copy
        sister_copy.children[i_sis] = x_copy
        x_copy.parent = sister_copy
        y_copy.parent = node_copy
        neighbors.append(tcopy)
    return neighbors


class MHSampler:
    def __init__(self, seqs, qmatrix, tree: Topology, out_prefix='chain', branch_prior_max=10.0,
                 p_topo=0.3, log_sigma=0.1, random_seed=None):
        self.seqs = seqs
        self.qmatrix = qmatrix
        self.tree = tree
        self.out_prefix = out_prefix
        self.branch_prior_max = float(branch_prior_max)
        self.p_topo = float(p_topo)
        self.log_sigma = float(log_sigma)
        if random_seed is not None:
            random.seed(random_seed)

    def _log_prior_lengths(self, topo: Topology) -> float:
        # uniform(0, M) for each non-root branch
        M = self.branch_prior_max
        for n in topo.nodes:
            if n.is_root():
                continue
            l = n.length
            if l is None or l <= 0 or l >= M:
                return -float('inf')
        # each branch has density 1/M -> log prior = -Nbranches * log M
        nbranches = sum(1 for n in topo.nodes if not n.is_root())
        return -nbranches * math.log(M)

    def _log_likelihood(self, topo: Topology) -> float:
        # build PhyloData and ML_Estimator and compute likelihood
        pd = PhyloData(self.seqs, nstates=len(set(str(self.seqs[0].seq))), tree=topo, qmatrix=self.qmatrix)
        ml = ML_Estimator(pd)
        res = ml.compute_likelihood()
        if isinstance(res, tuple):
            return float(res[0])
        return float(res)

    def _propose_branch_length(self, topo: Topology):
        # choose a random non-root node and propose log-scale symmetric move
        candidates = [n for n in topo.nodes if not n.is_root()]
        if not candidates:
            return None
        node = random.choice(candidates)
        old = node.length
        if old is None or old <= 0:
            old = 0.1
        log_old = math.log(old)
        log_new = random.gauss(log_old, self.log_sigma)
        new = math.exp(log_new)
        # create copy and set new length
        tcopy = copy.deepcopy(topo)
        node_copy = next(n for n in tcopy.nodes if n.uid == node.uid)
        node_copy.length = new
        return tcopy

    def _propose_topology(self, topo: Topology):
        # choose a random internal node with parent and binary children
        candidates = [n for n in topo.nodes if n.parent is not None and len(n.children) == 2]
        if not candidates:
            return None
        node = random.choice(candidates)
        neighbors = _nni_neighbors(topo, node)
        if not neighbors:
            return None
        # choose one neighbor uniformly
        return random.choice(neighbors)

    def run(self, niter=1000, burnin=100, thin=1):
        trees_path = f"{self.out_prefix}.trees"
        tsv_path = f"{self.out_prefix}.tsv"

        # open files
        tf = open(trees_path, 'w')
        tv = open(tsv_path, 'w')
        tv.write('sample\tprior_log\tlikelihood\tposterior_log\n')

        current_tree = copy.deepcopy(self.tree)
        current_prior = self._log_prior_lengths(current_tree)
        if current_prior == -float('inf'):
            raise RuntimeError('Initial tree has branch lengths outside prior support')
        current_like = self._log_likelihood(current_tree)
        current_post = current_prior + current_like

        saved = 0
        accepts = 0
        for it in range(1, niter + 1):
            if random.random() < self.p_topo:
                prop = self._propose_topology(current_tree)
            else:
                prop = self._propose_branch_length(current_tree)

            if prop is None:
                # no valid proposal; skip
                continue

            prop_prior = self._log_prior_lengths(prop)
            if prop_prior == -float('inf'):
                accept = False
            else:
                prop_like = self._log_likelihood(prop)
                prop_post = prop_prior + prop_like
                log_alpha = prop_post - current_post
                accept = math.log(random.random()) < log_alpha

            if accept:
                current_tree = prop
                current_prior = prop_prior
                current_like = prop_like
                current_post = prop_post
                accepts += 1

            # record after burnin and respecting thinning
            if it > burnin and ((it - burnin) % thin == 0):
                saved += 1
                # write tree (Newick) and TSV line
                root = _find_root(current_tree)
                newick = _newick_from_node(root) if root is not None else '();'
                tf.write(newick + '\n')
                tv.write(f"{saved}\t{current_prior:.6e}\t{current_like:.6e}\t{current_post:.6e}\n")

        tf.close()
        tv.close()
        return {'saved': saved, 'accept_rate': accepts / float(niter)}


if __name__ == '__main__':
    # small smoke test if invoked directly
    from dna_models import jc69
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    # simple 3-taxa example
    records = [SeqRecord(Seq('ACGT'), id='A'), SeqRecord(Seq('ACGT'), id='B'), SeqRecord(Seq('ACGT'), id='C')]
    topo = Topology(3)
    # connect leaves manually: assume nodes 0..2 are leaves, 3.. are internal, last is root
    node1 = topo.nodes[3]
    root = topo.nodes[4]
    # build ((leaf0, leaf1), leaf2) rooted at root
    node1.add_child(topo.nodes[0]); node1.add_child(topo.nodes[1])
    root.add_child(node1); root.add_child(topo.nodes[2])
    topo.nodes[0].metadata['name'] = 'A'
    topo.nodes[1].metadata['name'] = 'B'
    topo.nodes[2].metadata['name'] = 'C'
    # assign some branch lengths
    for n in topo.nodes:
        if not n.is_root():
            n.length = 0.1

    sampler = MHSampler(records, jc69(ML_Estimator.__init__.__globals__['QMatrix']), topo, out_prefix='test_chain', random_seed=1)
    res = sampler.run(niter=200, burnin=50, thin=5)
    print('Done', res)
