from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import pandas as pd
import numpy as np

RELEASE = False
if RELEASE:
    from .maths.expm import expm_core as expm
else:
    from scipy.linalg import expm

class Node:
    def __init__(self, parent=None, children=None, branch_length:float=None, name:str=None, **metadata):
        self.parent = None
        if parent:
            parent.add_child(self)

        self.children = children
        self.branch_length = branch_length
        # compatibility alias used elsewhere
        self.brlength = branch_length
        self.metadata = metadata or {}
        if name:
            self.metadata['name'] = name
    
    def add_child(self, child_node):
        if self.children is None:
            self.children = [child_node]
        else:
            self.children.append(child_node)
        child_node.parent = self

    @property
    def is_root(self):
        return self.parent is None
    
    @property
    def is_leaf(self):
        return self.children is None

class Topology:
    def __init__(self, n_leaves):
        self.nleaves = n_leaves
        self.nodes = [Node(None, None) for _ in range(n_leaves * 2 - 1)]

class PhyloData:
    def __init__(self, alignment, states, tree:Topology = None, qmatrix=None):
        # In this version I want to use pandas dataframe for alignment, 
        # to quick extract each column for likelihood estimation.

        # Also I want to keep compability with Bio.SeqRecord and Bio.MultipleSeqAlignment,
        # so I will implement a compatibility layer for format conversion.

        
        self.tree = tree
        self.qmatrix = qmatrix

        ali_type = type(alignment)
        if ali_type == pd.DataFrame:
            self.alignment = alignment
        elif ali_type == list and all(isinstance(x, SeqRecord) for x in alignment):
            # convert to DataFrame: rows are taxa, columns are sites
            self.alignment = pd.DataFrame(
                {rec.id: list(str(rec.seq)) for rec in alignment}
                ).T
        elif ali_type == MultipleSeqAlignment:
            self.alignment = pd.DataFrame(
                {rec.id: list(str(rec.seq)) for rec in alignment}
            ).T
        else:
            raise ValueError(f"Unsupported alignment type: {ali_type}")
        
        self.states = states
        self.nstates = len(states) if states else alignment.nunique() if isinstance(alignment, pd.DataFrame) else None

class SubstMatrix:
    def __init__(self, nstates, statelabels=None, **matrixinfo):

        """
        nstates: number of states in the model (e.g. 4 for DNA, 20 for amino acids)
        statelabels: optional list of state labels (e.g. ['A','C','G','T'] for DNA)
        matrixinfo: must specify exactly one of the following sets of attributes:
            set 1. Rmatrix (nstates x nstates) + freqs (nstates)
            set 2. Qmatrix (nstates x nstates)
            set 3. Rconsts (list of rate constants) + parameters (dict of parameter values) + freqs (nstates)
        matrices can be represented in different ways:
        - nested lists
        - numpy arrays
        - pandas DataFrames (with statelabels as index and columns)
        """
        self.nstates = nstates
        self.statelabels = statelabels

        # matrixinfo can specify the following attributes:
        # set 1. Rmatrix + freqs
        # set 2. Qmatrix
        # set 3. Rconsts + parameters + freqs

        set1_decide_flag = 'Rmatrix' in matrixinfo and 'freqs' in matrixinfo
        set1_check_flag = 'Rmatrix' in matrixinfo or 'freqs' in matrixinfo
        set2_decide_flag = 'Qmatrix' in matrixinfo
        set2_check_flag = 'Qmatrix' in matrixinfo
        set3_decide_flag = 'Rconsts' in matrixinfo and 'parameters' in matrixinfo and 'freqs' in matrixinfo
        set3_check_flag = 'Rconsts' in matrixinfo or 'parameters' in matrixinfo or 'freqs' in matrixinfo

        self.SET1_TYPE = 'set1'
        self.SET2_TYPE = 'set2'
        self.SET3_TYPE = 'set3'

        # check validity
        if sum([set1_decide_flag, set2_decide_flag, set3_decide_flag]) != 1:
            raise ValueError("Must specify exactly one of the three sets of attributes for SubstMatrix")
        
        if set1_decide_flag:
            self.type = self.SET1_TYPE
            self.Rmatrix = matrixinfo['Rmatrix']
            self.freqs = matrixinfo['freqs']

        elif set2_decide_flag:
            self.type = self.SET2_TYPE
            self.Q = matrixinfo['Qmatrix']
        
        elif set3_decide_flag:
            self.type = self.SET3_TYPE
            self.Rconsts = matrixinfo['Rconsts']
            self.parameters = matrixinfo['parameters']
            self.freqs = matrixinfo['freqs']
    
    @property
    def Qmatrix(self):
        if self.type == self.SET1_TYPE:
            # Compute Qmatrix from Rmatrix and freqs: Q[i,j] = R[i,j] * pi[j] for i != j
            R = np.asarray(self.Rmatrix, dtype=float)
            pi = np.asarray(self.freqs, dtype=float)
            n = self.nstates
            Q = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        Q[i, j] = R[i, j] * pi[j]
            for i in range(n):
                Q[i, i] = -np.sum(Q[i, :])
            return Q
        elif self.type == self.SET2_TYPE:
            return np.asarray(self.Q, dtype=float)
        elif self.type == self.SET3_TYPE:
            # construct Rmatrix from Rconsts and parameters, then compute Qmatrix
            raise NotImplementedError("SET3_TYPE not yet implemented")

    def Pmatrix(self, t):
        # compute transition probability matrix P(t) = exp(Qt)
        return expm(self.Qmatrix * t)
    
    def state_to_index(self, state):
        if self.statelabels is not None:
            return self.statelabels.index(state)
        else:
            raise ValueError("State labels not defined for this SubstMatrix")