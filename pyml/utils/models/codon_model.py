from ...import SubstMatrix
RELEASE = False
if RELEASE:
    from ...maths.expm import expm_core as expm
else:
    from scipy.linalg import expm

codon_statelabels = [
    'AAA', 'AAC', 'AAG', 'AAT',
    'ACA', 'ACC', 'ACG', 'ACT',
    'AGA', 'AGC', 'AGG', 'AGT',
    'ATA', 'ATC', 'ATG', 'ATT',
    'CAA', 'CAC', 'CAG', 'CAT',
    'CCA', 'CCC', 'CCG', 'CCT',
    'CGA', 'CGC', 'CGG', 'CGT',
    'CTA', 'CTC', 'CTG', 'CTT',
    'GAA', 'GAC', 'GAG', 'GAT',
    'GCA', 'GCC', 'GCG', 'GCT',
    'GGA', 'GGC', 'GGG', 'GGT',
    'TAA', 'TAC', 'TAG', 'TAT',
    'TCA', 'TCC', 'TCG', 'TCT',
    'TGA', 'TGC', 'TGG', 'TGT',
    'TTA', 'TTC', 'TTG', 'TTT'
]

# Codon tables

codon_tables = []

standard = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

class CodonModel:
    def __init__(self, codon_table):
        self.codon_table = codon_table
        self.statelabels = []
        for codon in codon_statelabels.keys():
            aa = codon_table[codon]
            if aa != '*':  # skip stop codons
                self.statelabels.append(codon)
        self.nstates = len(self.statelabels)
        # self.substmodel = SubstMatrix(nstates=self.nstates, statelabels=self.statelabels, Rmatrix=None, freqs=None)  
        # placeholder for now, to be implemented with specific codon substitution models (e.g. GY94)

class GY94(CodonModel):
    def __init__(self, codon_table=standard, kappa=1.0, omega=1.0):
        """
        kappa: transition/transvertion ratio
        omega: dN/dS ration
        NOTE: degenerate codons temporarily not considered; gaps/missings not considered.
        """
        super().__init__(codon_table)
        self.kappa = kappa
        self.omega = omega
        self.statesfreqs = [1/super().nstates] * super().nstates


        # Yang (2006) Computational Molecular Evolution, p. 48:
        # q_{i,j} = {0 if i and j differ at more than one position,
        #            freq_j if synonymous transversion,
        #            kappa * freq_j if synonymous transition,
        #            omega * freq_j if non-synonymous transversion,
        #            kappa * omega * freq_j if non-synonymous transition}
        self.substs = {
            'synonymous_transition': lambda freq, kappa, omega: kappa * freq,
            'synonymous_transversion': lambda freq, kappa, omega: freq,
            'missense_transition': lambda freq, kappa, omega: kappa * omega * freq,
            'missense_transversion': lambda freq, kappa, omega: omega * freq,
            '1+': lambda freq, kappa, omega: 0 # more than 1 nt variation in a single moment is not considered
        }
        self.Qmatrix = [[0] * super().nstates for _ in range(super().nstates)]
        self.create_Qmatrix_shape()
    
    def create_Qmatrix_shape(self):
        
        for i, codon_i in enumerate(super().statelabels):
            for j, codon_j in enumerate(super().statelabels):
                if i == j:
                    # diagonal elements are temporarily not filled
                    continue
                subst_type = self.subst_type(codon_i, codon_j)
                if subst_type is None:
                    # duplexed 'diagonel' indicating codon duplexes
                    raise ValueError("Unknown error: You may probably have defined duplexed codons")
                self.Qmatrix[i][j] = self.substs[subst_type]

        # diagonal elements: Q[i][i] = -sum(Q[i][j] for j != i)
        # for i in range(len(super().statelabels)):
        #     self.Qmatrix[i][i] = -sum(self.Qmatrix[i][j] for j in range(super().nstates) if j != i)
        # NOTE: move to property ,ethod to compute diagonal elements
    
    def subst_type(self, codon_i, codon_j):
        # determine the type of substitution between two codons (synonymous/missense, transition/transversion)
        # return one of the keys in self.substs
        diff_positions = [pos for pos in range(3) if codon_i[pos] != codon_j[pos]]
        if len(diff_positions) == 0:
            return None
        elif len(diff_positions) > 1:
            return '1+'
        else:
            pos = diff_positions[0]
            base_i = codon_i[pos]
            base_j = codon_j[pos]
            if (base_i in ['A', 'G'] and base_j in ['A', 'G']) or (base_i in ['C', 'T'] and base_j in ['C', 'T']):
                # transition
                if self.codon_table[codon_i] == self.codon_table[codon_j]:
                    return 'synonymous_transition'
                else:
                    return 'missense_transition'
            else:
                # transversion
                if self.codon_table[codon_i] == self.codon_table[codon_j]:
                    return 'synonymous_transversion'
                else:
                    return 'missense_transversion'
    
    @property
    def Rmatrix(self):
        R = [[0] * super().nstates for _ in range(super().nstates)]
        for i in range(super().nstates):
            for j in range(super().nstates):
                if i != j:
                    # Rmatrix does not consider frequancies.
                    R[i][j] = self.Qmatrix[i][j](1, self.kappa, self.omega)
        
        for i in range(super().nstates):
            R[i][i] = -sum(R[i][j] for j in range(super().nstates) if j != i)
        
        return R

    @property
    def Qmatrix(self):
        Q = [[0] * super().nstates for _ in range(super().nstates)]
        for i in range(super().nstates):
            for j in range(super().nstates):
                if i != j:
                    Q[i][j] = self.Qmatrix[i][j](self.statesfreqs[j], self.kappa, self.omega)
        
        for i in range(super().nstates):
            Q[i][i] = -sum(Q[i][j] for j in range(super().nstates) if j != i)
        
        return Q
    
    def Pmatrix(self, t):
        # compute transition probability matrix P(t) = exp(Qt)
        return expm(self.Qmatrix * t)
    
class MG94(GY94):
    def __init__(self, codon_table=standard, omega=1.0):
        """
        MG94 model: do not consider transition/transversion bias (kappa=1.0).
        """
        # fix kappa = 1.0 so that GY94 reduces to MG94
        super().__init__(codon_table, 1, omega)