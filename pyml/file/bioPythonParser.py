# compatibility layer with Biopython

from Bio import Phylo
from ..phyloData import Node, Topology

def BioPhylo2Topology(phylo):
    # Convert Biopython Phylo tree to our Topology and Node structure
    # This is a recursive function that traverses the Biopython tree and constructs our Topology

    if not isinstance(phylo, Phylo.BaseTree.Tree):
        raise ValueError("Input must be a Biopython Phylo tree")

    def convert_clade(clade, parent_node=None):
        node= Node(parent=parent_node, branch_length=clade.branch_length, name=clade.name)
        nodes = []
        for child in clade.clades:
            _, child_nodes = convert_clade(child, parent_node=node)
            nodes.extend(child_nodes)
        return node, [node] + nodes
    
    root_clade = phylo.root
    root_node, nodes = convert_clade(root_clade)
    
    topo = Topology(n_leaves=len(phylo.get_terminals()))
    topo.nodes = nodes
    return topo