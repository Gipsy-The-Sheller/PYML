from random import choice

def NNI(tree, node):
    """
    Nearest Neighbor Interchange (NNI) operation on a tree at a given internal node.
    """

    # For a node to perform NNI:
    # 1. go back to its parent and find its sister node (the other child of the parent).
    # 2. exchange one of its children with its sister node.

    if node not in tree.nodes:
        raise ValueError("Node not in tree")
    
    if not node.children:
        raise ValueError("Node must be internal (have children) for NNI")
    
    if not node.parent:
        raise ValueError("Node must have a parent for NNI")
    
    sister = [i for i in node.parent.children if i != node][0]
    to_swap = node.children[choice([0, 1])]

    sister.parent.children.remove(sister)
    sister.parent = node
    node.children.append(sister)

    node.children.remove(to_swap)
    to_swap.parent = node.parent
    node.parent.children.append(to_swap)
