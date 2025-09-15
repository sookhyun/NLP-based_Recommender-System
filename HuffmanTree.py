from graphviz import Digraph

class HuffmanNode:
    def __init__(self, count, index=None, left=None, right=None):
        self.count = count 
        self.index = index  # Leaf index for items; internal nodes get assigned later
        self.left = left
        self.right = right
        self.symbol = None


def build_huffman_tree(min_index, root_index=None, item_counts=None, huffnodes=None):
    # Start with leaf nodes (items)
    nodes = [HuffmanNode(count=cnt, index=idx) for idx, cnt in item_counts.items()]
    if huffnodes:
        nodes = huffnodes
    next_index = min_index# max(item_counts.keys()) + 1

    # Build tree
    while len(nodes) > 1:
        # Sort by count (ascending)
        nodes = sorted(nodes, key=lambda x: x.count)

        left = nodes.pop(0)
        right = nodes.pop(0)

        # Merge
        parent = HuffmanNode(count=left.count + right.count, index=next_index, left=left, right=right)
        nodes.append(parent)
        next_index += 1

    if root_index != None:
        nodes[0].index = root_index

    return next_index - min_index, nodes 


def generate_codebook(tree):    
    root = tree[0]
    codebook = {}

    # Assign binary codes and path of node indices
    def assign_code(node, path, code):
        if node.left is None and node.right is None:  # Leaf node
            codebook[node.index] = (code, [p.index for p in path])
        else:
            if node.left:
                assign_code(node.left, path + [node], code + [0])
            if node.right:
                assign_code(node.right, path + [node], code + [1])

    assign_code(root, [], [])
    return codebook  # codebook and total node count


def visualize_huffman_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
    
    if node.left:
        graph.node(str(node.left.index), f'{node.left.index}:{node.left.count}')
        graph.edge(str(node.index), str(node.left.index), label='0')
        visualize_huffman_tree(node.left, graph)

    if node.right:
        graph.node(str(node.right.index), f'{node.right.index}:{node.right.count}')
        graph.edge(str(node.index), str(node.right.index), label='1')
        visualize_huffman_tree(node.right, graph)
    
    return graph
