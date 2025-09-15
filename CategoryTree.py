
class TreeNode:
    def __init__(self, index=None):

        self.index = index  # Leaf index for items; internal nodes get assigned later
        self.members = {}
        self.parent= None
        
    def add_member(self, idx):
        self.members[idx] = TreeNode(idx)
        self.members[idx].parent =self



def build_tree(level=0, tree=None, df=None, _debug_=False):
    if level >5:
        return

    curlevel = 'L'+str(level)
    nextlevel = 'L'+str(level+1)
    
    for l in df[curlevel].unique():
        if l<0:
            continue

        children = df.loc[df[curlevel]== l][nextlevel].unique()
        _ = [tree[l].add_member(ch) for ch in children if ch >=0] 

        if _debug_:
            print(f'Level {level} category: {l}')    
            print(tree[l].members.keys())
        
        if tree[l].members.keys():
            df_new = df.loc[df[curlevel]== l]
            build_tree(level+1,tree[l].members, df_new)  


def add_to_node(tnodes, target, new):
    if tnodes is None:
        return
    if tnodes.index == target:
        tnodes.add_member(new)
        return
    for n in tnodes.members.keys():    
        add_to_node(tnodes.members[n], target, new)


def get_path(tr, idx, path):
    if tr is None:
        return 

    if tr.index == idx:
        path.append(tr.index)

        while True:
            if tr.parent is None:
                break
            path.append(tr.parent.index)
            tr = tr.parent
        return 
        
    for id in tr.members.keys():
        get_path(tr.members[id], idx,path)
