from ItemMap import ItemMap

class TreeNode:
    def __init__(self, index=None):

        self.index = index  # Leaf index for items; internal nodes get assigned later
        self.members = {}
        self.parent= None
        self.is_item = False
        
    def add_member(self, idx, is_item=False):
        self.members[idx] = TreeNode(idx)
        self.members[idx].parent =self
        self.members[idx].is_item = is_item


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
    if (tnodes.index == target) and (tnodes.is_item == False):
        tnodes.add_member(new, True) # set is_item to True
        return
    for n in tnodes.members.keys():    
        add_to_node(tnodes.members[n], target, new)


def build_category_tree(rootcode_, df_, itemmap_ : ItemMap):
    tree_cat = TreeNode(rootcode_) 
    build_tree(0,{rootcode_:tree_cat}, df_) 
    for catid, itemid in itemmap_.items():
        for itemidx in itemid.keys():
            add_to_node(tree_cat,catid,itemidx) 
    return tree_cat


def get_path(tr, idx, path):
    if tr is None:
        return 

    if (tr.index == idx) and (tr.is_item == True):
        path.append(tr.index)

        while True:
            if tr.parent is None:
                break             
            path.append(tr.parent.index)
            tr = tr.parent
        return 
        
    for id in tr.members.keys():
        get_path(tr.members[id], idx,path)
