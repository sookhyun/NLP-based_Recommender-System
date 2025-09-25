import pandas
import numpy as np
from typing import Dict, List, Optional, Union

class ItemMap:
    def __init__(self, dfitem, dfcat):
        self.df_item = dfitem
        self.dict_items = self.get_dict_items()
        self.flat_items = self.get_flat_items(self.dict_items)
        self.by_item = {item:(index, freq) for index, (item,freq) in enumerate(list(self.flat_items.items()))}
        self.by_index = {index:(item, freq) for index, (item,freq) in enumerate(list(self.flat_items.items()))}   
        self.dim_items = len(self.flat_items.values())
        self.discarded = -1
        
        cat_set = set()
        for lev in dfcat.columns:
            cat_set.update(dfcat[lev].unique())
        cat_set.remove(10000)
        self.by_category ={val: i for i, val in enumerate(cat_set)}    
        self.dim_categories = len(self.by_category)
    

    def get_dict_items(self):   
        dict_items = {}    
        for category, item in self.df_item.groupby('categoryid'):
            if category <0:
                continue
            d = dict(zip(item['itemid'], item['frequency']))
            dict_items[category] = d
        return dict_items
        
    def get_flat_items(self, dictitems):
        dict_flat = {}
        for category, itemsdict in dictitems.items():
            dict_flat.update(itemsdict)
        sorted_dict_flat = dict(sorted(dict_flat.items(), key=lambda item: item[1]))
        return sorted_dict_flat

    def get_item_index(self, item: Union[int, List]):
        if isinstance(item, int):
            if item in self.by_item:
                return self.by_item.get(item)[0]
            else:
                return self.by_item.get(self.discarded)[0]
        elif isinstance(item, list):
            ans = []
            for w in item:
                if w in self.by_item:
                    ans.append(self.by_item.get(w)[0])
                else:
                    print(w, 'not in items list')
                    ans.append(self.by_item.get(self.discarded)[0])
            return ans
        else:
            raise ValueError(f"Item {item} should be an integer or a list of integers.")

    
    def get_item(self, index: Union[int, List]):
        if isinstance(index, (int, np.int64)):
            if index in self.by_index:
                return self.by_index.get(index)[0]
            else:
                raise ValueError(f"Index {index} not in valid range")
        elif isinstance(index, list):
            ans = []
            for j in index:
                if j in self.by_index:
                    ans.append(self.by_index.get(j)[0])
                else:
                    raise ValueError(f"Index {j} not in valid range.")
            return ans     

    
    def get_category_index(self, category: Union[int, List]):
        if isinstance(category, int):
            if item in self.by_category:
                return self.by_category[category]
            else:
                return self.by_category[self.discarded]
        elif isinstance(category, list):
            ans = []
            for c in category:
                if c in self.by_category:
                    ans.append(self.by_category[c])
                else:
                    print(c, 'not in category list')
                    ans.append(self.by_category[self.discarded])
            return ans
        else:
            raise ValueError(f"Category {category} should be an integer or a list of integers.")
    
