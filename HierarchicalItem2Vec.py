from tqdm import tqdm
from time import monotonic 
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from parameters import Params
from ItemMap import ItemMap
from batch_tool import BatchToolItem
from HuffmanTree import generate_codebook
from CategoryTree import get_path


class HierarchicalItem2Vec(nn.Module):
    def __init__(self, map : ItemMap, params: Params, huff_tree, cat_tree):
        super().__init__()

        self.debug = False
        self.map = map
        self.params = params
        self.huff_tree = huff_tree
        self.cat_tree = cat_tree           
        self.paths = generate_codebook(self.huff_tree)  # {item_id: (codes, nodes)}   

        # context items        
        self.item_embeddings = nn.Embedding(
            self.map.dim_items, 
            params.dim_embedding
        )
        nn.init.uniform_(self.item_embeddings.weight, -0.5, 0.5)

        
        # target items
        self.category_embeddings = nn.Embedding(
            self.map.dim_categories, 
            params.dim_embedding
        )
      
        self.node_weights = nn.ParameterDict()
        self.node_map = {}
        node_idx = 0
        
        # Create unique node weights
        #      node weights: parameter matrix storing the classifiers for each internal node.
        for item, (codes, nodes) in self.paths.items():
            for node in nodes:
                node_id = id(node)
                if node_id not in self.node_map:
                    name = f'node_{node_idx}'
                    self.node_map[node_id] = name
                    self.node_weights[name] = nn.Parameter(torch.randn(params.dim_embedding) * 0.1)
                    node_idx += 1

    
    
    def CosineLoss(self, item_emb, cat_ids):

        device = item_emb.device
        cat_indices = self.map.get_category_index(cat_ids) 
        cat_embs = self.category_embeddings(torch.tensor(cat_indices, device=device))
        item_emb = item_emb.unsqueeze(0).expand_as(cat_embs)
        cosine_sim = F.cosine_similarity(item_emb, cat_embs, dim=1)
        loss = 1 - cosine_sim  # cosine loss

        return loss.mean()    


    def HierarchicalSoftmaxLoss(self, center_embs, context_ids):
        """
        center_embs: FloatTensor (B, dim_embedding)        
        context_ids: LongTensor (B,)
        
        Note) The node embeddings in hierarchical softmax act as the context embeddings. The context item 
        is indirectly defined via its unique path of left/right decisions from the root.        
        """

        total_loss = 0.0
        total_steps = 0
        
        B = context_ids.shape[0] # batch size
       
        for i in range(B):
            context_id = context_ids[i].item()
            center_emb = center_embs[i]  # (D,)
         
            codes, nodes = self.paths[self.map.get_item(context_id)]  # codes: list[int], nodes: list[node]
            
            for code, node in zip(codes, nodes):
                weight = self.node_weights[self.node_map[id(node)]]  # (D,)
                logit = torch.dot(center_emb, weight)
                label = torch.tensor(code, dtype=torch.float32, device=logit.device,requires_grad=False)
                # Use BCE with logits for stability
                loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), label.unsqueeze(0))
                total_loss += loss
                total_steps += 1

        return total_loss / total_steps

    
    def forward(self, center_ids, context_ids):
        """
        center_ids: LongTensor (B,)
        context_ids: LongTensor (B,)
        
        """
        # Input embedding
        center_embeds = self.item_embeddings(center_ids)     # (B, D)      

        # Hierarchical softmax loss        
        hsoftmax_loss = self.HierarchicalSoftmaxLoss(center_embeds, context_ids)

        # Multi-category cosine similarity loss
        batch_cat_ids=[]
        for i in range(center_ids.size(0)):
            path = []
            get_path(self.cat_tree, self.map.get_item(center_ids[i].item()), path)  
            path.pop(0) # remove item
            path = path[:-1] # remove root
            batch_cat_ids.append(path)
            
        cat_loss = 0.0
        if any(batch_cat_ids):
            cosine_losses = []
            for i, cat_ids in enumerate(batch_cat_ids):
                if not cat_ids:
                    continue

                cosine_losses.append(self.CosineLoss(center_embeds[i], cat_ids))
            if cosine_losses:
                cat_loss = torch.stack(cosine_losses).mean()

        total_loss = hsoftmax_loss + self.params.lambda_cat * cat_loss
        return total_loss


    def get_normalized_embeddings(self):
        with torch.no_grad():
            raw_embeddings = self.item_embeddings.weight  # shape: [num_items, dim_embedding]
            norm_embeddings = F.normalize(raw_embeddings, p=2, dim=1)  # unit vectors        
        return norm_embeddings
    

    def find_closest_k_items(self, test_ids, topk):
        """
        Hierarchical softmax doesn't store per-item target embeddings, so the item similarity 
        space lives in the input embedding matrix.
        
        """
        # Get normalized input item embedding
        embeddings = self.get_normalized_embeddings()
        
        # Get embeddings for test items
        test_vecs = embeddings[test_ids]  # [num_test_items, dim_embedding]

        # Compute cosine similarity: [num_test_items, num_items]
        similarities = torch.mm(test_vecs, embeddings.t())

        # Mask out self-similarity (set diagonal to -inf)
        for i, item_id in enumerate(test_ids):
            similarities[i, item_id] = -float('inf')
            
        # Get top-k item indices
        topk_dists, topk_ids = similarities.topk(topk)    
        
        print("\n-----------")  
        for i, id in enumerate(test_ids):
            print(str(self.map.get_item(id.item())) + " || ",end='')
            dists = [d.item() for d in topk_dists[i]][1:]
            topk_items = [self.map.get_item(k.item()) for k in topk_ids[i]][1:]
            for j, (w, sim) in enumerate(zip(topk_items,dists)):
                print(f"{w} ({sim:.3f})", end=' ')
            print('\n')
        print("-----------")        
        
        return 


class Trainer:
    def __init__(self, model: HierarchicalItem2Vec, params: Params, optimizer,
                 train_iter, valid_iter, map: ItemMap, method: BatchToolItem, debug=False):
        self.model = model
        model.debug = debug
        self.params = params
        self.optimizer = optimizer
        self.map = map
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.method = method

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}

        # sending all to device
        self.model.to(self.params.device)
        self.test_tokens = None


    def train(self):
        self.do_test()
        for epoch in range(self.params.n_epochs):
            # load data
            self.train_dataloader = DataLoader(
                self.train_iter,
                batch_size=self.params.batch_size,
                shuffle=True,
            )
            self.valid_dataloader = DataLoader(
                self.valid_iter,
                batch_size=self.params.batch_size,
                shuffle=False,
            )
            
            # train model
            st_time = monotonic()
            self._train_epoch(epoch)
            self.epoch_train_mins[epoch] = round((monotonic()-st_time)/60, 1)

            # validate model
            self._validate_epoch(epoch)
            print(f"""Epoch: {epoch+1}/{self.params.n_epochs}\n""",
            f"""    Train Loss: {self.loss['train'][-1]:.2}\n""",
            f"""    Valid Loss: {self.loss['valid'][-1]:.2}\n""",
            f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
            """\n"""
            )
           
            if self.params.checkpoint_frequency:
                self._save_checkpoint(epoch)

        self.do_test()
    
    def _train_epoch(self,_epoch):
        self.model.train()
        running_loss = []

        progress = tqdm(self.train_dataloader, desc=f"Epoch {_epoch+1}/{self.params.n_epochs}")
        for i, batch_data in enumerate(progress, 1):                    
            inputs_, outputs_ = batch_data[0], batch_data[1]

            flat_inputs = torch.flatten(inputs_).tolist()
            input_indices = self.map.get_item_index(flat_inputs)
            inputs = torch.tensor(input_indices)
            output_indices = self.map.get_item_index(torch.tensor(outputs_).tolist())
            outputs = torch.tensor(output_indices)               
            inputs, outputs = inputs.to(self.params.device), outputs.to(self.params.device)     
        
            self.optimizer.zero_grad()
            loss = self.model(inputs, outputs)

            if self.model.debug >1 :
                before = self.model.item_embeddings(inputs).clone()
            loss.backward()
            self.optimizer.step()
            if self.model.debug >1 :
                after = self.model.item_embeddings(inputs)
                print("Change:", (after - before).abs().mean())
                print(self.model.item_embeddings.weight.grad[inputs])
            
            running_loss.append(loss.item())
            progress.set_postfix(loss=loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss['train'].append(epoch_loss)

    def _validate_epoch(self,_epoch):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            progress = tqdm(self.valid_dataloader, desc=f"Epoch {_epoch+1}/{self.params.n_epochs}")
            for i, batch_data in enumerate(progress, 1): 
                inputs_, outputs_ = batch_data[0], batch_data[1]
                
                flat_inputs = torch.flatten(inputs_).tolist()
                input_indices = self.map.get_item_index(flat_inputs)
                inputs = torch.tensor(input_indices)
                output_indices = self.map.get_item_index(torch.tensor(outputs_).tolist())
                outputs = torch.tensor(output_indices)   
                inputs, outputs = inputs.to(self.params.device), outputs.to(self.params.device)
            
                loss = self.model(inputs, outputs)
                
                running_loss.append(loss.item())
                progress.set_postfix(loss=loss.item())
                
            epoch_loss = np.mean(running_loss)
            self.loss['valid'].append(epoch_loss)


    def do_test(self, topk: int = 5):

        sampling_window=100
        test_size = 10

        if self.test_tokens != None:
            ttokens= []
            for w in self.test_tokens:
                idw = self.map.get_item_index(w)
                ttokens.append(idw)
                if idw == self.map.discarded:
                    print(f"Item {w} not in Inventory List")
                    return
        else:
            ttokens= np.array(random.sample(range(sampling_window), test_size//2)) # high frequency tokens
            ttokens= np.append(ttokens,random.sample(range(1000,1000+sampling_window), test_size//2)) #low frequency tokens

        ttokens = torch.tensor(ttokens, dtype=torch.long).to(self.params.device)

        self.model.find_closest_k_items(ttokens, topk)
    

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.params.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.params.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.params.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.params.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
    

