from parameters import Params
from ItemMap import ItemMap

class BatchToolItem:
    def __init__(self, tokenmap: ItemMap, params: Params):
        self.map = tokenmap
        self.params = params
        #self.discard_probs = self.get_discard_probs()

    def frequency_from_percentile(self, percentile= 90):
        freq_list = []
        for _, (_, freq) in self.map.by_token.items():
            if freq == freq:
                freq_list.append(freq/self.map.total_tokens)
            
        return np.percentile(freq_list, percentile)

    def get_discard_probs(self):
        discard_probs = {}
        for _, (word, freq) in self.map.by_token.items():
            prob_raw = 1-np.sqrt(self.params.threshold /(freq/self.map.total_tokens))
            prob = max(prob_raw,0)
            discard_probs[word] = prob
        return discard_probs

    def collate_fn(self, batches):

        # lengh of dataset 
        print(len(batches)) # batch[0] = ([input_id], output_id)

#        for batch_X, batch_y in batches:
#            print("Batch X:", batch_X)
#            print("Batch y:", batch_y)        
                
        #  [ discard low frequency items ] - we want cold items included.  
        #discard_id = self.map.discard_id
            #    p = random.random()
            #    p_discard = self.discard_probs.get(target_id)        
            #    if p_discard >= p or target_id == discard_id:
            #        continue
                
            #    p = random.random()
            #    p_discard = self.discard_probs.get(context_id)
            #    if p_discard >= p or context_id == discard_id:
            #        continue

            #    inputs.append(target_id)
            #    outputs.append(context_id)

        #torch_input = torch.tensor(inputs, dtype=torch.long)
        #torch_output = torch.tensor(outputs, dtype=torch.long)
        # new_batches = TensorDataset(torch_input, torch_output)
        
        return batches