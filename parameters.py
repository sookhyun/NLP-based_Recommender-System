
from dataclasses import dataclass, field
import torch

@dataclass
class Params:

    # skipgram parameters
    dim_embedding = 300
    max_norm_embedding = None
    min_freq = 1 #50    
    threshold = 1.0e-2 # 1.0e-5 subsampling threshold
    window_size = 4 # context window one-side length
    n_neg_samples = 5 
    neg_exponent = 0.75
    discarded = "<>" # special characters or below threshold frequency
    tokenizer = 'basic_english'

    lambda_cat =0.1
    
    # training parameters
    batch_size : int = 10
    criterion = None
    shuffle = True
    learning_rate = 5e-4
    n_epochs = 5
    train_steps = 1
    val_steps = 1
    checkpoint_frequency = 1

    model_name = 'SkipGram'
    model_dir = "weights/{}".format(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 