import random
from datasets import load_dataset
import torch

def split_stories(stories, val_fraction, seed):
    stories_copy = list(stories)
    rng = random.Random(seed)
    rng.shuffle(stories_copy)
    train_stories = stories_copy[:int(len(stories_copy)*(1-val_fraction))]
    val_stories = stories_copy[len(train_stories):]
    return train_stories, val_stories 
    
def load_stories(data_slice):
    ds = load_dataset("roneneldan/TinyStories", split=data_slice)
    return ds['text']
    
def encode_stories(stories,tok):
    text = "\n\n".join(stories)
    stories_ids = tok.encode(text)
    stories_tensor = torch.tensor(stories_ids,dtype=torch.long)
    return stories_tensor
    
def get_batch(data,batch_size,block_size):
    starts = torch.randint(0,len(data)-block_size,(batch_size,))
    x_list = []
    y_list = []
    for i in starts:
      x_block = data[i:i+block_size]
      y_block = data[i+1:i+block_size+1]
      x_list.append(x_block)
      y_list.append(y_block)
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x, y
