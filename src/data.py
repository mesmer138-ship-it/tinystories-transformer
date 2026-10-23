import random
from datasets import load_dataset

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
