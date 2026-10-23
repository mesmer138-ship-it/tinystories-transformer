import random

def split_stories(stories, val_fraction, seed):
    stories_copy = list(stories)
    rng = random.Random(seed)
    rng.shuffle(stories_copy)
    train_stories = stories_copy[:int(len(stories_copy)*(1-val_fraction))]
    val_stories = stories_copy[len(train_stories):]
    return train_stories, val_stories 
    
def load_stories(data_slice):
    return cfg.data_slice
