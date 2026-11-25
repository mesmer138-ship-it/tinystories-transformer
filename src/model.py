import torch
import math
class Head(torch.nn.Module):
  def __init__(self,head_size,block_size,n_embd,scale_attn):
      super().__init__()
      self.query = torch.nn.Linear(n_embd,head_size,bias=False)
      self.key = torch.nn.Linear(n_embd,head_size,bias=False)
      self.value = torch.nn.Linear(n_embd,head_size,bias=False)
      self.head_size = head_size
      mask = torch.ones((block_size,block_size))
      tril = torch.tril(mask)
      self.register_buffer("tril",tril)
      self.scale_attn = scale_attn
  def forward(self,x):
      q = self.query(x)
      k = self.key(x)
      v = self.value(x)
      scores = q@k.transpose(-1,-2)
      T = x.shape[-2]
      if self.scale_attn:
        pre_mask = scores/math.sqrt(self.head_size) 
      else:
        pre_mask = scores
      self.scores = pre_mask
      masked_scores = pre_mask.masked_fill(self.tril[:T,:T]==0,float('-inf'))
      softmax_result = torch.softmax(masked_scores,dim=-1)
      self.weights = softmax_result
      return(softmax_result @ v)

class MultiHeadAttention(torch.nn.Module):
  def __init__(self,n_heads,n_embd,block_size,scale_attn):
      super().__init__()
      head_size = n_embd // n_heads
      self.heads = torch.nn.ModuleList([Head(head_size,block_size,n_embd,scale_attn) for _ in range(n_heads)])
      self.proj = torch.nn.Linear(n_embd,n_embd,bias=True)
  def forward(self,x):
      outputs = [h(x) for h in self.heads]
      concat = torch.cat(outputs,dim=-1)
      return self.proj(concat)
    
class MLP(torch.nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.layer = torch.nn.Linear(n_embd,4*n_embd,bias=True)
    self.relu = torch.nn.ReLU()
    self.second_layer = torch.nn.Linear(4*n_embd,n_embd,bias=True)
  def forward(self,x):
    hidden = self.layer(x)
    hidden_relu = self.relu(hidden)
    mlp_output = self.second_layer(hidden_relu)
    return mlp_output
      
class Block(torch.nn.Module):
  def __init__(self,n_heads,n_embd,block_size,scale_attn):
    super().__init__()
    self.ln1 = torch.nn.LayerNorm(n_embd)
    self.attn = MultiHeadAttention(n_heads,n_embd,block_size,scale_attn)
    self.ln2 = torch.nn.LayerNorm(n_embd)
    self.mlp = MLP(n_embd)
  def forward(self,x):
    x = x+self.attn(self.ln1(x))
    x = x+self.mlp(self.ln2(x))
    return x
class TinyLM(torch.nn.Module):
  def __init__(self,vocab_size,n_embd,n_heads,n_layer,block_size,scale_attn):
    super().__init__()
    self.token_emb = torch.nn.Embedding(vocab_size,n_embd)
    self.pos_emb = torch.nn.Embedding(block_size,n_embd)
    self.blocks = torch.nn.Sequential(*[Block(n_heads,n_embd,block_size,scale_attn) for _ in range(n_layer)])
    self.ln_f = torch.nn.LayerNorm(n_embd)
    self.lm_head = torch.nn.Linear(n_embd,vocab_size)
    self.block_size = block_size

  def forward(self,idx):
    B,T = idx.shape
    tok = self.token_emb(idx)
    pos = self.pos_emb(torch.arange(T,device=idx.device))
    x = tok + pos
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)
    return logits
    
  def generate(self,idx,max_new_tokens):
    with torch.no_grad():
      for _ in range(max_new_tokens):
        idx_copy = idx[:,-(self.block_size):]
        logits = self(idx_copy)
        softmax_result = torch.softmax(logits[:,-1,:],dim=-1)
        next_id = torch.multinomial(softmax_result,num_samples=1)
        idx = torch.cat([idx,next_id],dim=1)
      return idx
        
        
        






    

