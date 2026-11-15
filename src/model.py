import torch
import math
class Head(torch.nn.Module):
  def __init__(self,head_size,block_size,n_embd):
      super().__init__()
      self.query = torch.nn.Linear(n_embd,head_size,bias=False)
      self.key = torch.nn.Linear(n_embd,head_size,bias=False)
      self.value = torch.nn.Linear(n_embd,head_size,bias=False)
      self.head_size = head_size
      mask = torch.ones((block_size,block_size))
      tril = torch.tril(mask)
      self.register_buffer("tril",tril)
  def forward(self,x):
      q = self.query(x)
      k = self.key(x)
      v = self.value(x)
      scores = q@k.transpose(-1,-2)
      scaled_scores = scores/math.sqrt(self.head_size)
      T = x.shape[-2]
      masked_scores = scaled_scores.masked_fill(self.tril[:T,:T]==0,float('-inf'))
      softmax_result = torch.softmax(masked_scores,dim=-1)
      return(softmax_result @ v)

class MultiHeadAttention(torch.nn.Module):
  def __init__(self,n_heads,n_embd,block_size):
      super().__init__()
      head_size = n_embd // n_heads
      self.heads = torch.nn.ModuleList([Head(head_size,block_size,n_embd) for x in range(n_heads)])
      self.proj = torch.nn.Linear(n_embd,n_embd,bias=True)
  def forward(self,x):
      outputs = [h(x) for h in self.heads]
      concat = torch.cat(outputs,dim=-1)
      return self.proj(concat)
      
      
      


