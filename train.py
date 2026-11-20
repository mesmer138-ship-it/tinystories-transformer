#Training Loop
import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils import set_seed
from src.model import TinyLM
from src.tokenizer import CharTokenizer
from src.data import load_stories, split_stories, encode_stories, get_batch
from config import Config
cfg = Config()

set_seed(cfg.seed)
data = load_stories(cfg.data_slice)
train_stories,val_stories = split_stories(data,cfg.val_fraction,cfg.seed)
tok = CharTokenizer(train_stories)
train_ids = encode_stories(train_stories,tok)
val_ids =  encode_stories(val_stories,tok)

writer = SummaryWriter(f"{cfg.log_dir}/{cfg.run_name}")
writer.add_text("config",str(cfg))
tiny_lm = TinyLM(tok.vocab_size,cfg.n_embd,cfg.n_heads,cfg.n_layer,cfg.block_size)
optimizer = torch.optim.AdamW(tiny_lm.parameters(),lr=cfg.learning_rate)

for num in range(cfg.max_steps):
  x,y = get_batch(train_ids,cfg.batch_size,cfg.block_size)
  logits = tiny_lm(x)

  B,T,C = logits.shape
  logits_flat = logits.view(-1,C)
  y_flat = y.view(-1)

  loss = torch.nn.functional.cross_entropy(logits_flat,y_flat)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  

  if num % cfg.eval_interval == 0:

    val_total = 0
    
    with torch.no_grad():
      for _ in range(cfg.eval_iters):
        v_x,v_y = get_batch(val_ids,cfg.batch_size,cfg.block_size)
        v_logits = tiny_lm(v_x)
        v_B,v_T,v_C = v_logits.shape
        v_logits_flat = v_logits.view(-1,v_C)
        v_y_flat = v_y.view(-1)

        val_loss = torch.nn.functional.cross_entropy(v_logits_flat,v_y_flat)
        val_total += val_loss.item()
      
    val_average_loss = val_total/cfg.eval_iters
    print(f"step {num}: train: {loss.item():.3f}, val: {val_average_loss:.3f}")
    writer.add_scalar("loss/train",loss.item(),num)
    writer.add_scalar("loss/val",val_average_loss,num)
writer.close()
