from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

EOS = 0
SOS = 1
MASK = 2
NUM_SPECIAL = 3


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        soft = F.softmax(sorted_logits, dim=-1)
        # print('soft vector: ', soft)
        cumulative_probs = torch.cumsum(soft, dim=-1)
        # print('cummulative sum: ', cumulative_probs)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self,
               config,
               pix_len,
               bdry_len,
              #  num_text_token,
               word_emb_path=None,
               pos_emb_path=None):
    """
    Initializes FaceModel.
    """
    super(SketchDecoder, self).__init__()
    self.pix_len = pix_len
    self.embed_dim = config['embed_dim']

    self.bdry_len = bdry_len
    # self.num_text_token = num_text_token
    # self.num_image_token = BBOX * BBOX + PIX_PAD + SVG_END
    self.num_bdry_token = 256
    self.num_image_token = self.num_bdry_token + 4096 + NUM_SPECIAL
    # self.total_token = num_text_token + self.num_image_token
    self.total_token = self.num_image_token
    self.total_seq_len = bdry_len + pix_len
    # self.total_seq_len = pix_len
    self.loss_img_weight = 3

    seq_range = torch.arange(self.total_seq_len)
    logits_range = torch.arange(self.total_token)

    seq_range = rearrange(seq_range, 'n -> () n ()')    # 1 * 512 * 1
    logits_range = rearrange(logits_range, 'd -> () () d')    # 1 * 1 * 6xxxx

    # logits_mask = (
    #     ((seq_range >= bdry_len) & (logits_range <= self.num_image_token)) |
    #     ((seq_range < bdry_len) & (logits_range <= self.num_image_token))
    # )

    # self.register_buffer('logits_mask', logits_mask, persistent=False)
    # Sketch encoders

    self.pixel_embed = Embedder(self.num_image_token, self.embed_dim, padding_idx=MASK)
    self.bdry_embed = Embedder(self.num_bdry_token, self.embed_dim, padding_idx=MASK)
    self.pos_embed = PositionalEncoding(max_len=self.total_seq_len, d_model=self.embed_dim)
    self.logit_fc = nn.Linear(self.embed_dim, self.total_token)
    
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'], nhead=config['num_heads'], dropout=config['dropout_rate'])
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)

    # assert word_emb_path is not None, 'text_emb_dir must be provided'
    # if word_emb_path is not None:
    #   self.text_emb = nn.Embedding.from_pretrained(torch.load(word_emb_path, map_location='cpu'))
    #   assert self.embed_dim == self.text_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'
    # if pos_emb_path is not None:
    #   self.text_pos_emb = nn.Embedding.from_pretrained(torch.load(pos_emb_path, map_location='cpu'))
    #   assert self.embed_dim == self.text_pos_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'


  def forward(self, pix, mask, bdry, bdry_mask, return_loss=False):
    '''
    pix.shape  [batch_size, max_len]
    xy.shape   [batch_size, max_len, 2]
    mask.shape [batch_size, max_len]
    text.shape [batch_size, text_len]
    '''
    pixel_v = pix[:, :-1] if return_loss else pix
    pixel_mask = mask[:, :-1] if return_loss else mask
    device = torch.device("cuda:0")
    bs, c_seqlen, device = bdry.shape[0], bdry.shape[1], bdry.device
    
    bdry_embed = self.bdry_embed(bdry)
    tokens = bdry_embed   # [bs, len_bdry, dim]
    # print(tokens.shape)
    context_embedding = torch.zeros((1, bs, self.embed_dim)).to(device) # [1, bs, dim]

    if pixel_v[0] is not None:
      c_seqlen += pixel_v.shape[1]  
    
      bs = pixel_v.shape[0]
      # Context embedding values
      

      # tokens.shape [batch_size, text_len, emb_dim]
      # tokens = self.text_emb(text)
      

      # Data input embedding
      pixel_embed = self.pixel_embed(pixel_v)
      embed_inputs = pixel_embed

      # tokens.shape [batch_size, bdry_len+max_len-1, emb_dim]
      tokens = torch.cat((tokens, embed_inputs), dim=1)


    embeddings = torch.cat([context_embedding, tokens.transpose(0,1)], axis=0)
    decoder_inputs = self.pos_embed(embeddings) 

    memory_encode = torch.zeros((1, bs, self.embed_dim)).to(device)
    
    # nopeak_mask.shape [c_seqlen+1, c_seqlen+1]
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(c_seqlen+1).to(device)  # masked with -inf
    if pixel_mask is not None:
      # pixel_mask.shape [batch_size, text_len+max_len]
      # TODO: text mask?
      # In DALLE, each padding in text is assignedw with a new uesless embedding
      pixel_mask = torch.cat([(torch.zeros([bs, context_embedding.shape[0]+self.bdry_len])==1).to(device), pixel_mask], axis=1)  

    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory_encode, memory_key_padding_mask=None,
                               tgt_mask=nopeak_mask, tgt_key_padding_mask=pixel_mask)

    # Logits fc
    logits = self.logit_fc(decoder_out)  # [seqlen, bs, dim] 
    logits = logits.transpose(1,0)  # [bs, textlen+seqlen, total_token] 

    # logits_mask = self.logits_mask[:, :c_seqlen+1]
    # max_neg_value = -torch.finfo(logits.dtype).max
    # logits.masked_fill_(logits_mask, max_neg_value)

    if return_loss:
      logits = rearrange(logits, 'b n c -> b c n')
      bdry_logits = logits[:, :, :self.bdry_len]
      pix_logits = logits[:, :, self.bdry_len:]

      pix_logits = rearrange(pix_logits, 'b c n -> (b n) c')
      pix_mask = ~mask.reshape(-1)
      pix_target = pix.reshape(-1) # + self.num_text_token

      bdry_loss = F.cross_entropy(bdry_logits, bdry.to(torch.int64))
      pix_loss = F.cross_entropy(pix_logits[pix_mask], pix_target[pix_mask].to(torch.int64), ignore_index=MASK)
      loss = (bdry_loss + self.loss_img_weight * pix_loss) / (self.loss_img_weight + 1)
      return loss, pix_loss, bdry_loss
    else:
      return logits
    
  def sample(self, n_samples, bdry, pixel_seq=None):
    """ sample from distribution (top-k, top-p) """
    pix_samples = []
    xy_samples = []
    top_k = 0
    top_p = 0.5

    # Sample per token
    # bdry = bdry[:, :self.bdry_len]
    pixlen = 0 if pixel_seq is None else pixel_seq.shape[1]
    for k in range(bdry.shape[1]+pixlen, self.total_seq_len):
      if k == bdry.shape[1]:
        pixel_seq = [None] * n_samples
      
      # pass through model
      with torch.no_grad():
        p_pred = self.forward(pixel_seq, None, bdry, None)
        p_logits = p_pred[:, -1, :]

      next_pixels = []
      # Top-p sampling of next pixel
      for logit in p_logits: 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        # next_pixel -= self.num_text_token
        next_pixels.append(next_pixel.item())
        # print(next_pixel.item())

      # Convert pixel index to xy coordinate
      # next_xys = []
      # for pixel in next_pixels:
      #   if pixel >= NUM_SPECIAL:
      #     xy = self.pixel2xy[pixel-NUM_SPECIAL]
      #   else:
      #     xy = np.array([pixel, pixel]).astype(int)
      #   next_xys.append(xy)
      # next_xys = np.vstack(next_xys)  # [BS, 2]
      next_pixels = np.vstack(next_pixels)  # [BS, 1]
        
      # Add next tokens
      nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
      # nextxy_seq = torch.LongTensor(next_xys).unsqueeze(1).cuda()
      
      if pixel_seq[0] is None:
        pixel_seq = nextp_seq
        # xy_seq = nextxy_seq
      else:
        pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
        # xy_seq = torch.cat([xy_seq, nextxy_seq], 1)
      # print(pixel_seq.shape)
      if pixel_seq.shape[1] == self.pix_len:
        seq = pixel_seq[0, :]
        seq = seq.tolist()
        # print(seq)
        # xys = []
        # for pixel in seq:
        #   if pixel >= NUM_SPECIAL:
        #     xy = self.pixel2xy[pixel-NUM_SPECIAL]
        #   else:
        #     xy = np.array([pixel, pixel]).astype(int)
        #   xys.append(xy)
        
        # xy_samples = xys
        break 

      # print(pixel_seq)
      # Early stopping
      done_idx = np.where(next_pixels==0)[0]
      if len(done_idx) > 0:
        done_pixs = pixel_seq[done_idx] 
        done_xys = xy_seq[done_idx]
        # done_ext = latent_ext[done_idx]
       
        # for pix, xy, ext in zip(done_pixs, done_xys, done_ext):
        for pix, xy in zip(done_pixs, done_xys):
          pix = pix.detach().cpu().numpy()
          # xy = xy.detach().cpu().numpy()
          pix_samples.append(pix)
          # xy_samples.append(xy)
          # latent_ext_samples.append(ext.unsqueeze(0))
  
      left_idx = np.where(next_pixels!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        pixel_seq = pixel_seq[left_idx]
        # xy_seq = xy_seq[left_idx]
        # text = text[left_idx]

    # return pix_samples, latent_ext_samples
    return pix_samples

