import torch
from torch import nn

# from segmentFormer.models.model.decoder import Decoder
# from segmentFormer.models.model.encoder import Encoder

from strokeFormer.segmentFormer.models.model.decoder import Decoder
from strokeFormer.segmentFormer.models.model.encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device,low_dim):
        super().__init__()
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               low_dim = low_dim,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               low_dim = low_dim,
                               device=device)
    def encode(self, src, src_mask):

        # embedding,mu,log_var = self.encoder(src, src_mask)
        # return embedding, mu, log_var
        embedding = self.encoder(src, src_mask)

        return embedding

    
    def decode(self, tgt, embedding, tgt_mask, dec_padding_mask):

        output = self.decoder(tgt, embedding, tgt_mask, dec_padding_mask)

        return output

    
    def forward(self, src, tgt, src_mask, tgt_mask, dec_padding_mask):

        # embedding,mu,log_var = self.encoder(src, src_mask)
        # output = self.decoder(tgt, embedding, tgt_mask, dec_padding_mask)
        # return output,mu,log_var  #out:(batch_size,seq_len,pos_dim)
        embedding = self.encoder(src, src_mask)
        output = self.decoder(tgt, embedding, tgt_mask, dec_padding_mask)
        return output  #out:(batch_size,seq_len,pos_dim)
