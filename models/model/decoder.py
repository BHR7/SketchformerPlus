import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.layers.denseExpander import DenseExpander

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, low_dim ,drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

        self.denseExpander = DenseExpander(max_len,d_model,low_dim)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        pre_decoder = self.denseExpander(enc_src)
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, pre_decoder, trg_mask, src_mask)

        # pass to LM head
        
        output = self.linear(trg)
        return output
