import torch
from torch import nn

from models.model.encoder import Encoder


class Classify(nn.Module):

    def __init__(self, enc_voc_size,n_class, d_model, n_head, max_len,
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
        
        self.classify_layer = nn.Linear(low_dim,n_class)

    def encode(self, src, src_mask):

        # embedding,mu,log_var = self.encoder(src, src_mask)
        # return embedding, mu, log_var
        embedding = self.encoder(src, src_mask)

        return embedding

    def classify(self,emb):

        #emb: [batch_size x low_dim]
        #return: [batch_size x n_class]

        #return self.classify_layer(emb)  
        return self.classify_layer(emb)
    
    def forward(self, src, src_mask):

        embedding = self.encoder(src, src_mask)
        embedding = embedding[:,0]
        output = self.classify(embedding)
        return output,embedding  #out:(batch_size,seq_len,pos_dim)

        # return embedding #for retrieval
