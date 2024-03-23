from torch import nn
import math
import torch
# from strokeTransformer.models.embedding.positional_encoding import PostionalEncoding
from models.embedding.positional_encoding import PostionalEncoding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Linear(vocab_size,d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len + 1, device) # for classification
        # self.pos_emb = PostionalEncoding(d_model, max_len, device) # for reconstruction
        self.drop_out = nn.Dropout(p=drop_prob)
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # for classification
    def forward(self, x):
        # for reconstruction
        '''
        tok_emb = self.tok_emb(x) * math.sqrt(self.d_model)
        
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
        '''

    
        # for classification
        x = self.tok_emb(x) * math.sqrt(self.d_model)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b,1,1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_emb = self.pos_emb(x)
        return self.drop_out(x + pos_emb)
