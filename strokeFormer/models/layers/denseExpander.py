import torch
import torch.nn as nn

class DenseExpander(nn.Module):
    def __init__(self,seq_len,d_model,low_dim):
        super(DenseExpander, self).__init__()
        self.expand_layer = nn.Linear(1,seq_len)
        #self.fc = nn.Linear(low_dim,d_model)
    def forward(self,x):
        #x = self.fc(x)
        x = x.unsqueeze(-1)  # (batch_size, feat_dim_out, 1)
        x = self.expand_layer(x)  # (batch_size, d_model, seq_len)
        x = x.permute(0,2,1)  # (batch_size, seq_len, feat_dim_out)
        #print(f'x.shape:{x.shape}')
        return x