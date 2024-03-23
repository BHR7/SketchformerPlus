from torch import nn

# from segmentFormer.models.blocks.encoder_layer import EncoderLayer
# from segmentFormer.models.embedding.transformer_embedding import TransformerEmbedding
# from segmentFormer.models.layers.bottleneck_layer import Bottleneck_layer
from strokeFormer.segmentFormer.models.blocks.encoder_layer import EncoderLayer
from strokeFormer.segmentFormer.models.embedding.transformer_embedding import TransformerEmbedding
from strokeFormer.segmentFormer.models.layers.bottleneck_layer import Bottleneck_layer


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob,low_dim,device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        
        self.bottleneck_layer = Bottleneck_layer(d_model=d_model,low_dim=low_dim)

    def forward(self, x, s_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        #sampled_z,mu,log_var = self.bottleneck_layer(x)
        #return sampled_z,mu,log_var
        sampled_z  = self.bottleneck_layer(x)

        #print("hello")
        return sampled_z #[batch_size x low_dim]
