from torch import nn
import torch


class Bottleneck_layer(nn.Module):
    def __init__(self,d_model,low_dim):
        super(Bottleneck_layer, self).__init__()
        self.fc1 = nn.Linear(d_model,d_model)
        self.fc2 = nn.Linear(d_model,1,bias=False)
        #self.fc3 = nn.Linear(d_model,low_dim)
        # self.fc3 = nn.Linear(d_model, low_dim)  # mu
        # self.fc4 = nn.Linear(d_model, low_dim)  # log_var 

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def forward(self,x):
        ui = torch.tanh(self.fc1(x)) #[B,L,d_model]
        ai = torch.softmax(self.fc2(ui),dim=1)  #[B,T,1]
        #print(f'ai.shape:{ai.shape}')
        enc_output = torch.sum(x * ai, axis=1, keepdims=False)
        #print(f'o.shape:{o.shape}')
        #return o,ai

        #enc_output  = torch.mean(x,dim=1,keepdim=False)
        # mu      = self.fc3(enc_output)
        # log_var = self.fc4(enc_output)

        # sampled_z = self.reparameterization(mu, log_var)

        # return sampled_z,mu,log_var  # [batch_size,d_model]
        #enc_output = self.fc3(enc_output)# [batch_size,low_dim]
        return enc_output