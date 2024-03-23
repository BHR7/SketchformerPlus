import os
import torch
import sys
import numpy as np
import torch.nn.functional as F
from utils.util import get_scheduler,mkdir,create_masks,create_padding_mask

from models.model.transformer import Transformer
from models.model.classify import Classify
from copy import deepcopy
from utils.schedulers import Scheduler

class SketchModel:
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.seq_len = opt.sketch_len
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir,opt.dataset, opt.timestamp)
        #self.pretrain_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.pretrain)
        self.optimizer = None
        self.loss_func1 = None
        self.loss_func2 = None
        self.loss_func3 = None

        self.loss = None
        #self.loss1 = None
        self.loss2 = None
        self.loss3 = None
        self.out = None
        self.confusion = None # confusion matrix
        self.multi_confusion = None

        self.net_name = opt.net_name


        self.build_model(opt)
        

    def build_model(self,opt):
        if(opt.do_reconstruction):
            print('do_reconstruction.........')
            self.model = Transformer(enc_voc_size=opt.src_vocab,dec_voc_size=opt.tgt_vocab,d_model=opt.d_model,
                                    max_len=opt.sketch_len,ffn_hidden=opt.dim_feedforward,n_head=opt.nhead,n_layers=opt.num_layers,
                                    drop_prob=opt.dropout,device=self.device,low_dim=opt.low_dim).to(self.device)

            self.model.train(self.is_train)

            self.loss_func1 = torch.nn.MSELoss(reduction='none')
            self.loss_func2 = torch.nn.CrossEntropyLoss()
            self.loss_func3 = torch.nn.L1Loss(reduction='none')
        elif(opt.do_classification):
            
            print('do_classificaton.........')
            self.model = Classify(enc_voc_size=opt.src_vocab,d_model=opt.d_model,n_class = opt.n_class,
                                    max_len=opt.sketch_len,ffn_hidden=opt.dim_feedforward,n_head=opt.nhead,n_layers=opt.num_layers,
                                    drop_prob=opt.dropout,device=self.device,low_dim=opt.low_dim).to(self.device)
            self.loss_func2 = torch.nn.CrossEntropyLoss()
        else:
            print('error........')

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=opt.learning_rate, 
                                              betas=(opt.beta1, 0.999))

            self.scheduler = get_scheduler(self.optimizer, opt)
            
        if not self.is_train: 
            self.load_network(opt.which_epoch, mode='test')


    def forward(self,src,tgt,src_mask,dec_padding_mask,tgt_mask):
        '''
        for reconstruction
        '''
        output = self.model(src=src,tgt=tgt,src_mask=src_mask,tgt_mask=tgt_mask,dec_padding_mask=dec_padding_mask)
        return output

    def forward2(self,src,src_mask):
        '''
        for classification
        '''
        output,embedding = self.model(src=src,src_mask=src_mask)
        return output,embedding  #[batch_size x n_classes]

    def backward(self, out, gt,loss_compute):
        """
        gt: (B*N, 2)
        out: (B,N,2)
        """
        #self.loss, self.output = loss_compute(out,gt,mu,log_var)
        self.loss, self.output = loss_compute(out,gt)

    def step(self,inp,tar,label):
        """
        """
        self.model.train()

        if(self.opt.do_reconstruction):
            tar_inp = tar[:, :-1, ...]
            tar_real = tar[:, 1:, ...]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            dec_padding_mask = torch.ones_like(dec_padding_mask)

            out  = self.forward(inp.to(self.device),tar_inp.to(self.device),enc_padding_mask.to(self.device),dec_padding_mask.to(self.device),combined_mask.to(self.device))
            self.backward(out, tar_real.to(self.device),LossCompute(self.loss_func1,self.loss_func2,self.loss_func3,self.optimizer,self.device,self.opt.do_reconstruction))
        
        elif(self.opt.do_classification):

            cls_token = torch.zeros((1,1,258))
            cls_token = cls_token.repeat(inp.shape[0],1,1)
            input = torch.cat((cls_token, inp), dim=1)
            
            enc_padding_mask,_,_ = create_masks(input,inp)
            preds = self.forward2(inp.to(self.device),enc_padding_mask.to(self.device))
            self.backward(preds,label.to(self.device),LossCompute(self.loss_func1,self.loss_func2,self.loss_func3,self.optimizer,self.device,self.opt.do_reconstruction))

    def encode(self, inp, inp_mask):  
        #for reconstruction test
        embedding = self.model.encode(inp,inp_mask)
        return embedding

    def decode(self, embedding, target, target_mask, look_ahead_mask):  
        #for reconstruction test
        dec_output = self.model.decode(target, embedding, look_ahead_mask, target_mask)

        return dec_output

    def encode_from_seq(self, inp_seq):
        #for reconstruction test
        encoder_input = torch.FloatTensor(inp_seq.detach().cpu().numpy() + np.zeros((1, 1)))  # why?
        enc_padding_mask = create_padding_mask(encoder_input).to(self.device)
        res = self.encode(encoder_input.to(self.device), enc_padding_mask.to(self.device))
        return res
    
    def make_dummy_input(self, nattn, batch_size):
        #for reconstruction test
        nignore = self.seq_len - nattn

        l = torch.zeros((1,self.opt.tgt_vocab)).squeeze()
        r = torch.zeros((1,self.opt.tgt_vocab)).squeeze()
        r[-1] = 1

        dummy = torch.cat((torch.ones((batch_size, nattn, 258)) * torch.tensor(l,dtype=torch.float32),
            torch.ones((batch_size, nignore, 258)) * torch.tensor(r,dtype=torch.float32)
        ), axis=1)


        return dummy
    
    def predict_from_embedding(self,emb):
        #sketchEmbedding
        #for reconstruction test
        with torch.no_grad():
            outputSize = self.opt.tgt_vocab-2
            emb = torch.as_tensor(emb + torch.zeros(1,1).to(self.device),dtype=torch.float32)
            decode_input = torch.zeros((1,self.opt.tgt_vocab)).squeeze()  #1xtgt_vocab
            decode_input[-2] = 1
            decode_input = torch.as_tensor(decode_input,dtype=torch.float32)
            output = (torch.ones((emb.shape[0],1,self.opt.tgt_vocab)) * decode_input).to(self.device)

            
            for i in range(self.seq_len-1):
                nattn =  i + 1
                
                enc_input_dummy = self.make_dummy_input(nattn=nattn, batch_size=emb.shape[0])

                #enc_padding_mask = create_padding_mask(enc_input_dummy)
                enc_padding_mask2, tgt_mask, dec_padding_mask = create_masks(enc_input_dummy, output)

                dec_padding_mask = torch.ones_like(dec_padding_mask)
                out = self.decode(emb,output.to(self.device), dec_padding_mask.to(self.device),tgt_mask.to(self.device))
                
                predict = out[:, -1:, ...]  # (batch_size, 1, vocab_size)
                #print(predict.shape)
                predict = torch.cat((predict[...,:outputSize],torch.softmax(predict[...,outputSize:],dim=-1)),dim=-1).to(self.device)
                
                finished_ones = torch.sum((torch.argmax(predict[...,outputSize:],dim=-1) == 1).long())
                output = torch.cat([output,predict],dim=1).to(self.device)
                if finished_ones == emb.shape[0]:
                        break

        #output = embedding2Sketch(reconSketch=output,strokeDecoder=self.strokeDecoder)
        return output
    
    def predict(self,src,label):
        """
        x: (B*N, F)
        """
        self.model.eval()

        if(self.opt.do_reconstruction):
            embedding = self.encode_from_seq(src.to(self.device))
            return_embedding = embedding

            dec_output = self.predict_from_embedding(embedding)

            return dec_output,return_embedding
        elif(self.opt.do_classification):
            cls_token = torch.zeros((1,1,258))
            cls_token = cls_token.repeat(src.shape[0],1,1)
            input = torch.cat((cls_token, src), dim=1)
            enc_padding_mask,_,_ = create_masks(input,src)
            preds,embedding = self.forward2(src.to(self.device),enc_padding_mask.to(self.device))

            #preds = torch.argmax(F.softmax(preds,dim=-1),dim=-1)
            preds = F.softmax(preds,dim=-1)
            return preds,embedding

            
    
    def print_detail(self):
        print(self.model)


    def update_learning_rate(self):
        """
        update learning rate (called once every epoch)
        """
        self.scheduler.step()
        # lr = self.optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def save_network(self, epoch):
        """
        save model to disk
        """
        mkdir(self.save_dir)

        if(self.opt.do_reconstruction):
            Experiment = 'reconstruction'
        else:
            Experiment = 'classification'

        model_path = os.path.join(self.save_dir, '{}_transformer_{}_{}.pkl'.format(self.net_name, Experiment,epoch))
        
        torch.save(self.model.state_dict(), model_path)
        
        print('saving the model from {}'.format(model_path))

    
    def load_network(self, epoch, mode='test'):
        """
        load model from disk
        """
        #model_path = os.path.join('strokeTransformer/',self.save_dir, '{}_transformer_{}.pkl'.format(self.net_name, epoch))

        if(self.opt.do_reconstruction):
            Experiment = 'reconstruction'
        else:
            Experiment = 'classification'
        model_path = os.path.join(self.save_dir, '{}_transformer_{}_{}.pkl'.format(self.net_name,Experiment, epoch))

        print('loading the model from {}'.format(model_path))

        self.model.load_state_dict(torch.load(model_path))
    

class LossCompute:
    """loss compute"""

    def __init__(self,criterion1,criterion2,criterion3,opt=None,device=None,do_reconstruction=True):
        self.criterion1 = criterion1
        self.criterion2 = criterion2 #CrossEntropyLoss()
        self.criterion3 = criterion3
        self.opt = opt
        self.device = device
        self.do_reconstruction = do_reconstruction

    def __call__(self, x, y):

        if(self.do_reconstruction):

            outputSize = self.opt.tgt_vocab-2
            #KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
            mask = torch.logical_not(np.equal(y[..., -1].cpu(), 1))
            mask = mask.unsqueeze(-1).to(self.device)


            pred_locations = x[:, :, :outputSize]
            pred_metadata = x[:, :, outputSize:]
            tgt_locations = y[:, :, :outputSize]
            tgt_metadata = y[:, :, outputSize:]

            location_loss = self.criterion1(pred_locations,tgt_locations)  #[B,1]  MSE LOSS

            location_loss2 = self.criterion3(pred_locations,tgt_locations)
            
            tgt_metadata = torch.argmax(tgt_metadata,dim=-1).long()
            tgt_metadata = tgt_metadata.contiguous().view(-1)
            pred_metadata = pred_metadata.contiguous().view(-1, pred_metadata.size(-1))

            metadata_loss = self.criterion2(pred_metadata,tgt_metadata)    #CrossEntrop loss

            loss = location_loss + 0.5 * location_loss2 + metadata_loss
            loss *= mask
            loss = torch.mean(loss)

            #loss = 0.5 * KLD + loss
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()
            return loss,x
        else:
            loss = self.criterion2(x,y)
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()
            return loss,torch.argmax(F.softmax(x),dim=-1)