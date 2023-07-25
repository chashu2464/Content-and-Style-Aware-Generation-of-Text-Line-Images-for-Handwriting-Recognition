import torch
from torch import nn
import numpy as np
from resnet import GenModel_FC, WriterClaModel, DisModel, RecModel
from parameters import *
from loss import recon_criterion, crit, log_softmax, kl_divergence_loss
from helper import pad_str, encoding
from encoder_vgg import Encoder

# defining the learning rate


class Collective_train(nn.Module):
    """
        Parameters
            num_writers (int)= Total Number of writer int the dataset
            Show_iter_num (int)= Show total number of iteration
            train_set= Train dataset one batch at a time one sample contain the tuple of the Image and List of Label, List of label contains the Label Tag and Writer-id
            epoch= Number of epoch
            mode= Mode is use to define which models takes the input, perform function , generate loss and optimized

    """
    def __init__(self, num_writers, show_iter_num):

        super(Collective_train, self).__init__()
        self.generative = GenModel_FC().to(device)
        self.writer_Model = WriterClaModel(num_writers).to(device)
        self.discri = DisModel().to(device)
        self.recog = RecModel().to(device)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        

    def forward(self, train_set, epoch, mode, cer_func=None):
        for i in range(100):

            tr_img, tr_label = train_set
            tr_label, writer_id = tr_label[0], tr_label[1]
            label = pad_str(tr_label)
           
            stio_ten = encoding(label=label, decoder=encoder)
            tr_label = torch.stack(stio_ten, dim=0)
            tr_label = tr_label.to(device)
            tr_img = tr_img.to(device).unsqueeze(0).to(device)

            if mode == "dis_update":
                """
                    Here the part of discriminative model, calclate the loss on the real Image by and then generate an other image which should be passed the decoder to get textual touch
                    So we have generated Image and its look can be calculated loss from the generated Image at the end losses  are combained.

                """
                sample_img = tr_img.requires_grad_()
                real_loss = self.discri.calc_dis_real_loss(sample_img)
                real_loss.backward()
                with torch.no_grad():
                    sr_img = tr_img.requires_grad_()
                    f_xs = self.generative.enc_image(sr_img)  # b,512,8,27
                    f_xt, f_embed = self.generative.enc_text(
                        tr_label
                    )  
                    xg = self.generative.generate(f_xs, f_embed)
                fake_loss = self.discri.calc_dis_fake_loss(xg.permute(1, 0, 2, 3))

                fake_loss.backward()
                total_loss = real_loss + fake_loss
                return total_loss
            elif mode == "cla_update":

                """ 
                    Calculate Writer loss by generated Image and then calculate the loss with respect to the writer Image style.
                    
                """ 
                tr_img_rec = tr_img.requires_grad_()
                f_xs = self.generative.enc_image(tr_img_rec)
                writer_loss = self.writer_Model(f_xs,writer_id)
                writer_loss.backward()
                return writer_loss

            elif mode == "rec_update":
                """
                    Recognizer Loss with respect with KL_divergence

                """
                tr_img_rec = tr_img.requires_grad_()
                output = self.recog(tr_img_rec, tr_label)
                divided = output.size(0) // tr_label.size(0)
                output = output.view(divided, -1)

                loss_kl = kl_divergence_loss(output, tr_label)
                loss_kl.backward()
                return loss_kl

            elif mode == "gen_update":
                
                """ 
                    Generated Loss
                     
                """ 
                self.iter_num += 1
                f_xs = self.generative.enc_image(tr_img)  # b,512,8,27
                f_xt, f_embed = self.generative.enc_text(tr_label)  # b,4096  b,512,8,27
                f_mix = self.generative.mix(f_xs, f_embed)
                xg = self.generative.generate(f_mix, f_embed)
                l_dis_ori = self.discri.calc_gen_loss(xg.permute(1, 0, 2, 3))
                return l_dis_ori
            
            
            elif mode == "eval":
                with torch.no_grad():
                    f_xs = self.generative.enc_image(tr_img)  # b,512,8,27
                    f_xt, f_embed = self.generative.enc_text(
                        tr_label
                    )  # b,4096  b,512,8,27
                    f_mix = self.generative.mix(f_xs, f_embed)
                    xg = self.generative.decode(f_mix, f_embed)
                    # print("coming into the the test case")
                    pred_xt = self.recog(
                        xg.view(
                            tr_img.size(0),
                            tr_img.size(1),
                            tr_img.size(2),
                            tr_img.size(3),
                        ),
                        tr_label,
                    )
                    self.iter_num += 1

                    """ Dis loss """
                    l_dis_ori = self.discri.calc_gen_loss(xg)
                    """ Rec loss """

                    divided = pred_xt.size(0) // tr_label.size(0)
                    pred_xt = pred_xt.view(divided, -1)

                    l_rec_ori = kl_divergence_loss(pred_xt, tr_label)

                    """Writer classification Loss """

                    l_cla_ori = self.writer_Model(xg, tr_img)
                return l_dis_ori, l_cla_ori, l_rec_ori
