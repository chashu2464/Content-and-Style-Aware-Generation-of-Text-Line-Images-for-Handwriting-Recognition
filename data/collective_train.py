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
    def __init__(self, num_writers, show_iter_num, oov):
        super(Collective_train, self).__init__()
        self.generative = GenModel_FC().to(device)
        self.writer_Model = WriterClaModel(num_writers).to(device)
        self.discri = DisModel().to(device)
        self.recog = RecModel().to(device)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov

    def forward(self, train_set, epoch, mode, cer_func=None):
        tr_img, tr_label = train_set
        tr_label = tr_label[0]
        label = pad_str(tr_label)
        # 111 = decoding(
        #     label=label, decoder=encoder
        # )  # (number_example, batch_size,text_len_max)
        stio_ten = encoding(label=label, decoder=encoder)
        tr_label = torch.stack(stio_ten, dim=0)
        tr_label = tr_label.to(device)
        tr_img = tr_img.to(device).unsqueeze(0).to(device)

        if mode == "dis_update":

            sample_img = tr_img.requires_grad_()
            real_loss = self.discri.calc_dis_real_loss(sample_img)
            real_loss.backward(retain_graph=True)
            with torch.no_grad():
                sr_img = tr_img.requires_grad_()
                f_xt, f_embed = self.generative.enc_text(tr_label)
                f_mix = self.generative.mix(sr_img, f_embed)
                #import pdb

                #pdb.set_trace()
                #f_mix_gen = self.generative.encoding(f_mix)
            #xg = self.generative.decode(f_mix_gen, tr_label)
            #output = self.recog(tr_img, tr_label)
            #import pdb

            #pdb.set_trace()
            fake_loss = self.discri.calc_dis_fake_loss(f_mix)
            fake_loss.backward()
            total_loss = real_loss + fake_loss
            return total_loss
        elif mode == "cla_update":
            """
                Generative Image take in the source image and outout f_xs which along
                with the source image is feed to the write calculate the difference btw the
                generated image and the source image. 
                
            """
            tr_img_rec = tr_img.requires_grad_()

            f_xs = self.generative.enc_image(tr_img_rec)
            writer_loss = self.writer_Model(f_xs, tr_img_rec)
            writer_loss.backward()
            return writer_loss

        elif mode == "rec_update":
            tr_img_rec = tr_img.requires_grad_()
            output = self.recog(tr_img_rec, tr_label)
            divided = output.size(0) // tr_label.size(0)
            output = output.view(divided, -1)
            loss_kl = kl_divergence_loss(output, tr_label)
            # loss_reconizer=crit(log_softmax(output.reshape(-1,len(vocab))),tr_label)
            # cer_func.add(output,loss_reconizer)
            loss_kl.backward()
            return loss_kl

        elif mode == "gen_update":
            self.iter_num += 1
            f_xs = self.generative.enc_image(tr_img)  # b,512,8,27
            f_xt, f_embed = self.generative.enc_text(tr_label)  # b,4096  b,512,8,27
            f_mix = self.generative.mix(f_xs, f_embed)

            xg = self.generative.decode(f_mix, f_xt)  # translation b,1,64,128
            l_dis_ori = self.dis.calc_gen_loss(xg)
            return l_dis_ori
        elif mode == "eval":
            with torch.no_grad():
                fx = self.generative.enc_image(tr_img)
                f_xt, f_embed = self.generative.enc_text(tr_label)
                f_mix = self.generative.mix(f_xs, f_embed)
                xg = self.generative.decode(f_mix, f_xt)

                pred_xt = self.recog(xg, tr_label)
                self.iter_num += 1
                """ Dis loss """
                l_dis_ori = self.discri.calc_gen_loss(xg)
                """ Rec loss """
                cer_te, cer_te2 = cer_func
                l_rec_ori = crit(
                    log_softmax(pred_xt.reshape(-1, len(vocab))), tr_label.reshape(-1)
                )

                """Writer classification Loss """

                l_cla_ori = self.writer_Model(xg, tr_label[1])
            return l_dis_ori, l_cla_ori, l_rec_ori
