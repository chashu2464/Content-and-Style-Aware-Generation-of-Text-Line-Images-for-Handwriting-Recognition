import torch
from torch.utils.data import DataLoader
import tqdm
from parameters import *
from data_set import CustomImageDataset
import matplotlib.pyplot as plt
from loss import CER
#from models import Visual_encoder, TextEncoder_FC
from helper import pad_str, decoding
from torch import optim
import numpy as np
import time
from encoder_vgg import Encoder

OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 200
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 8
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5
lr_cla = 1 * 1e-5

CurriculumModelID = 5


def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_cla = list()
    loss_cla_tr = list()
    loss_l1 = list()
    loss_rec = list()
    loss_rec_tr = list()
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    for train_data_list in train_loader:
        """rec update"""
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, "rec_update", cer_tr)
        rec_opt.step()

        """classifier update"""
        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, "cla_update")
        cla_opt.step()

        """dis update"""
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, "dis_update")
        dis_opt.step()

        """gen update"""
        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_l1, l_rec = model(
            train_data_list, epoch, "gen_update", [cer_te, cer_te2]
        )
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_cla_tr.append(l_cla_tr.cpu().item())
        loss_l1.append(l_l1.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_cla = np.mean(loss_cla)
    fl_cla_tr = np.mean(loss_cla_tr)
    fl_l1 = np.mean(loss_l1)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print(
        "epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f, l1=%.2f, cer=%.2f-%.2f-%.2f, time=%.1f"
        % (
            epoch,
            fl_dis_tr,
            fl_dis,
            fl_cla_tr,
            fl_cla,
            fl_rec_tr,
            fl_rec,
            fl_l1,
            res_cer_tr,
            res_cer_te,
            res_cer_te2,
            time.time() - time_s,
        )
    )
    return res_cer_te + res_cer_te2


def test(test_loader, epoch, modelFile_o_model):
    if type(modelFile_o_model) == str:
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print("Loading " + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model))  # load
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_cla = list()
    loss_rec = list()
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()
    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, "eval", [cer_te, cer_te2])

        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_cla = np.mean(loss_cla)
    fl_rec = np.mean(loss_rec)

    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print(
        "EVAL: l_dis=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f, time=%.1f"
        % (fl_dis, fl_cla, fl_rec, res_cer_te, res_cer_te2, time.time() - time_s)
    )


def main(train_loader, test_loader, num_writers):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(device)

    if CurriculumModelID > 0:
        model_file = "save_weights/contran-" + str(CurriculumModelID) + ".model"
        print("Loading " + model_file)
        model.load_state_dict(torch.load(model_file))  # load
        # pretrain_dict = torch.load(model_file)
        # model_dict = model.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and not k.startswith('gen.enc_text.fc')}
        # model_dict.update(pretrain_dict)
        # model.load_state_dict(model_dict)

    dis_params = list(model.dis.parameters())
    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    cla_params = list(model.cla.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    cla_opt = optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)
    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    for epoch in range(CurriculumModelID, epochs):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = "save_weights"
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + "/contran-%d.model" % epoch)

        if epoch % EVAL_EPOCH == 0:
            test(test_loader, epoch, model)

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer:
                min_cer = cer
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print("Early stop at %d and the best epoch is %d" % (epoch, min_idx))
                model_url = "save_weights/contran-" + str(min_idx) + ".model"
                os.system("mv " + model_url + " " + model_url + ".bak")
                os.system("rm save_weights/contran-*.model")
                break


def rm_old_model(index):
    models = glob.glob("save_weights/*.model")
    for m in models:
        epoch = int(m.split(".")[0].split("-")[1])
        if epoch < index:
            os.system("rm save_weights/contran-" + str(epoch) + ".model")


if __name__ == "__main__":

    print("batch_size", batch_size)
    print("program is running using::", device)
    print(time.ctime())
    TextDatasetObj = CustomImageDataset()
    no_workers = batch_size // num_example
    dataset = torch.utils.data.DataLoader(
        TextDatasetObj, batch_size=batch_size, shuffle=True, num_workers=no_workers,
    )
    net=Encoder().to(device)
  
    trainable_parameter = sum(
	param.numel() for param in net.parameters() if param.requires_grad
    )
    

    print(f" Total parameters = {trainable_parameter/ 1e6:.2f} .Millions")
    for Image, Label in tqdm.tqdm(dataset):
        
        V_out = net(Image[0].to(device).unsqueeze(1))
        
        break
    




    # train_size = int(0.8 * (len(dataset)))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # main(
    #     train_loader=train_data_loader,
    #     test_loader=test_data_loader,
    #     num_writers=NUM_WRITERS,
    # )

    # max_str = 0
    # str_2_int = []
    # label = list()
    # global_step = len(dataset)
    # V_encoder = Visual_encoder()
    # V_encoder.to(device=device)
    # T_encoder = TextEncoder_FC()
    # T_encoder.to(device=device)
    # # for i in tqdm.tqdm(range(10)):
    # for img, label in tqdm.tqdm(dataset):
    #     label = pad_str(label)
    #     stio_ten = decoding(
    #         label=label, decoder=encoder
    #     )  # (number_example, batch_size,text_len_max)
    #     # stio_ten = encoding(label=label, encoder=encoder)
    #     concate = torch.stack(stio_ten, dim=0)
    #     for sample in range(num_example):
    #         # (batch,512,Img,img_height,img_width)
    #         # image = torch.transpose(
    #         #     torch.transpose(img[sample], 1, 3), 2, 3
    #         # )  # [8, 342, 2270, 3] [8,3,342,2270]
    #         image = img[sample].to(device=device)
    #         print("image shape", image.shape)
    #         V_out = V_encoder(image.unsqueeze(1))
    #         T_out, ful_cat = T_encoder(concate[sample].to(device=device))
    #     break

    #     # for index in range(num_example):
    #     #     # print(img[index].shape)
    #     #     # print(stio_ten[index].size())
    #     #     V_out = V_encoder(img[index])
