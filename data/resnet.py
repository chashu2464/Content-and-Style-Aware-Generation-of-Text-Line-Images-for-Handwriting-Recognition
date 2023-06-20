import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import device
from models import Visual_encoder
from text_style import TextEncoder_FC
from decoder import Decorder
from encoder_vgg import Encoder


class AdaLN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.rho, 0.9)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.beta, 0.0)

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation=F.leaky_relu,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(in_channels,)
        self.stride = stride
        self.activation = activation
        self.adaln = AdaLN(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.adaln(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        # define the number of layers
        self.n_layers = 6
        self.final_size = 1024
        in_dim = 1
        out_dim = 16
        self.ff_cc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=7,
            stride=1,
            padding="same",
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_dim, out_dim,) for _ in range(self.n_layers)]
        )
        self.cnn_f = nn.Conv2d(
            out_dim, self.final_size, kernel_size=7, stride=1, padding="same"
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        ff_cc = self.ff_cc(x)
        resnet = self.res_blocks(ff_cc)
        output = self.cnn_f(resnet)
        return output.squeeze(-1).squeeze(-1)

    def calc_dis_fake_loss(self, input_fake):

        resp_fake = self.forward(input_fake.permute(1, 0, 2, 3))
        fake_img = torch.zeros(resp_fake.shape).to(device)
        fake_loss = self.bce(resp_fake, fake_img)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        resp_real = self.forward(input_real.permute(1, 0, 2, 3))
        label = torch.ones(resp_real.shape).to(device=device)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        resp_fake = self.forward(input_fake.permute(1, 0, 2, 3))
        label = torch.ones(resp_fake.shape).to(device)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss


class WriterClaModel(nn.Module):
    """
        WriterclaModel  classification method for the writer class
            
        contain covultional layer at the start and end I guess to make intput and output data 
        compatible shape and size with other modules of the code. Then the inner part contains
        residual block of code  dependes on number of layer.           
    """

    def __init__(self, num_writers) -> None:
        super(WriterClaModel, self).__init__()
        self.n_layers = 6
        in_dim = 1
        out_dim = 16
        self.num_writers = num_writers
        #
        self.cnn_f = nn.Conv2d(
            in_channels=in_dim,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1
            # padding_mode="reflect",
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_dim, out_dim,) for _ in range(self.n_layers)]
        )
        self.ff_cc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=self.num_writers,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        cnn_f = self.cnn_f(x)
        resnet = self.res_blocks(cnn_f)
        ff_cc = self.ff_cc(resnet)
        loss = self.cross_entropy(ff_cc.view(-1, self.num_writers), y.view(-1).long())
        return loss


class GenModel_FC(nn.Module):
    def __init__(self):
        super(GenModel_FC, self).__init__()
        self.enc_image = Visual_encoder().to(device)
        self.encoding = Encoder().to(device)
        self.enc_text = TextEncoder_FC().to(device)
        self.dec = Decorder().to(device)
        self.linear_mix = nn.Conv2d(4, 2, 1, 1).to(device)
        self.dowm_sampling = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1
        )

    def decode(self, content, label_text):
        # decode content and style codes to an image
        self.dec(content, label_text)

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        dowm_sampling_embed = self.dowm_sampling(feat_embed)
        feat_mix = torch.cat(
            [feat_xs, dowm_sampling_embed.permute(1, 0, 2, 3)], dim=1
        )  # b,1024,8,27
        ff = self.linear_mix(feat_mix)  # b,8,27,1024->b,8,27,512
        return ff


class Generator(nn.Module):
    def __init__(
        self, class_num, num_res_blocks=4, norm_layer=AdaLN, activation=F.leaky_relu
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.norm_layer = norm_layer
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    64, 64, norm_layer=self.norm_layer, activation=self.activation
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn2 = norm_layer(32)
        self.conv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn3 = norm_layer(16)
        self.conv4 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.linear = nn.Linear(3, out_features=class_num)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.linear(x)
        return x


class RecModel(nn.Module):
    def __init__(self):
        super(RecModel, self).__init__()
        self.enc = Encoder()
        self.dec = Decorder()

    def forward(self, image, text):
        visual_out = self.enc(image)
        text_visual = self.dec(visual_out, text)
        return text_visual
