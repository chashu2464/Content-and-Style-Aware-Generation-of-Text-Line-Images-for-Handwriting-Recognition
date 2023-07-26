from text_style import TextEncoder_FC
from models import Visual_encoder
import torch
from torch import nn
from parameters import device


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


import torch.nn.functional as F


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        print("coming here In REsBLOCKS")
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()
        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        print("Shape of X", x.shape)
        print("coming into the Resblock")
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):  # Residual connection
    def __init__(self, fin, fout, fhid=None, activation="lrelu", norm="none"):
        super().__init__()
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(
            self.fin,
            self.fhid,
            3,
            1,
            padding=1,
            pad_type="reflect",
            norm=norm,
            activation=activation,
            activation_first=True,
        )
        self.conv_1 = Conv2dBlock(
            self.fhid,
            self.fout,
            3,
            1,
            padding=1,
            pad_type="reflect",
            norm=norm,
            activation=activation,
            activation_first=True,
        )
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(
                self.fin, self.fout, 1, 1, activation="none", use_bias=False
            )

    def forward(self, x):
        print("Shape of X", x.shape)
        print("coming in ACTFIRSTRESBLOCK")
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm="none", activation="relu"):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        print("Shape of X", x.shape)
        print("coming into the Linear")
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
        use_bias=True,
        activation_first=True,
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, 1, 1,)

    def forward(self, x):
        print("Shape of X", x.shape)
        print("coming into the CONV2D")
        if self.activation_first:

            if self.activation:
                x = self.activation(x)
            pad = self.pad(x)

            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # x = x.view(b * c, *x.size()[2:])
        x = self.bn(x)
        # x = x.view(b, c, *x.size()[1:])

        return x

    # class AdaptiveInstanceNorm2d(nn.Module):
    #     def __init__(self, num_features, eps=1e-5, momentum=0.1):
    #         super(AdaptiveInstanceNorm2d, self).__init__()
    #         self.num_features = num_features
    #         self.eps = eps
    #         self.momentum = momentum
    #         self.weight = None
    #         self.bias = None
    #         self.register_buffer('running_mean', torch.zeros(num_features))
    #         self.register_buffer('running_var', torch.ones(num_features))

    #     def forward(self, x):
    #         print("Shape of X",x.shape)
    #         print("Coming into the AdaptiveInstanceNorm2d")
    #         assert self.weight is not None and \
    #                self.bias is not None, "Please assign AdaIN weight first"
    #         b, c = x.size(0), x.size(1)
    #         running_mean = self.running_mean.repeat(b)
    #         running_var = self.running_var.repeat(b)
    #         x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
    #         import pdb;pdb.set_trace()
    #         out = F.batch_norm(
    #             x_reshaped, running_mean, running_var, self.weight, self.bias,
    #             True, self.momentum, self.eps)
    #         return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


# class Decoder(nn.Module):
#     def __init__(self, ups=3, n_res=2, dim=2, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
#         super(Decoder, self).__init__()

#         self.model = []
#         self.model += [ResBlocks(n_res, dim, res_norm,
#                                  activ, pad_type=pad_type)]
#         for i in range(ups):
#             self.model += [#nn.Upsample(scale_factor=1),
#                            Conv2dBlock(1, 1 // 2, 5, 1, 2,
#                                        norm='in',
#                                        activation=activ,
#                                        pad_type=pad_type)]
#             #dim //= 2
#         self.model += [Conv2dBlock(1, 1, 7, 1, 3,
#                                    norm=res_norm,
#                                    activation='tanh',
#                                    pad_type=pad_type)]
#         self.model = nn.Sequential(*self.model)

#     def forward(self, x):
#         return self.model(x)
class Decoder(nn.Module):
    def __init__(
        self,
        ups=3,
        n_res=2,
        dim=2,
        out_dim=1,
        res_norm="adain",
        activ="relu",
        pad_type="reflect",
    ):
        super(Decoder, self).__init__()

        # self.model = nn.Sequential()

        # Residual Blocks
        self.renet = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)

        # Upsampling Blocks
        for i in range(ups):
            # self.model.add_module(f'upsample_{i+1}', nn.Upsample(scale_factor=2))
            self.con = Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="in", activation=activ, pad_type=pad_type
            )
            # dim //= 2

        # Final Convolutional Block
        self.con_final = Conv2dBlock(
            dim, out_dim, 7, 1, 3, norm=res_norm, activation="tanh", pad_type=pad_type
        )

    def forward(self, x):

        net = self.renet(x)

        return self.model(x)


class GenModel_FC(nn.Module):
    def __init__(self):
        super(GenModel_FC, self).__init__()
        self.enc_image = Visual_encoder().to(device)
        self.enc_text = TextEncoder_FC().to(device)
        self.dec = Decoder().to(device)
        self.linear_mix = nn.Conv2d(2, 2, 1, 1).to(device)
        self.dowm_sampling = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1
        )

    def decode(self, content, adain_params):
        # decode content and style codes to an image
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        dowm_sampling_embed = self.dowm_sampling(feat_embed)
        feat_mix = torch.cat([feat_xs, dowm_sampling_embed], dim=1)  # b,1024,8,27
        ff = self.linear_mix(feat_mix)  # b,8,27,1024->b,8,27,512
        return ff
