import os, glob
import torch

scale_factor = 1
number_feature=2000
IMAGE_HEIGHT =  64//10
IMAGE_WIDTH =250//10
# device = torch.device("cuda" if torch.cu+2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
batch_size = 2
embedding_size = 64
Max_str = 25
text_max_len = Max_str

vocab = {
    " ",
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
}

cfg = {
    "E": [
        64,
        64,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}
encoder = {data: i for i, data in enumerate(vocab)}
decoder = {i: data for i, data in enumerate(vocab)}
"""
encoder= {"A":0,"B":1}
decoder={"0":A,"1":B}
"""
tokens = {"GO_TOKEN": 0, "END_TOKEN": 1, "PAD_TOKEN": 2}
NUM_WRITERS = 500
