from parameters import *

def pad_str(data):
    '''
    data:str [('hello',"what",),("on the road","where we are")]
    data :- lenght of data is dependent on the number_example and the batch_size data[num_examples][batchsize]

    '''
    data = list(data)
    for i in range(len(data)):
        # for j in range(len(data[i])):
        # data[i] = tuple(s.ljust(text_max_len, " ") for s in data[i])
        if len(data[i]) < text_max_len:
            max_str = str()
            data[i] += " " * (text_max_len - len(data[i]))
            # data[i]=max_str
        else:
            data[i] = data[i]
    return tuple(data)


def encoding(label: list, encoder):
    print("coming here")
    lst = torch.zeros((batch_size, num_example))
    for row in range(len(label)):
        for col in range(len(label[row])):
            string = list()
       
            label[row] = tuple(
                torch.tensor([encoder[char] for char in label[row][col]])
            )
    return label


def decoding(label, decoder):
    # Label[example][batch_size]
    words = []
    for str1 in label:
        chars = []
        for word in str1:
            chars.append([decoder[char] for char in word])
        words.append(torch.tensor(chars))
    return words  # [examples][batch_size]


def encoding(label, decoder):
    print("should be coming here")
    # Label[example][batch_size]
    words = [
        torch.tensor([[decoder[char] for char in word] for word in str1])
        for str1 in label
    ]
    return words  # [examples][batch_size]
