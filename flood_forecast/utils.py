import torch
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


def flatten_list_function(input_list):
    return [item for sublist in input_list for item in sublist]
