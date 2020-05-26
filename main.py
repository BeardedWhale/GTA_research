from conf import parse_opts
from model import generate_model
from torchsummary import summary
import torch

if __name__ == '__main__':
    config = parse_opts()
    model, params = generate_model(config)
    model(torch.ones((3, 64, 224, 224)))
    # summary(model, input_size=(3, 64, 112, 112))
    # print(f'CUDA: {config.cuda_id}')
