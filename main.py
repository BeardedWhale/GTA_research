from conf import parse_opts
from model import generate_model
from torchsummary import summary

if __name__ == '__main__':
    config = parse_opts()
    model, params = generate_model(config)
    summary(model, input_size=(3, 32, 112, 112))
    # print(f'CUDA: {config.cuda_id}')
