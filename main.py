from conf import parse_opts


if __name__ == '__main__':
    config = parse_opts()
    print(f'CUDA: {config.cuda_id}')