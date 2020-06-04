from conf import parse_opts
from experiments import run_classification_experment, run_embeddings_experment, run_new_classification_experment
from train import train

if __name__ == '__main__':
    # uncomment to run experiments
    # run_classification_experment()
    # run_embeddings_experment()
    # run_new_classification_experment()
    config = parse_opts()
    train(config)
