# import comet_ml in the top of your file
from comet_ml import Experiment
import torch
import numpy as np
from tqdm import tqdm
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="Cbyqfs9Z8auN5ivKsbv2Z6Ogi",
                        project_name="test", workspace="beardedwhale")
from logger import  Logger
from utils import STEP
CLASS_MAP = ['Arson', 'Fight', 'Vandalism', 'Explosion', 'Arrest', 'Robbery', 'Assault', 'Shooting']
logger = Logger(experiment, STEP.TEST, n_classes=8, topk=[1, 2, 3], class_map=CLASS_MAP, metrics=['KEK'])


for epoch in range(20):
    for batch in tqdm(range(100)):
        target = torch.tensor(np.random.choice([i for i in range(len(CLASS_MAP))], size=100))
        outputs = torch.tensor(np.random.random((100, 8)))
        loss = np.random.rand()
        kek = np.random.rand()
        logger.update(loss, outputs, target, KEK=kek)
        print(logger.main_state())

    logger.update_epoch(epoch)
    print(logger.state_message())
    logger.reset()