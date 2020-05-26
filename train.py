from typing import Dict

from torch.autograd import Variable

from logger import Logger
from quadriplet_loss import batch_hard_quadriplet_loss
from utils import STEP
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def epoch_step(epoch, loaders: Dict[STEP, DataLoader], model: torch.nn.Module,
               loggers: Dict[STEP, Logger], criterion, optimizer,
               conf):
    n_iter = sum([len(loader) for loader in loaders.values()])
    steps = list(loaders.keys())
    assert list(loggers.keys()) == list(loaders.keys()), "excpected same steps for loaders and loggers"
    with tqdm(total=n_iter, file=sys.stdout) as t:
        t.set_description(f'EPOCH: {epoch + 1}/{conf.n_epochs}')
        for step in steps:
            loader = loaders[step]
            logger = loggers[step]
            if step == STEP.TRAIN:
                model.train()
            else:
                model.eval()
            for i, (inputs, targets, scene_targets) in enumerate(loader):
                if conf.cuda_available:
                    targets = targets.cuda(device=conf.cuda_id0, non_blocking=True)
                inputs = Variable(inputs)
                targets = Variable(targets)

                if conf.use_quadriplet:
                    embs, outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    batch_hard_loss = 0.5 * batch_hard_quadriplet_loss(targets, scene_targets, embs)
                    loss += batch_hard_loss
                else:

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                if step == STEP.TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                logger.update(loss.data, outputs, targets)
                t.update(1)
                t.set_postfix(ordered_dict=logger.main_state(), refresh=True)
            logger.update_epoch(epoch)
            logger.reset()

            if epoch % conf.checkpoint == 0:
                save_file_path = os.path.join(conf.result_path,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': conf.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
