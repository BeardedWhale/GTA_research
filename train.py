from utils import AverageMeter, calculate_accuracy, per_class_accuracies, PerClassAcc, CLASS_MAP, targets_to_one_hot


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, experiment=None):
    print('train at epoch {}'.format(epoch))

    model.train()
    per_class_acc =  PerClassAcc(opt.n_finetune_classes)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    all_targets = []
    all_outputs = []
    end_time = time.time()
    for i, (inputs, targets, scene_targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_time.add(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(device=opt.cuda_id, non_blocking=True)
        inputs = Variable(inputs)
        targets = Variable(targets)


        if opt.use_quadriplet:
            embs, outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_hard_loss = 0.5 * batch_hard_quadriplet_loss(targets, scene_targets, embs)
            loss += batch_hard_loss
        else:

            outputs = model(inputs)
            loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs, targets)
        pr_cl = per_class_accuracies(outputs, targets)
        per_class_acc.add(pr_cl)
        losses.add(loss.data, inputs.size(0))
        accuracies.add(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.add(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if experiment:
            experiment.log_metric('TRAIN Loss batch', losses.val.cpu())
            experiment.log_metric('TRAIN Acc batch', accuracies.val.cpu())

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

        all_outputs.extend(torch.nn.functional.sigmoid(outputs).tolist())
        target_one_hot = targets_to_one_hot(targets, opt.n_finetune_classes)
        all_targets.extend(target_one_hot.tolist())

    experiment.log_confusion_matrix(all_targets, all_outputs, title=f'TRAIN matrix EPOCH {epoch}', step=epoch,
        file_name=f"TRAIN-confusion-matrix-{epoch}.json")
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    if experiment:
        experiment.log_metric('TRAIN Loss epoch', losses.avg.cpu())
        experiment.log_metric('TRAIN Acc epoch', accuracies.avg.cpu())
        experiment.log_metric('TRAIN LR', optimizer.param_groups[0]['lr'])

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
    for i in range(opt.n_finetune_classes):
        total, true = per_class_acc.classes_count_d[i]
        experiment.log_metric(f'TRAIN {CLASS_MAP[i]} Acc', true/(1.0*total))