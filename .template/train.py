import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from utils import load_cifar
from models import archs
from preprocess import PreprocessDataset, TestDataset

import os
import argparse
import numpy as np
from datetime import datetime

xp = chainer.cuda.cupy


def argpars():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('-n', '--net', required=True)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-s', '--size', type=int, default=32)
    return parser.parse_args()


def main():
    # -Set config-
    args = argpars()
    print('NETWORK:', args.net)
    print('BATCH SIZE:', args.batch)
    print('EPOCH:', args.epoch)
    print('SIZE:', args.size)
    print('GPU:', args.gpu)
    print('TIMESTAMP:', timestamp)
    # -End of Set config-

    # =========================================================================
    # -Load data
    # Apply function that loads objective data
    train_x, train_y, val_x, val_y = load_cifar()
    # =========================================================================

    # -Set num of class-
    n_class = len(np.unique(train_y))

    # -Model definition-
    Net = archs[args.net]
    model = L.Classifier(Net(n_class), lossfun=F.softmax_cross_entropy)

    # -Select cupy or numpy-
    if args.gpu > -1:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    else:
        xp = np
    # -End of Select cupy or numpy-

    # -Cast-
    train_y = xp.array(train_y).astype(xp.int8)
    val_y = xp.array(val_y).astype(xp.int8)

    # -Set optimizer -
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # -Preprocessing-
    train = chainer.datasets.TupleDataset(PreprocessDataset(train_x, args.size), train_y)
    val = chainer.datasets.TupleDataset(TestDataset(val_x, args.size), val_y)

    # -Set iterators-
    train_iter = chainer.iterators.SerialIterator(
        train, args.batch)
    val_iter = chainer.iterators.SerialIterator(
        val, args.batch, repeat=False, shuffle=False)

    # -Set Updater-
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # -Set trainer-
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'))

    # -Set trainer extensions-
    # --Snapshot--
    snapshot_name = 'snaptshot_{}'.format(timestamp)
    snapshot_trigger = \
        training.triggers.MinValueTrigger('main/loss', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, snapshot_name), trigger=snapshot_trigger)
    # --End of Snapshot--
    # --Evaluator--
    trainer.extend(extensions.Evaluator(val_iter, model), trigger=(1, 'epoch'))
    # --Report--
    log_dir = 'logs'
    log_name = 'log_{}'.format(timestamp)
    log_name = os.path.join(log_dir, log_name)
    trainer.extend(extensions.LogReport(
        ['epoch',
         'main/accuracy',
         'validation/main/accuracy',
         'elapsed_time'],
        log_name=log_name))
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/accuracy',
         'validation/main/accuracy',
         'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # --End of Report--
    # -End of Set trainer extensions-

    # -Start learning-
    trainer.run()

    # -Serialize model-
    model.to_cpu()
    model_name = '{}_{}.npz'.format(timestamp, args.net)
    model_name = os.path.join('result', 'model_npzs', model_name)
    chainer.serializers.save_npz(model_name, model.predictor)
    # -End of Serialize model-

    # -Rename snapshot-
    os.rename('result/' + snapshot_name, 'result/snapshots/' + snapshot_name)

    return 0


if __name__ == '__main__':
    timestamp = datetime.today().strftime('%y%m%d%H%M%S')
    main()
