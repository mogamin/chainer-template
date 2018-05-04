import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer.training import extensions

from utils import load_cifar
from models import archs
from preprocess import PreprocessDataset, TestDataset

import argparse
import numpy as np


def argpars():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', required=True)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    return parser.parse_args()


def main():
    # -set config-
    args = argpars()
    Net = archs[args.net]
    print('NETWORK:', args.net)
    print('BATCH SIZE:', args.batch)
    print('EPOCH:', args.epoch)
    # -end set config-

    xp = cuda.cupy if args.gpu > -1 else np

    # =========================================================================
    # -load data
    # Apply function that loads objective data
    train_x, train_y, val_x, val_y = load_cifar()
    # =========================================================================

    # -set num of class-
    n_class = len(np.unique(train_y))

    # -cast-
    train_x = xp.array(train_x).astype(xp.float32)
    train_y = xp.array(train_y).astype(xp.int8)
    val_x = xp.array(val_x).astype(xp.float32)
    val_y = xp.array(val_y).astype(xp.int8)

    # -model definition-
    model = L.Classifier(Net(n_class), lossfun=F.softmax_cross_entropy)
    if args.gpu > 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # -set the optimizer -
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # -preprocessing-
    train = chainer.datasets.TupleDataset(PreprocessDataset(train_x), train_y)
    val = chainer.datasets.TupleDataset(TestDataset(val_x), val_y)

    # -define iterators-
    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    val_iter = chainer.iterators.SerialIterator(
        val, args.batch, repeat=False, shuffle=False)

    # -define updater-
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # -set trainer-
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'))

    # -set trainer extensions-
    trainer.extend(extensions.Evaluator(
        val_iter, model), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(
        ['epoch',
         'main/accuracy',
         'validation/main/accuracy',
         'elapsed_time']))
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/accuracy',
         'validation/main/accuracy',
         'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # -start learning-
    trainer.run()

    return 0


if __name__ == '__main__':
    main()
