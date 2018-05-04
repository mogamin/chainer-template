import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions

from utils import load_cifar
from models import archs
from preprocess import PreprocessDataset, TestDataset

import argparse
import numpy as np
import multiprocessing


def argpars():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('-n', '--net', required=True)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-c', '--cpu', type=int, default=1)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    return parser.parse_args()


def main():
    # -Set config-
    args = argpars()
    print('NETWORK:', args.net)
    print('BATCH SIZE:', args.batch)
    print('EPOCH:', args.epoch)
    print('GPU:', args.gpu)
    print('CPU:', args.cpu)
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
        xp = chainer.cuda.cupy
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        multiprocessing.set_start_method('spawn')
    else:
        xp = np
    # -End of Select cupy or numpy-

    # -Cast-
    train_x = xp.array(train_x).astype(xp.float32)
    train_y = xp.array(train_y).astype(xp.int8)
    val_x = xp.array(val_x).astype(xp.float32)
    val_y = xp.array(val_y).astype(xp.int8)

    # -Set optimizer -
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # -Preprocessing-
    train = chainer.datasets.TupleDataset(PreprocessDataset(train_x), train_y)
    val = chainer.datasets.TupleDataset(TestDataset(val_x), val_y)

    # -Set iterators-
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batch, n_processes=args.cpu)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batch, repeat=False, shuffle=False, n_processes=1)

    # -Set Updater-
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # -Set trainer-
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'))

    # -Set trainer extensions-
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
    # -End of Set trainer extensions-

    # -Start learning-
    trainer.run()

    return 0


if __name__ == '__main__':
    main()
