import chainer
import chainer.links as L
import chainer.functions as F
from VGG import VGG16, VGG19


class MLP(chainer.Chain):
    def __init__(self, n_class):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 4096)
            self.l2 = L.Linear(None, 4096)
            self.l2 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


class CNN(chainer.Chain):
    def __init__(self, n_class):
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, 64, ksize=3, stride=1, pad=1, nobias=True)
            self.conv2 = L.Convolution2D(
                64, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.conv3 = L.Convolution2D(
                128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.fc1 = L.Linear(None, 4096)
            self.fc2 = L.Linear(None, 4096)
            self.fc3 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2, stride=2)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


archs = {
    'MLP': MLP,
    'VGG16': VGG16,
    'VGG19': VGG19,
}
