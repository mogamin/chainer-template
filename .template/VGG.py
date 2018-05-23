import chainer
import chainer.links as L
import chainer.functions as F


class ConvBlock(chainer.Chain):
    def __init__(self, out_ch, drop_ratio=0, pool=False):
        self.drop_ratio = drop_ratio
        self.pool = pool
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                None, out_ch, ksize=3, stride=1, pad=1, nobias=True)
            self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.drop_ratio > 0:
            h = F.dropout(h, ratio=self.drop_ratio)
        if self.pool:
            h = F.max_pooling_2d(h, ksize=2, stride=2)
        return h


class LinBlock(chainer.Chain):
    def __init__(self, out_ch, drop_ratio=0):
        self.drop_ratio = drop_ratio
        super(LinBlock, self).__init__()
        with self.init_scope():
            self.ln = L.Linear(None, out_ch, nobias=True)
            self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.bn(self.ln(x)))
        if self.drop_ratio > 0:
            h = F.dropout(h, ratio=self.drop_ratio)
        return h


class VGG(chainer.ChainList):
    def __init__(self, n_class, n_blocks):

        super(VGG, self).__init__()
        # set conv layers
        for n_conv, out_ch, drop_ratio in n_blocks:
            for _ in range(n_conv - 1):
                self.add_link(ConvBlock(out_ch, drop_ratio))
            self.add_link(ConvBlock(out_ch, pool=True))

        # set lin layers
        self.add_link(LinBlock(1024, 0.5))
        self.add_link(LinBlock(1024, 0.5))
        self.add_link(L.Linear(None, n_class))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class VGG16(VGG):
    def __init__(self, n_class):
        n_blocks = [[2, 64, .3],
                    [2, 128, .4],
                    [3, 256, .4],
                    [3, 512, .4],
                    [3, 512, .4]]
        super(VGG16, self).__init__(n_class, n_blocks)


class VGG19(VGG):
    def __init__(self, n_class):
        n_blocks = [[2, 64, .3],
                    [2, 128, .4],
                    [4, 256, .4],
                    [4, 512, .4],
                    [4, 512, .4]]
        super(VGG19, self).__init__(n_class, n_blocks)


archs = {
    'VGG16': VGG16,
    'VGG19': VGG19,
}
