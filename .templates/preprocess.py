import chainer
import numpy as np
from scipy import ndimage


def randomcrop(img, cropsize=28):
    sub = img.shape[0] - cropsize
    x, y = np.random.randint(0, sub, 2)
    return img[y:y + cropsize, x:x + cropsize, :]


def randomflip(img):
    p = np.random.randint(3)
    if p == 0:
        img = img[:, ::-1, :]
    elif p == 1:
        img = img[::-1, :, :]
    elif p == 3:
        img = img[::-1, ::-1, :]

    return img


def randomrotate(img):
    p = np.random.randint(3)
    if p == 0:
        img = ndimage.rotate(img, 90, reshape=False)
    elif p == 1:
        img = ndimage.rotate(img, 180, reshape=False)
    elif p == 3:
        img = ndimage.rotate(img, 270, reshape=False)

    return img


def centercrop(img, cropsize=28):
    x = (img.shape[0] - cropsize)/2
    y = (img.shape[1] - cropsize)/2
    return img[y: y+cropsize, x: x+cropsize, :]


def randomerase(img, prob=0.5, sh=0.4, sl=0.02, r1=0.3):
    if np.random.rand(1) < prob:
        height, width = img.shape[: 2]
        S = height * width
        r2 = 1 / r1
        while True:
            se = np.random.uniform(sl, sh) * S
            re = np.random.uniform(r1, r2)
            he = int(np.sqrt(se * re))
            we = int(np.sqrt(se * 1 / re))
            x = np.random.randint(width)
            y = np.random.randint(height)
            if (x + we) < width and (y + he) < height:
                break
        img[y: y + he, x: x + we, :] = np.random.randint(0, 255)
        return img
    else:
        return img


class PreprocessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, imgs, eraseprob=0.5):
        self.imgs = imgs
        self.mean = np.mean(imgs, axis=0)
        self.eraseprob = eraseprob

    def __len__(self):
        return len(self.imgs)

    def subtractmean(self, img):
        return img - self.mean

    def get_example(self, i):
        img = self.imgs[i]
        img = self.subtractmean(img)
        img = randomerase(img, self.eraseprob)
        img = randomflip(img)
        # img = randomrotate(img)
        img = randomcrop(img)

        return img.transpose(2, 0, 1)


class TestDataset(PreprocessDataset):
    def get_example(self, i):
        img = self.imgs[i]
        img = self.subtractmean(img)
        img = centercrop(img)

        return img.transpose(2, 0, 1)
