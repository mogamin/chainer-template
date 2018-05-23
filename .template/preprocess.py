import chainer
import numpy as np
import cv2
xp = chainer.cuda.cupy


def randomcrop(img, sub=5):
    '''
    sub x sub times num of data
    '''
    height, width = img.shape[:2]
    shorter = np.min(img.shape[:2])
    cropsize = shorter - sub
    x_sub = width - cropsize
    y_sub = height - cropsize
    y = np.random.randint(0, y_sub)
    x = np.random.randint(0, x_sub)

    return img[y:y + cropsize, x:x + cropsize, :]


def randomflip(img):
    p = np.random.randint(4)
    if p == 0:
        img = img[:, ::-1, :]
    elif p == 1:
        img = img[::-1, :, :]
    elif p == 2:
        img = img[::-1, ::-1, :]

    return img


def randomrotate(img):
    p = np.random.randint(3)
    if p > 0:
        img = xp.rot90(img)
        if p > 1:
            img = xp.rot90(img)
            if p > 2:
                img = xp.rot90(img)

    return img


def centercrop(img):
    height, width = img.shape[:2]
    cropsize = np.min(img.shape[:2])
    y = (height - cropsize) // 2
    x = (width - cropsize) // 2
    return img[y:y + cropsize, x:x + cropsize, :]


def randomerase(img, max_intensity, prob=0.5, sh=0.4, sl=0.02, r1=0.3):
    if np.random.rand(1) < prob:
        height, width = img.shape[:2]
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
        img[y: y + he, x: x + we, :] = np.random.randint(0, max_intensity)
        return img
    else:
        return img


class PreprocessDataset(chainer.dataset.DatasetMixin):
    '''
    following processes are applied
        - mean subtraction
        - random erasing
        - random flipping
        - random cropping
        - resize arbitrary shape
    '''

    def __init__(self, imgs, size, eraseprob=0.5, max_intensity=255):
        self.imgs = imgs
        self.mean = np.mean(imgs, axis=0)
        self.eraseprob = eraseprob
        self.size = (size, size)
        self.max_intensity = max_intensity

    def __len__(self):
        return len(self.imgs)

    def subtractmean(self, img):
        return img - self.mean

    def get_example(self, i):
        img = self.imgs[i]
        img = self.subtractmean(img)
        img = randomerase(img, self.max_intensity, self.eraseprob)
        img = randomflip(img)
        img = randomrotate(img)
        img = randomcrop(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = xp.array(img).astype(xp.float32) / self.max_intensity

        return img.transpose(2, 0, 1)


class TestDataset(PreprocessDataset):

    def get_example(self, i):
        img = self.imgs[i]
        img = self.subtractmean(img)
        img = centercrop(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = xp.array(img).astype(xp.float32) / self.max_intensity

        return img.transpose(2, 0, 1)
