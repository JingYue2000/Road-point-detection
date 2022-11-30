import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import copy
import scipy.io as sio
import matplotlib.pyplot as plt

# https://github.com/raymon-tian/hourglass-facekeypoints-detection

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),
])


class KFDataset(Dataset):
    def __init__(self, config, X=None, kpt_gts=None, seg_gts=None, imgnames=None, transform=transform):
        """

        :param X: (N,96*96)
        :param gts: (N,15,2)
        """
        self.__X = X
        self.__kpt_gts = kpt_gts
        self.__seg_gts = seg_gts
        self.__imgnames = imgnames
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']
        self.__fname = config['fname']
        self.__is_test = config['is_test']
        self.__stride = config['stride']
        self.__w_part = config['w_part']
        self.__facX = .08
        self.__facY = .15
        self.__transform = transform
        # self.__ftrain = config['ftrain']
        # self.load(self.__ftrain)

    def load(self, cols=None):
        fname = self.__fname

        data = sio.loadmat(fname)
        imgnames = data['fname']
        imgs = data['imgPath']
        kpt_gts = data['ptsAll']
        seg_gts = data['segGT']
        kpt_gts = kpt_gts[:, :, :2]
        self.__X = imgs
        self.__kpt_gts = kpt_gts
        self.__seg_gts = seg_gts
        self.__imgnames = imgnames

        # print(imgs.shape)    # (979, 256, 256, 3)
        # print(kpt_gts.shape)   # (979, 16, 2)
        # print(seg_gts.shape)    # (979, 256, 256)
        # print(imgnames.shape)   # (979,)

        return imgs, kpt_gts, seg_gts, imgnames

    def __len__(self):
        return len(self.__X)

    def __getitem__(self, item):
        C, H, W = 3, 256, 256
        stride = self.__stride
        x = self.__X[item]
        kpt_gt = self.__kpt_gts[item]
        seg_gt = self.__seg_gts[item]
        img_name = self.__imgnames[item]

        if np.random.randint(0, 2) and not self.__is_test:  # random flip
            x = np.array(x[:, ::-1, :])
            kpt_gt = np.array(kpt_gt[::-1, :])
            seg_gt = np.array(seg_gt[:, ::-1])
            for i, p in enumerate(kpt_gt):
                kpt_gt[i, 0] = W - 1 - p[0]

        keypoint_heatmaps = self._putPointHeatmaps(kpt_gt, H, W, stride, self.__sigma)
        part_heatmaps = self._putPartHeatmaps(kpt_gt, H, W, stride, self.__facX, self.__facY)
        if self.__w_part:
            heatmaps = np.concatenate((keypoint_heatmaps, part_heatmaps), axis=0)
        else:
            heatmaps = keypoint_heatmaps
        # print(img_name)
        # plt.imshow(x.astype(np.uint8))
        # plt.plot(kpt_gt[:,0], kpt_gt[:,1])
        # for i, p in enumerate(kpt_gt):
        #     plt.text(p[0], p[1], '{}'.format(i))
        # plt.show()
        # plt.imshow(seg_gt.astype(np.uint8))
        # plt.show()

        if self.__debug_vis == True:
            all_heatmaps = np.zeros((heatmaps.shape[1], heatmaps.shape[2]))
            for i in range(heatmaps.shape[0]):
                # if i < 16: continue
                img = copy.deepcopy(x).astype(np.uint8)
                # self.visualize_heatmap_target(img,copy.deepcopy(heatmaps[i]),kpt_gt,1)
                all_heatmaps += heatmaps[i]
            fig, ax = plt.subplots()
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            height, width, channels = img.shape
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.imshow(img)
            # plt.savefig('ori.png', dpi=300)
            plt.show()
            fig, ax = plt.subplots()
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.imshow(img)
            plt.imshow(all_heatmaps, alpha=.5)
            # plt.plot(gt[:,0], gt[:,1])
            plt.savefig('partmap.png', dpi=300)
            plt.show()

        x = Image.fromarray(x.astype('uint8')).convert('RGB')
        if self.__transform:
            x = self.__transform(x)
            '''
            unloader = transforms.ToPILImage()
            image = x.cpu().clone()  # we clone the tensor to not do changes on it
            image = image.squeeze(0)  # remove the fake batch dimension
            image = unloader(image)
            plt.imshow(image)
            plt.show()
            '''
        heatmaps = heatmaps.astype(np.float32)
        # heatmaps = keypoint_heatmaps.astype(np.float32)
        return x, heatmaps, kpt_gt, seg_gt

    def _putPartHeatmaps(self, keypoints, crop_size_y, crop_size_x, stride, facX=0.08, facY=0.15):
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num - 1):
            flag = ~np.isnan(all_keypoints[k, 0]) and ~np.isnan(all_keypoints[k + 1, 0])
            center, theta, length = self._getPartParams(all_keypoints[k], all_keypoints[k + 1])
            heatmap = self._generateHeatmap(flag, crop_size_y, crop_size_x, stride, center, length, theta, facX, facY)
            heatmap = heatmap[np.newaxis, ...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)
        return heatmaps_this_img

    def _putPointHeatmaps(self, keypoints, crop_size_y, crop_size_x, stride, sigma):
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k, 0])
            facX = facY = 1
            length = sigma
            theta = np.pi / 2
            center = all_keypoints[k]
            heatmap = self._generateHeatmap(flag, crop_size_y, crop_size_x, stride, center, length, theta, facX, facY)
            heatmap = heatmap[np.newaxis, ...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)
        return heatmaps_this_img

    def _generateHeatmap(self, visible_flag, crop_size_y, crop_size_x, stride, center, length, theta, facX=0.08,
                         facY=0.15):
        grid_y = int(crop_size_y / stride)
        grid_x = int(crop_size_x / stride)
        if visible_flag == False:
            return np.zeros((grid_y, grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        sigma1 = length * facX
        sigma2 = length * facY
        theta = np.pi / 2 - theta

        a = ((np.cos(theta) ** 2) / (2 * sigma1 ** 2)) + ((np.sin(theta) ** 2) / (2 * sigma2 ** 2))
        b = -((np.sin(2 * theta)) / (4 * sigma1 ** 2)) + ((np.sin(2 * theta)) / (4 * sigma2 ** 2))
        c = ((np.sin(theta) ** 2) / (2 * sigma1 ** 2)) + ((np.cos(theta) ** 2) / (2 * sigma2 ** 2))

        exponent = a * (xx - center[0]) ** 2 + 2 * b * (xx - center[0]) * (yy - center[1]) + c * (yy - center[1]) ** 2
        heatmap = np.exp(-exponent)
        heatmap[heatmap < 1e-18] = .0  # cut the tails
        return heatmap

    def _getPartParams(self, point1, point2):
        center = point1 + (point2 - point1) / 2  # middle point coordinate
        vec = point2 - point1
        theta = np.arctan(vec[1] / vec[0])
        length = np.linalg.norm(vec)
        return center, theta, length

    def visualize_heatmap_target(self, oriImg, heatmap, gt, stride):
        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=.5)
        # plt.plot(gt[:,0], gt[:,1])
        plt.show()


if __name__ == '__main__':
    from utils.util import config

    config['debug_vis'] = False
    config['fname'] = '/data2/data1/zc12345/private_datasets/tsd-max/train-8.mat'
    config['is_test'] = True
    config['save_freq'] = 10
    config['debug'] = True
    dataset = KFDataset(config, transform=transform)
    dataset.load()
    dataLoader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)
    for i, (x, y, kpt_gt, seg_gt) in enumerate(dataLoader):
        print(x.size())  # torch.Size([8, 3, 256, 256])
        # x_np = x.numpy()
        # x_np_0 = x_np[0].swapaxes(0, 1).swapaxes(1, 2)
        # plt.imshow(x_np_0)
        # plt.show()
        print(y.size())  # torch.Size([8, 31, 64, 64])

        print(kpt_gt.size())  # torch.Size([8, 16, 2])
        print(seg_gt.size())  # torch.Size([8, 256, 256])
        exit()
