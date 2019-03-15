import vot
import sys
import torch
import cv2

import resnet
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc, ndimage
from skimage import transform
from torch.autograd import Variable
from pyhog import pyhog
from mpl_toolkits.axes_grid1 import ImageGrid

# Resnet params
resnet_outputlayer = [2, 3]
resnet_numlayers = len(resnet_outputlayer)
resnet_layerweights = [1, 1]
# Resnet init
resnet_model = resnet.ResNet(layers=[3, 4, 6, 3], outlayers=resnet_outputlayer)
resnet_model_dict = resnet_model.state_dict()
resnet_params = torch.load('/home/icv/PycharmProjects/RCF4/resnet34-333f7ec4.pth')

resnet_load_dict = {k: v for k, v in resnet_params.items() if 'fc' not in k}
resnet_model_dict.update(resnet_load_dict)
resnet_model.load_state_dict(resnet_model_dict)
resnet_model.cuda()


def imshow_grid(images, shape=[3, 10]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i] / np.max(images[i]), cmap=plt.cm.gray)  # The AxesGrid object work as a list of axes.

    plt.show()


def findpeaks(response):
    peak = []
    max_sorce = np.max(response)
    pad_response = np.pad(response, (1, 1), 'constant', constant_values=(0, 0))

    for i in range(response.shape[0]):
        for j in range(response.shape[1]):

            if pad_response[i + 1][j + 1] > pad_response[i][j] and pad_response[i + 1][j + 1] > pad_response[i][j + 1]:
                # print("pass 1")
                if pad_response[i + 1][j + 1] > pad_response[i][j + 2] and pad_response[i + 1][j + 1] > \
                        pad_response[i + 1][j]:
                    # print("pass 2")
                    if pad_response[i + 1][j + 1] > pad_response[i + 2][j] and pad_response[i + 1][j + 1] > \
                            pad_response[i + 2][j + 1]:
                        # print("pass 3")
                        if pad_response[i + 1][j + 1] > pad_response[i + 2][j + 2] and pad_response[i + 1][j + 1] > \
                                pad_response[i + 1][j + 2]:
                            if pad_response[i + 1][j + 1] / max_sorce > 0.2:
                                coordinate = [i, j]
                                peak.append(coordinate)
    return peak


def get_search_windows(size, im_size):
    if (size[0] / size[1] > 2):
        # For object with large height
        window_size = np.floor(np.multiply(size, [1 + 0.6, 1 + 2.9]))
        print('Large height')

    elif np.prod(size) / np.prod(im_size) > 0.05:
        window_size = np.floor(size * (1 + 1))
        print('Normal')
    else:
        window_size = np.floor(size * (1 + 1.7))
        print('Small Size')

    return window_size


def get_ori(image, position, wsz, scale_factor=None):
    sz_ori = wsz

    patch_wsz = wsz
    if scale_factor != None:
        patch_wsz = np.floor(patch_wsz * scale_factor)

    y = np.floor(position[0]) - np.floor(patch_wsz[0] / 2) + np.arange(patch_wsz[0], dtype=int)
    x = np.floor(position[1]) - np.floor(patch_wsz[1] / 2) + np.arange(patch_wsz[1], dtype=int)

    x, y = x.astype(int), y.astype(int)

    # check bounds
    x[x < 0] = 0
    y[y < 0] = 0

    x[x >= image.shape[1]] = image.shape[1] - 1
    y[y >= image.shape[0]] = image.shape[0] - 1

    ori = image[np.ix_(y, x)]

    if scale_factor != None:
        ori = misc.imresize(ori, sz_ori.astype(int))

    return ori


def pre_process_image(ori):
    imgMean = np.array([0.485, 0.456, 0.406], np.float)
    imgStd = np.array([0.229, 0.224, 0.225])
    ori = transform.resize(ori, (224, 224))
    ori = (ori - imgMean) / imgStd
    ori = np.transpose(ori, (2, 0, 1))
    ori = torch.from_numpy(ori[None, :, :, :]).float()
    ori = Variable(ori)
    if torch.cuda.is_available():
        ori = ori.cuda()
    return ori


def get_resnet_feature(ori):
    resnet_feature_ensemble = resnet_model(ori)

    return resnet_feature_ensemble


def get_scale_window(image, position, target_size, sfs, scale_window, scale_model_sz):
    # pos = [position[1],position[0]]
    # ts = np.array([target_size[1],target_size[0]])

    out = []
    for i in range(len(sfs)):
        patch_sz = np.floor(target_size * sfs[i])
        scale_patch = get_ori(image, position, patch_sz)
        im_patch_resized = transform.resize(scale_patch, scale_model_sz, mode='reflect')
        temp_hog = pyhog.features_pedro(im_patch_resized, 4)
        out.append(np.multiply(temp_hog.flatten(), scale_window[i]))

    return np.asarray(out)


def get_filter(resnet_feature_ensemble, yf, cos_window):
    resnet_num = []
    resnet_den = []

    for i in range(resnet_numlayers):
        resnet_feature = resnet_feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))

        x = ndimage.zoom(resnet_feature, (
        float(cos_window.shape[0]) / resnet_feature.shape[0], float(cos_window.shape[1]) / resnet_feature.shape[1], 1),
                         order=1)

        x = np.multiply(x, cos_window[:, :, None])
        xf = np.fft.fft2(x, axes=(0, 1))

        resnet_num.append(np.multiply(yf[:, :, None], np.conj(xf)))
        resnet_den.append(np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2)))

    return resnet_num, resnet_den


def get_scale_filter(image, position, target_size, current_scale_factor, scaleFactors, scale_window, model_size, ysf):
    sw = get_scale_window(image, position, target_size, current_scale_factor * scaleFactors, scale_window, model_size)
    swf = np.fft.fftn(sw, axes=[0])
    s_num = np.multiply(ysf[:, None], np.conj(swf))
    s_den = np.real(np.sum(np.multiply(swf, np.conj(swf)), axis=1))

    return s_num, s_den


def get_psr(response):
    y, x = np.unravel_index(response.argmax(), response.shape)

    mask = np.ones(response.shape, dtype=np.bool)
    mask[y - 5:y + 6, x - 5:x + 6] = False
    corr = response.flatten()
    mask = mask.flatten()
    sidelobe = corr[mask]

    mn = sidelobe.mean()
    sd = sidelobe.std()

    psr = (response.max() - mn) / sd

    return psr


def normalize_response(response):
    response_min, response_max = response.min(), response.max()

    nor_response = (response - response_min) / (response_max - response_min)

    return nor_response


def tracking(image, search_pos, target_position, window_size, resnet_num, resnet_den, cos_window, scalefactor,
             update_flag,cell_size,lam):
    theshold = 15.0
    search_factor = 1.1 * scalefactor
    ori = get_ori(image, search_pos, window_size, search_factor)

    ori = pre_process_image(ori)
    resnet_feature_ensemble = get_resnet_feature(ori)

    for i in range(resnet_numlayers):

        feature = resnet_feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))
        x = ndimage.zoom(feature, (
        float(cos_window.shape[0]) / feature.shape[0], float(cos_window.shape[1]) / feature.shape[1], 1), order=1)
        x = np.multiply(x, cos_window[:, :, None])
        xf = np.fft.fft2(x, axes=(0, 1))
        response = np.real(
            np.fft.ifft2(np.divide(np.sum(np.multiply(resnet_num[i], xf), axis=2), (resnet_den[i] + lam)))) * \
                   resnet_layerweights[i]

        if i == 0:
            resnet_response = response
        else:
            resnet_response = np.add(resnet_response, response)

    final_response = normalize_response(resnet_response)

    peaks = findpeaks(final_response)

    resnet_psr = get_psr(resnet_response)

    if len(peaks) > 2 and resnet_psr < 15:
        update_flag = False
    else:
        update_flag = True

    center_h, center_w = np.unravel_index(final_response.argmax(), final_response.shape)

    h_delta, w_delta = [(center_h - final_response.shape[0] / 2) * search_factor * cell_size,
                        (center_w - final_response.shape[1] / 2) * search_factor * cell_size]

    center = [target_position[0] + h_delta, target_position[1] + w_delta]

    return center, update_flag


def scale_variation(image, target_position, target_size, scale_num, scale_den, scale_factor, ScaleFactors, scale_window,
                    model_size,lam):
    sw = get_scale_window(image, target_position, target_size, scale_factor * ScaleFactors, scale_window, model_size)
    swf = np.fft.fftn(sw, axes=[0])
    scale_response = np.real(
        np.fft.ifftn(np.sum(np.divide(np.multiply(scale_num, swf), (scale_den[:, None] + lam)), axis=1)))
    scale_index = np.argmax(scale_response)
    new_scale_factor = scale_factor * ScaleFactors[scale_index]

    return new_scale_factor


def update_position_filter(image, target_position, window_size, scale_factor, position_yf, position_cos_window,
                           resnet_num, resnet_den, update_rate):
    search_factor = 1.1 * scale_factor
    ori = get_ori(image, target_position, window_size, search_factor)
    ori = pre_process_image(ori)
    resnet_feature_ensemble = get_resnet_feature(ori)

    for i in range(resnet_numlayers):
        feature = resnet_feature_ensemble[i].data[0].cpu().numpy().transpose((1, 2, 0))
        x = ndimage.zoom(feature, (
        float(position_cos_window.shape[0]) / feature.shape[0], float(position_cos_window.shape[1]) / feature.shape[1],
        1), order=1)
        x = np.multiply(x, position_cos_window[:, :, None])
        xf = np.fft.fft2(x, axes=(0, 1))
        new_resnet_num = np.multiply(position_yf[:, :, None], np.conj(xf))
        new_resnet_den = np.real(np.sum(np.multiply(xf, np.conj(xf)), axis=2))
        resnet_num[i] = (1 - update_rate) * resnet_num[i] + update_rate * new_resnet_num
        resnet_den[i] = (1 - update_rate) * resnet_den[i] + update_rate * new_resnet_den

    return resnet_num, resnet_den


def update_scale_filter(image, target_position, target_size, scale_num, scale_den, scale_factor, ScaleFactors,
                        scale_window, model_size, scale_ysf, update_rate):
    sw = get_scale_window(image, target_position, target_size, scale_factor * ScaleFactors, scale_window, model_size)
    swf = np.fft.fftn(sw, axes=[0])
    new_s_num = np.multiply(scale_ysf[:, None], np.conj(swf))
    new_s_den = np.real(np.sum(np.multiply(swf, np.conj(swf)), axis=1))

    scale_num = (1 - update_rate) * scale_num + update_rate * new_s_num
    scale_den = (1 - update_rate) * scale_den + update_rate * new_s_den

    return scale_num, scale_den


class RCFTracker:
    def __init__(self, image, region):

        self.target_size = np.array([region.height, region.width])

        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        self.sz = get_search_windows(self.target_size, image.shape[:2])

        # position prediction params
        self.lam = 1e-4
        output_sigma_factor = 0.1
        self.cell_size = 4
        self.interp_factor = 0.01
        self.x_num = []
        self.x_den = []
        self.update_flag = True
        # scale estimation params
        self.current_scale_factor = 1.0
        nScales = 33
        scale_step = 1.02  # step of one scale level
        scale_sigma_factor = 1 / float(4)
        self.interp_factor_scale = 0.01
        scale_model_max_area = 32 * 16
        scale_model_factor = 1.0
        self.min_scale_factor = np.power(scale_step,
                                         np.ceil(np.log(5. / np.min(self.sz)) / np.log(scale_step)))

        self.max_scale_factor = np.power(scale_step,
                                         np.floor(np.log(np.min(np.divide(image.shape[:2],
                                                                          self.target_size)))
                                                  / np.log(scale_step)))

        if scale_model_factor * scale_model_factor * np.prod(self.target_size) > scale_model_max_area:
            scale_model_factor = np.sqrt(scale_model_max_area / np.prod(self.target_size))

        self.scale_model_sz = np.floor(self.target_size * scale_model_factor)

        # Gaussian shaped label for position perdiction
        l1_patch_num = np.floor(self.sz / self.cell_size)
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor / self.cell_size
        grid_y = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
        grid_x = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        self.yf = np.fft.fft2(y, axes=(0, 1))

        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # Gaussian shaped label for scale estimation
        ss = np.arange(nScales) - np.ceil(nScales / 2)
        scale_sigma = np.sqrt(nScales) * scale_sigma_factor
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.scaleFactors = np.power(scale_step, -ss)
        self.ysf = np.fft.fft(ys)
        if nScales % 2 == 0:
            self.scale_window = np.hanning(nScales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(nScales)

        # Extracting hierarchical convolutional features and training
        # get_filter(feature_ensemble,yf,cos_window)
        img = get_ori(image, self.pos, self.sz)
        self.pre_img = img
        img = pre_process_image(img)
        resnet_feature_ensemble = get_resnet_feature(img)

        self.resnet_num, self.resnet_den = get_filter(resnet_feature_ensemble, self.yf, self.cos_window)

        # Extracting the sample feature map for the scale filter and training
        # get_scale_filter(image,position,target_size,current_scale_factor,scaleFactors,scale_window,model_size,ysf)
        self.s_num, self.s_den = get_scale_filter(image, self.pos, self.target_size, self.current_scale_factor,
                                                  self.scaleFactors, self.scale_window, self.scale_model_sz, self.ysf)

    def track(self, image):

        # tracking(image,target_position,window_size,num,den,cos_window,scalefactor)
        self.pos, self.update_flag = tracking(image, self.pos, self.pos, self.sz, self.resnet_num, self.resnet_den,
                                              self.cos_window, self.current_scale_factor, self.update_flag,self.cell_size,self.lam)

        # scale_variation(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size)
        self.current_scale_factor = scale_variation(image, self.pos, self.target_size, self.s_num, self.s_den,
                                                    self.current_scale_factor, self.scaleFactors, self.scale_window,
                                                    self.scale_model_sz,self.lam)

        if self.current_scale_factor < self.min_scale_factor:
            self.current_scale_factor = self.min_scale_factor
        elif self.current_scale_factor > self.max_scale_factor:
            self.current_scale_factor = self.max_scale_factor

        # update
        # update_position_filter(image, target_position, window_size, scale_factor, position_yf, position_cos_window,
        #                      position_num, position_den, update_rate)
        if self.update_flag == True:
            self.resnet_num, self.resnet_den = update_position_filter(image, self.pos, self.sz,
                                                                      self.current_scale_factor, self.yf,
                                                                      self.cos_window,
                                                                      self.resnet_num, self.resnet_den,
                                                                      self.interp_factor)

        # update_scale_filter(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size,scale_ysf,update_rate)
        self.s_num, self.s_den = update_scale_filter(image, self.pos, self.target_size, self.s_num, self.s_den,
                                                     self.current_scale_factor, self.scaleFactors, self.scale_window,
                                                     self.scale_model_sz, self.ysf, self.interp_factor_scale)

        self.final_size = self.target_size * self.current_scale_factor

        return  vot.Rectangle(self.pos[1] - self.final_size[1] / 2,
                              self.pos[0] - self.final_size[0] / 2,
                              self.final_size[1],
                              self.final_size[0])


handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = RCFTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    handle.report(region)
handle.quit()
