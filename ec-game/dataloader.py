import copy
import json
import numpy as np
from collections import OrderedDict
import time
import torch
from torch.autograd import Variable

from util import *

random = np.random
random.seed(1234)


def next_batch_joint(images, batch_size, num_dist, tt, t=0):
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = [], [], [], [], []
    total_indices = []
    keys = range(len(images))
    assert len(keys) >= num_dist
    if t: img = images.div(images.norm(dim=1, keepdim=True))
    if len(images) > 50000:
        img_indices = random.choice(len(images), 50000, replace=False)
        images = torch.index_select(images, 0, torch.tensor(img_indices))
    for batch_idx in range(batch_size):
        if t:
            spk_img = random.randint(len(images))
            sim = img.matmul(img[spk_img])
            sim[spk_img] = -100000
            p = torch.nn.functional.softmax(sim)
            img_indices = [spk_img] + list(random.choice(range(len(images)), num_dist-1, replace=False, p=p.numpy()))
            which = 0
        else:
            # img_indices = random.permutation(len(images))[:num_dist]
            img_indices = random.choice(len(images), num_dist, replace=False)
            which = random.randint(0, num_dist)  # (1)
            spk_img = img_indices[which]

        spk_imgs.append(spk_img)  # (batch_size, 2048)
        lsn_imgs += list(img_indices)  # batch_size * num_dist
        whichs.append(which)  # (batch_size)
    # if len(keys) == num_dist: # easier way out
    #     whichs = list(range(num_dist))
    #     spk_imgs = keys
    #     lsn_imgs = [keys for _ in range(num_dist)]
    spk_imgs = torch.index_select(images, 0, torch.tensor(spk_imgs)).numpy()
    lsn_imgs = torch.index_select(images, 0, torch.tensor(lsn_imgs)).numpy()  # view(batch_size, num_dist,-1).numpy()
    whichs = np.array(whichs)
    spk_imgs = Variable(torch.from_numpy(spk_imgs), requires_grad=False)  # .view(batch_size, -1)
    lsn_imgs = torch.from_numpy(lsn_imgs)
    lsn_imgs = Variable(lsn_imgs, requires_grad=False).view(batch_size, num_dist, -1)
    whichs = Variable(torch.LongTensor(whichs), requires_grad=False).view(batch_size)
    if tt == torch.cuda:
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        whichs = whichs.cuda()
    return (spk_imgs, lsn_imgs, 0, 0, 0, 0, 0, whichs)


def next_batch_joint_video(videos, batch_size, num_dist, tt, t=0):
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = [], [], [], [], []
    total_indices = []
    # keys = range(len(images))
    keys = random.permutation(len(videos))[:batch_size]  # fix batch size of videos, distractors from the batch
    assert len(keys) >= num_dist
    if len(keys) == num_dist:
        whichs = spk_imgs = list(range(num_dist))
        lsn_imgs = [list(range(num_dist)) for _ in range(num_dist)]
    else:
        for batch_idx in range(batch_size):
            img_indices = random.permutation(batch_size)[:num_dist]
            which = random.randint(0, num_dist)  # (1)
            spk_imgs.append(img_indices[which])  # list of (batchsize, )
            lsn_imgs.append(list(img_indices))  # list of (batchsize, num_dist)
            whichs.append(which)  # (batch_size)
    batch_videos = [videos[key] for key in keys]
    whichs = Variable(torch.LongTensor(np.array(whichs)), requires_grad=False).view(batch_size)
    spk_imgs = Variable(torch.LongTensor(np.array(spk_imgs)), requires_grad=False)
    lsn_imgs = Variable(torch.LongTensor(np.array(lsn_imgs)), requires_grad=False)
    if tt == torch.cuda:
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        whichs = whichs.cuda()
    return (batch_videos, spk_imgs, lsn_imgs, 0, 0, 0, 0, 0, whichs)


def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans
