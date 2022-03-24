import time
import math
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *

millis = int(round(time.time() * 1000))
torch.manual_seed(millis)
torch.cuda.manual_seed(millis)


def sample_gumbel(shape, tt=torch, eps=1e-20):
    U = Variable(tt.FloatTensor(shape).uniform_(0, 1))
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp, tt=torch, idx_=10):
    y = (logits + sample_gumbel(logits.size(), tt)) / temp
    if idx_ == 0:
        y[:, 3] = -float('inf')
    return F.softmax(y, dim=-1)


def gumbel_softmax(logits, temp, hard, tt=torch, idx_=10):
    y = gumbel_softmax_sample(logits, temp, tt, idx_)  # (batch_size, num_cat)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = tt.FloatTensor(y.size()).zero_().scatter_(1, y_max_idx.data, 1)
        y = Variable(y_hard - y.data, requires_grad=False) + y

    return y, y_max_idx


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class SingleAgent(torch.nn.Module):
    def __init__(self, args):
        super(SingleAgent, self).__init__()
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print("Sharing visual system for each agent.")
            self.beholder = Beholder(args)
        self.video = "video" in args.dataset or "box" in args.dataset
        if self.video:
            self.video_rnn = nn.GRU(args.D_hid, args.D_hid, args.num_layers, batch_first=True)
        self.speaker = Speaker(args)
        self.listener = RnnListener(args)
        self.tt = torch if args.cpu else torch.cuda
        self.unit_norm = args.unit_norm

        self.no_share_bhd = args.no_share_bhd
        self.D_img = args.D_img
        self.D_hid = args.D_hid

    def generate_ec(self, spk_h_img, a_spk_caps_in=0, a_spk_cap_lens=0, spk_sample_how=None):
        with torch.no_grad():
            if self.no_share_bhd:
                spk_h_img = self.beholder1(spk_h_img)  # shared
            else:
                spk_h_img = self.beholder(spk_h_img)  # shared

            spk_logits, comm_action, spk_cap_len_ = self.speaker(spk_h_img, a_spk_caps_in, a_spk_cap_lens,
                                                                 spk_sample_how)  # NOTE argmax / gumbel
        return comm_action

    def generate_ec_video(self, videos):
        # bs, num_dist = b_lsn_imgs.size()
        bs = num_dist = len(videos)
        video_lens = torch.LongTensor([video.size(0) for video in videos])
        video_tensor = torch.from_numpy(pad_sequences(videos, dtype='float32')).cuda()
        video_tensor = video_tensor.view(bs * max(video_lens), -1)
        if self.no_share_bhd:
            video_h = self.beholder1(video_tensor)  # shared
        else:
            video_h = self.beholder(video_tensor)  # shared
        video_h = video_h.view(bs, max(video_lens), -1)

        outs = [video_h[i, 0:video_lens[i], :].mean(0) for i in range(bs)]
        out = torch.stack(outs, 0)
        spk_h_img = torch.index_select(out, 0, a_spk_img)
        spk_logits, comm_action, spk_cap_len_ = self.speaker(spk_h_img, a_spk_caps_in, a_spk_cap_lens, spk_sample_how,
                                                             video_out=video_h)  # NOTE argmax / gumbel

    def forward_image(self, data1, spk_sample_how):
        a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens = data1[:4]  # spk_imgs : (batch_size, 2048)

        num_dist = b_lsn_imgs.size()[1]

        if self.no_share_bhd:
            spk_h_img = self.beholder1(a_spk_img)  # shared
        else:
            spk_h_img = self.beholder(a_spk_img)  # shared

        spk_logits, comm_action, spk_cap_len_ = self.speaker(spk_h_img, a_spk_caps_in, a_spk_cap_lens,
                                                             spk_sample_how)  # NOTE argmax / gumbel

        lenlen = False
        if lenlen:
            print(spk_cap_len_[:10])
            end_idx = torch.max(torch.ones(spk_cap_len_.size()).cuda(), (spk_cap_len_ - 2).float())
            end_idx_ = torch.arange(0, end_idx.size(0)).cuda() * spk_logits.size(1) + end_idx.int()

            end_loss_ = 3 * torch.ones(end_idx_.size()).long().cuda()
        else:
            end_idx_ = 0
            end_loss_ = 0

        lsn_imgs = b_lsn_imgs.view(-1, self.D_img)
        if self.no_share_bhd:
            lsn_h_imgs = self.beholder2(lsn_imgs)
        else:
            lsn_h_imgs = self.beholder(lsn_imgs)
        lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid)
        rnn_hid = self.listener(comm_action[:, :-1], spk_cap_len_ - 1, spk_logits[:, :-1, :])
        rnn_hid = rnn_hid.unsqueeze(1).repeat(1, num_dist, 1)  # (batch_size, num_dist, D_hid)

        return spk_logits, (rnn_hid, lsn_h_imgs), comm_action, (end_idx_, end_loss_), (
        torch.min(spk_cap_len_.float()), torch.mean(spk_cap_len_.float()), torch.max(spk_cap_len_.float()))

    def forward_video(self, data1, spk_sample_how):
        videos, a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens = data1[:5]  # spk_imgs : (batch_size, 2048)
        bs, num_dist = b_lsn_imgs.size()
        video_lens = torch.LongTensor([video.size(0) for video in videos])
        video_tensor = torch.from_numpy(pad_sequences(videos, dtype='float32')).cuda()
        video_tensor = video_tensor.view(bs * max(video_lens), -1)
        if self.no_share_bhd:
            video_h = self.beholder1(video_tensor)  # shared
        else:
            video_h = self.beholder(video_tensor)  # shared
        video_h = video_h.view(bs, max(video_lens), -1)

        """
        import pdb; pdb.set_trace() 
        pack = torch.nn.utils.rnn.pack_padded_sequence(video_h, video_lens.cpu(), batch_first=True, enforce_sorted=False)
        video_out, h_n = self.video_rnn(pack)
        out = h_n.transpose(0,1).contiguous().view(bs, self.D_hid)
        """

        outs = [video_h[i, 0:video_lens[i], :].mean(0) for i in range(bs)]
        out = torch.stack(outs, 0)

        # out = video_h[:,0,:]

        spk_h_img = torch.index_select(out, 0, a_spk_img)
        spk_logits, comm_action, spk_cap_len_ = self.speaker(spk_h_img, a_spk_caps_in, a_spk_cap_lens, spk_sample_how,
                                                             video_out=video_h)  # NOTE argmax / gumbel

        lenlen = False
        if lenlen:
            print(spk_cap_len_[:10])
            end_idx = torch.max(torch.ones(spk_cap_len_.size()).cuda(), (spk_cap_len_ - 2).float())
            end_idx_ = torch.arange(0, end_idx.size(0)).cuda() * spk_logits.size(1) + end_idx.int()

            end_loss_ = 3 * torch.ones(end_idx_.size()).long().cuda()
        else:
            end_idx_ = 0
            end_loss_ = 0

        """
        lsn_imgs = b_lsn_imgs.view(-1, self.D_img)
        if self.no_share_bhd:
            lsn_h_imgs = self.beholder2(lsn_imgs)
        else:
            lsn_h_imgs = self.beholder(lsn_imgs)
        lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid)
        """
        lsn_h_imgs = torch.index_select(out, 0, b_lsn_imgs.view(-1))
        lsn_h_imgs = lsn_h_imgs.view(bs, num_dist, -1)
        rnn_hid = self.listener(comm_action[:, :-1], spk_cap_len_ - 1, spk_logits[:, :-1, :])
        rnn_hid = rnn_hid.unsqueeze(1).repeat(1, num_dist, 1)  # (batch_size, num_dist, D_hid)

        return spk_logits, (rnn_hid, lsn_h_imgs), comm_action, (end_idx_, end_loss_), (
        torch.min(spk_cap_len_.float()), torch.mean(spk_cap_len_.float()), torch.max(spk_cap_len_.float()))

    def forward(self, data1, spk_sample_how):
        return self.forward_video(data1, spk_sample_how) if self.video else self.forward_image(data1, spk_sample_how)


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe, alex=False):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64 * 6 * 6 if alex else 64 * 7 * 7
        self.alex = alex

        self.conv1 = nn.Conv2d(256 if alex else 512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        if not self.alex:
            x = x.view(-1, 512, 7, 7)
        else:
            x = x.view(-1, 256, 6, 6)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class Beholder(torch.nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        if args.dataset == "coco" or "tsm" in args.dataset or "box" in args.dataset:
            self.img_to_hid = torch.nn.Linear(args.D_img, args.D_hid)  # shared visual system
        elif args.dataset in ["imagenet", "coco_2014", "cc"]:
            self.img_to_hid = None
        else:
            self.img_to_hid = ResnetVisualEncoder(args.D_hid, alex="alex" in args.dataset)

        self.unit_norm = args.unit_norm
        self.drop = nn.Dropout(p=args.dropout)
        self.two_fc = args.two_fc
        if self.two_fc:
            self.hid_to_hid = torch.nn.Linear(args.D_hid, args.D_hid)

    def forward(self, img):
        h_img = img
        if self.img_to_hid is None: return h_img
        h_img = self.img_to_hid(h_img)

        h_img = self.drop(h_img)

        if self.two_fc:
            h_img = self.hid_to_hid(F.relu(h_img))

        if self.unit_norm:
            norm = torch.norm(h_img, p=2, dim=1, keepdim=True).detach() + 1e-9
            h_img = h_img / norm.expand_as(h_img)
        return h_img


class RnnListener(torch.nn.Module):
    def __init__(self, args):
        super(RnnListener, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True) if args.num_directions == 1 else \
            nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)
        self.hid_to_hid = nn.Linear(args.num_directions * args.D_hid, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers
        self.num_directions = args.num_directions
        self.vocab_size = args.vocab_size
        self.unit_norm = args.unit_norm

        self.tt = torch if args.cpu else torch.cuda

    def forward(self, spk_msg, spk_msg_lens, spk_logit=0):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size()[0]
        seq_len = spk_msg.size()[1]

        h_0 = Variable(self.tt.FloatTensor(self.num_layers * self.num_directions, batch_size, self.D_hid).zero_())

        spk_msg_emb = torch.matmul(spk_logit, self.emb.weight)
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = torch.nn.utils.rnn.pack_padded_sequence(spk_msg_emb, spk_msg_lens.cpu(), batch_first=True,
                                                       enforce_sorted=False)

        _, h_n = self.rnn(pack, h_0)

        h_n = h_n[-self.num_directions:, :, :]
        out = h_n.transpose(0, 1).contiguous().view(batch_size, self.num_directions * self.D_hid)
        out = self.hid_to_hid(out)

        if self.unit_norm:
            norm = torch.norm(out, p=2, dim=1, keepdim=True).detach() + 1e-9
            out = out / norm.expand_as(out)

        return out


class Speaker(torch.nn.Module):
    def __init__(self, args):
        super(Speaker, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)

        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)

        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.num_layers = args.num_layers
        self.drop = nn.Dropout(p=args.dropout)

        self.vocab_size = args.vocab_size
        self.temp = args.temp
        self.hard = args.hard
        self.tt = torch if args.cpu else torch.cuda
        self.tt_ = torch
        self.seq_len = args.seq_len

        self.attn = DotAttn()
        self.h = nn.Linear(args.D_hid, args.D_hid)
        self.h1 = nn.Linear(args.D_hid + args.D_emb, args.D_emb)

    def forward(self, h_img, caps_in, caps_in_lens, sample_how, video_out=None):

        batch_size = h_img.size()[0]  # caps_in.size()[0]

        h_img = h_img.view(1, batch_size, self.D_hid).repeat(self.num_layers, 1, 1)

        initial_input = self.emb(torch.ones([batch_size, 1], dtype=torch.int64).cuda() * 2)
        out_, hid_ = self.rnn(initial_input, h_img)
        logits_ = []
        labels_ = []
        for idx in range(self.seq_len):
            logit_ = self.hid_to_voc(out_.view(-1, self.D_hid))
            c_logit_, comm_label_ = gumbel_softmax(logit_, self.temp, self.hard, self.tt, idx)
            input_ = torch.matmul(c_logit_.unsqueeze(1), self.emb.weight)
            if video_out is not None:
                # video_tensor, _ = pad_packed_sequence(video_out)
                weighted_lang, lang_attn = self.attn(self.drop(video_out), self.h(hid_[0]))
                weighted_lang = weighted_lang.unsqueeze(1)
                input_ = torch.cat((input_, weighted_lang), dim=-1)
                input_ = self.h1(input_)
            out_, hid_ = self.rnn(input_, hid_)
            logits_.append(c_logit_.unsqueeze(1))
            labels_.append(comm_label_)
        logits_ = torch.cat(logits_, dim=1)
        labels_ = torch.cat(labels_, dim=-1)
        tmp = torch.zeros(logits_.size(-1))
        tmp[3] = 1
        logits_[:, -1, :] = tmp
        labels_[:, -1] = 3
        pad_g = ((labels_ == 3).cumsum(1) == 0)
        labels_ = pad_g * labels_
        pad_ = torch.zeros(logits_.size()).cuda()
        pad_[:, :, 0] = 1
        logits_ = torch.where(pad_g.unsqueeze(-1).repeat(1, 1, logits_.size(-1)), logits_, pad_)

        cap_len = pad_g.cumsum(1).max(1).values + 1

        return logits_, labels_, cap_len
