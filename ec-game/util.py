import codecs
import os
import sys
import time
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sum_num_captions(org):
    return sum([len(x) for x in org])


def get_coco_idx():
    a, b = 56644, 56643
    a_, b_ = [], []

    cand = 0
    while len(a_) < 14500:
        if not cand in a_:
            a_.append(cand)
            cand = cand + 4
        else:
            cand += 1
        cand = cand % 56644

    cand = 0
    while len(b_) < 14500:
        if not cand in b_:
            b_.append(cand)
            cand += 4
        else:
            cand += 1
        cand = cand % 56643

    assert (len(set(a_)) == 14500)
    assert (len(set(b_)) == 14500)

    return a_, b_


def recur_mkdir(dir):
    ll = dir.split("/")
    ll = [x for x in ll if x != ""]
    for idx in range(len(ll)):
        ss = "/".join(ll[0:idx + 1])
        check_mkdir("/" + ss)


class Logger(object):
    def __init__(self, path, no_write=False, no_terminal=False):
        self.no_write = no_write
        if self.no_write:
            print("Don't write to file")
        else:
            self.log = codecs.open(path + "log.log", "wb", encoding="utf8")

        self.no_terminal = no_terminal
        self.terminal = sys.stdout

    def write(self, message):
        if not self.no_write:
            self.log.write(message)
        if not self.no_terminal:
            self.terminal.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def check_dataset_sanity(args):
    assert args.dataset == "coco" or args.dataset == "multi30k"
    if args.dataset == "coco":
        assert (args.src, args.trg) == ("en", "jp") or (args.src, args.trg) == ("jp", "en")
    elif args.dataset == "multi30k":
        assert (args.src, args.trg) == ("en", "de") or (args.src, args.trg) == ("de", "en")


def sort_per_len(caps):
    lens = [(idx, len(cap)) for idx, cap in enumerate(caps)]
    lens.sort(key=lambda x: x[1], reverse=True)
    lens = np.array([x[0] for x in lens])
    assert len(lens) == len(caps)
    return lens


def trim_caps(caps, minlen, maxlen):
    new_cap = [[cap for cap in cap_i if len(cap) <= maxlen and len(cap) >= minlen] for cap_i in caps]
    print("Before : {} captions / After : {} captions".format(sum_num_captions(caps), sum_num_captions(new_cap)))
    return new_cap


def print_params(names, sizes):
    agents = "l1_agent l2_agent".split()
    comps = "speaker listener beholder".split()

    dd = OrderedDict()
    for aa in agents:
        dd[aa] = {}
        for cc in comps:
            dd[aa][cc] = {}

    for name, size in zip(names, sizes):
        name_ = name.split(".")
        aa, cc, rest = name_[0], name_[1], ".".join(name_[2:])
        dd[aa][cc][rest] = "{} ({})".format(rest, size[0]) if len(size) == 1 else "{} ({}, {})".format(rest, size[0],
                                                                                                       size[1])

    ss = ""
    for aa in agents:
        ss += "\t{}\n".format(aa)
        for cc in comps:
            if len(dd[aa][cc]) > 0:
                ss += "\t\t{}:\t".format(cc)
                for rr in sorted(dd[aa][cc].keys()):
                    ss += dd[aa][cc][rr] + ", "
                ss += "\n"

    return ss


def print_captions(gen_indices, i2w, joiner):
    return [joiner.join([i2w[ii] for ii in gen_idx]).replace("@@ ", "") for gen_idx in gen_indices]


def decode(gen_indices, i2w):
    return [" ".join([i2w[ii] for ii in gen_idx]).replace("@@ ", "") for gen_idx in gen_indices]


def pick(i1, i2, whichs):
    res = []
    img = [i1, i2]
    for idx, which in enumerate(whichs):
        res.append(img[which][idx])
    return res


def idx_to_onehot(indices, nb_digits):  # input numpy array
    y = torch.LongTensor(indices).view(-1, 1)
    y_onehot = torch.FloatTensor(indices.shape[0], nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


def max_logit_to_onehot(logits):
    max_element, max_idx = torch.max(logits.cuda(), 1)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    onehot.scatter_(1, max_idx.data.long().cpu(), 1)
    onehot = Variable(torch.FloatTensor(onehot), requires_grad=False).cuda()
    return onehot, max_idx.data


def sample_logit_to_onehot(logits):
    indices = torch.multinomial(logits, 1)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    for ii, jj in enumerate(indices.data.cpu().numpy().flatten().tolist()):
        onehot[ii][jj] = 1
    onehot = Variable(onehot, requires_grad=False).cuda()
    return onehot, indices.data


def logit_to_acc(logits, y):  # logits: [batch_size, num_of_classes]
    y_max, y_max_idx = torch.max(logits, 1)  # [batch_size]
    eq = torch.eq(y_max_idx, y)
    acc = float(eq.sum().data) / float(eq.nelement())
    return acc


def logit_to_top_k(logits, y, k):  # logits: [batch_size, num_of_classes]
    logits_sorted, indices = torch.sort(logits, 1, descending=True)
    y = y.view(-1, 1)
    indices = indices[:, :k]
    y_big = y.expand(indices.size())
    eq = torch.eq(indices, y_big)
    eq2 = torch.sum(eq, 1)
    return eq2.sum().data[0], eq2.nelement()


def loss_and_acc(logits, labels, loss_fn):
    loss = loss_fn(logits, labels)
    acc = logit_to_acc(logits, labels)
    return (loss, acc)


def loss_acc_dict():
    return {
        "spk": { \
            "loss": 0}, \
        "lsn": { \
            "loss": 0, \
            "acc": 0} \
        }


def loss_acc_meter():
    return {
        "spk": { \
            "loss": AverageMeter()}, \
        "lsn": { \
            "loss": AverageMeter(), \
            "acc": AverageMeter()} \
        }


def get_loss_dict():
    return {"l1": loss_acc_dict(), "l2": loss_acc_dict()}


def get_log_loss_dict():
    return {"l1": loss_acc_meter(), "l2": loss_acc_meter()}


def get_loss_dict_():
    return {"loss": 0, "accuracy": 0, "average_len": 0}


def get_log_loss_dict_():
    return {"loss": AverageMeter(), "accuracy": AverageMeter(), "average_len": AverageMeter(), "ts": AverageMeter()}


def get_avg_from_loss_dict_(log_loss_dict):
    res = get_loss_dict_()
    for k, v in log_loss_dict.items():
        res[k] = v.avg
    return res


def get_avg_from_loss_dict(log_loss_dict):
    res = get_loss_dict()
    for k1, v1 in log_loss_dict.items():  # en_agent / fr_agent
        for k2, v2 in v1.items():  # spk / lsn
            for k3, v3 in v2.items():  # loss / acc
                res[k1][k2][k3] = v3.avg
    return res


def print_loss_(epoch, alpha, avg_loss_dict, mode="train"):
    prt_msg = "epoch {:5d} {} ".format(epoch, mode)
    prt_msg += "| loss"
    prt_msg += " {:.4f}".format(avg_loss_dict["loss"])
    prt_msg += "| prediction accuracy"
    prt_msg += " {:.2f}%".format(avg_loss_dict["accuracy"])
    prt_msg += "| average message length"
    prt_msg += " {:.4f}".format(avg_loss_dict["average_len"])
    prt_msg += " |"
    return prt_msg


def print_loss(epoch, alpha, avg_loss_dict, mode="train"):
    prt_msg = "epoch {:5d} {} ".format(epoch, mode)
    for agent in "l1 l2".split():
        prt_msg += "| "  # en_agent / fr_agent
        for person in "spk lsn".split():
            prt_msg += " {}_{}".format(agent, person)  # spk / lsn
            if person == "spk":
                prt_msg += " {:.3f}".format(avg_loss_dict[agent][person]["loss"])
            elif person == "lsn":
                prt_msg += " {:.3f} * {} = {:.3f}".format(avg_loss_dict[agent][person]["loss"], alpha,
                                                          avg_loss_dict[agent][person]["loss"] * alpha)
                prt_msg += " {:.2f}%".format(avg_loss_dict[agent][person]["acc"])
            prt_msg += " |"
    return prt_msg


def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


def check_mkdir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)


def idx_to_emb(idx, maxmax, tt):
    ans = tt.ByteTensor(len(idx), maxmax).fill_(0)
    for aaa, iii in enumerate(idx):
        ans[aaa][iii] = 1
    return Variable(ans, requires_grad=False)


def remove_duplicate(data):
    hash_0 = list(np.round(data[:, 0].numpy(), 3))
    hash_1 = list(np.round(data[:, 1].numpy(), 3))
    hash_2 = list(np.round(data[:, 2].numpy(), 3))
    hash_1000 = list(np.round(data[:, 1000].numpy(), 3))
    hash_2046 = list(np.round(data[:, 2046].numpy(), 3))
    hash_2047 = list(np.round(data[:, 2047].numpy(), 3))

    seen_e2i = {}
    string_ = []
    for idx in range(len(hash_0)):
        keystr = str(hash_0[idx]) + '/' + str(hash_1[idx]) + '/' + str(hash_2[idx]) + '/' + str(
            hash_1000[idx]) + '/' + str(hash_2046[idx]) + '/' + str(hash_2047[idx])
        if keystr in seen_e2i:
            string_.append([seen_e2i[keystr], idx])
        else:
            seen_e2i[keystr] = idx

    string_2 = []
    for pair in string_:
        if torch.sum(torch.abs(data[pair[0]] - data[pair[1]])).numpy() < 15:
            string_2.append(pair)
    s = set([i[-1] for i in string_2])
    index_ = []
    for i in range(len(data)):
        if i not in s:
            index_.append(i)
    data = torch.index_select(data, 0, torch.tensor(index_, dtype=torch.int64))
    return data[:-10000], data[-10000:]


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
