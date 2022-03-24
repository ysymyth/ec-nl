import sys
import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn

from util import *
from models import *
from dataloader import *
from forward import *

random = np.random
random.seed(1234)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    # main ones to change
    parser.add_argument("--dataset", type=str, default="cc", help="Which Image Dataset To Use EC Pretraining")
    parser.add_argument("--vocab_size", type=int, default=4035, help="EC vocab size")
    parser.add_argument("--seq_len", type=int, default=15, help="Max Len")
    parser.add_argument("--save_every", type=int, default=100, help="Save model output.")
    parser.add_argument("--num_games", type=int, default=1000,  help="Total number of batches to train for")
    parser.add_argument("--extract", type=str, default="", help="extract")
    parser.add_argument("--wandb", type=int, default=0, help="use wandb")

    # others
    parser.add_argument("--gpuid", type=int, default=0, help="Which GPU to run")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--two_fc", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size For Training")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Batch size For Validation")
    parser.add_argument("--num_dist", type=int, default=256, help="Number of Distracting Images For Training")
    parser.add_argument("--num_dist_", type=int, default=128, help="Number of Distracting Images For Validation")
    parser.add_argument("--D_img", type=int, default=2048, help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=512, help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=256, help="Token embedding dimensionality")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout keep probability")
    parser.add_argument("--temp", type=float, default=1.0, help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=False, help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--TransferH", action="store_true", default=False, help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--print_every", type=int, default=50, help="Save model output.")
    parser.add_argument("--ECemb", type=int, default=5000, help="Set The EC Embedding Size")
    parser.add_argument("--valid_every", type=int, default=100, help="Validate model every k batches")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_directions", type=float, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--unit_norm", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--loss_type", type=str, default="xent")
    parser.add_argument("--fix_spk", action="store_true", default=False)
    parser.add_argument("--fix_bhd", action="store_true", default=False)
    parser.add_argument("--no_share_bhd", action="store_true", default=False)
    parser.add_argument("--sample_how", type=str, default="gumbel")
    parser.add_argument("--load", type=str, default="", help="load weights")
    parser.add_argument("--no_write", action="store_true", default=False)
    parser.add_argument("--no_terminal", action="store_true", default=False)
    parser.add_argument("--reset_lsn", type=int, default=-1, help="reset listener")

    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    # TODO: provide pt files
    dataset2pt = {'coco_2014': './data/coco.pt', 'cc': './data/cc.pt'}
    feat_path = data_path = dataset2pt[args.dataset]
    data = torch.load(feat_path)
    if args.dataset == "coco_2014":
        train_data, valid_data = data[:50000], data[-5000:]
    elif args.dataset == "cc":
        train_data, valid_data = data[:-50000], data[-50000:]

    print("Dataset Loaded")

    mill = int(round(time.time() * 100000)) % 100000
    name = f'{args.dataset}_vocab_{args.vocab_size}_seq_{args.seq_len}_reset_{args.reset_lsn}_nlayers_{args.num_layers}'
    run = str(mill)
    if args.wandb:
        import wandb
        wandb.init(project='EC-games', name=run, group=name, config=args)

    path_dir = f"./ckpt/{name}/run{run}/"
    os.makedirs(path_dir, exist_ok=True)

    sys.stdout = Logger(path_dir, no_write=args.no_write, no_terminal=args.no_terminal)
    print(args)
    print(name)

    model = SingleAgent(args)
    print(model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    loss_fn = {'xent': nn.CrossEntropyLoss(), 'mse': nn.MSELoss(), 'mrl': nn.MarginRankingLoss(),
               'mlml': nn.MultiLabelMarginLoss(), 'mml': nn.MultiMarginLoss()}
    tt = torch
    if not args.cpu:
        loss_fn = {k: v.cuda() for (k, v) in loss_fn.items()}
        tt = torch.cuda

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = -1
    train_loss_dict_ = get_log_loss_dict_()

    for epoch in range(args.num_games):
        t = 0
        loss = forward_joint(train_data, model, train_loss_dict_, args, loss_fn, args.num_dist, tt, None, t)
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        # log training info
        if epoch % args.print_every == 0:
            avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
            print(print_loss_(epoch, args.alpha, avg_loss_dict_, "train"))
            train_loss_dict_ = get_log_loss_dict_()
            train_dict = {'train_' + k: float(avg_loss_dict_[k]) for k in avg_loss_dict_}
            train_dict.update({'epoch': epoch})

        # eval
        if epoch % args.valid_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss_dict_ = get_log_loss_dict_()
                for idx in range(args.print_every):
                    _ = forward_joint(valid_data, model, valid_loss_dict_, args, loss_fn, args.num_dist_, tt, t=t)
                avg_loss_dict_ = get_avg_from_loss_dict_(valid_loss_dict_)
                valid_dict = {'valid_' + k: float(avg_loss_dict_[k]) for k in avg_loss_dict_}
                print(valid_dict)
                train_dict.update(valid_dict)
                s_new = print_loss_(epoch, args.alpha, avg_loss_dict_, "valid")
                print(s_new)

                if float(s_new.split()[-6][:-2]) > 0:  # True
                    # save model
                    if epoch % args.save_every == 0:
                        path = path_dir + "model_{}_{}_{}.pt".format(float(s_new.split()[-6][:-2]), epoch,
                                                                     args.vocab_size)
                        with open(path, "wb") as path_model:
                            torch.save(model.state_dict(), path_model)
                        print("Epoch :", epoch, "Prediction Accuracy =", float(s_new.split()[-6][:-2]),
                              "Saved to Path :", path_dir)

                        # extract emergent corpus
                        for dataset in ["coco2014", "cc"]:
                            if dataset in args.extract:
                                load_data = torch.load(dataset2pt[dataset])
                                bs = args.batch_size
                                extract = []
                                for ep in range(len(load_data) // bs + 1):
                                    data = load_data[bs * ep: bs * (ep + 1)]
                                    if isinstance(data, list):
                                        ec = model.generate_ec_video(data)
                                    else:
                                        data = data.cuda()
                                        ec = model.generate_ec(data)
                                    extract.append(ec.detach().cpu())
                                extract = torch.cat(extract, 0)
                                print(extract.size())
                                torch.save(extract, path + f'-{dataset}.pt')
                                extract = None

                    if args.TransferH:
                        args.hard = True

        if epoch % args.print_every == 0 and args.wandb:
            wandb.log(train_dict)
        model.train()

        if args.reset_lsn != -1 and epoch % args.reset_lsn == 0:
            print("Reset listener", epoch)
            for m in model.listener.children():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    end_time = time.time()
    print("Total Runtime :", end_time - start_time)
