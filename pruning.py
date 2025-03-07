import argparse
import os
import sys
import time

from helpers.utils import (
    check_dirs_exist,
    get_device,
    accuracy,
    load_model,
    save_model,
    print_nonzeros,
    set_seeds,
    Logger
)
from helpers import dataset
import models
from helpers.trainer import Trainer
from helpers.pruner import FiltersPruner
from distillers_zoo import (
    KLDistiller,
    Similarity,
    Attention,
    HighAttenPreserving,
    HAPPlotter,
    AFDBuilder
)

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description='Attention Distilled With Pruned Model Process')
parser.add_argument('--n-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--t-model', type=str, default=None)
parser.add_argument('--s-model', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160])
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.2, 0.2, 0.2])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--prune-mode', type=str, default='None')
parser.add_argument('--soft-prune', action='store_true', default=False)  # Do soft pruning or not
parser.add_argument('--prune-rates', nargs='+', type=float, default=[1.0])  # No prune by default
parser.add_argument('--samp-batches', type=int, default=None)  # Sample batches to compute gradient for pruning. Use
# all batches by default
parser.add_argument('--use-actPR', action='store_true', default=False)  # Compute actual pruning rates for conv layers
# or not
parser.add_argument('--use-greedy', action='store_true', default=False)  # Prune filters by greedy or independent
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # Do pruning process once by default
parser.add_argument('--distill', type=str, default='None')  # Which distillation methods to use
parser.add_argument('--hap-ts', type=int, default=3)  # Number of sampled teacher layers for "HAP" distillation
parser.add_argument('--kd-t', type=float, default=4.0)  # Temperature for KL distillation
parser.add_argument('--alpha', type=float, default=0.9)  # For KL-divergence distillation
parser.add_argument('--betas', nargs='+', type=float, default=[50.0])  # For custom-method distillation
parser.add_argument('--gamma', type=float, default=0.5)  # For our filter pruning method "N-G-GM"
parser.add_argument('--t-path', type=str, default=None)  # The .pt file path of teacher model
parser.add_argument('--s-path', type=str, default=None)  # The .pt file path of student model
parser.add_argument('--s-copy-t', action='store_true', default=False)  # During self-distillation, whether student
# copy teacher during initialization
parser.add_argument('--log-name', type=str, default='logs.txt')  # The name of the log file
parser.add_argument('--dev-idx', type=int, default=0)  # The index of the used cuda device
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = f'saves/{args.log_name}'
if args.s_model is None:  # Pruning + Self Distillation
    args.s_model = args.t_model
    if args.s_copy_t:
        args.s_path = args.t_path


class PrunedModelTrainer(Trainer):
    """  A trainer for gradually self-distillation combined with attention mechanism and hard or soft pruning. """
    def __init__(self, t_model, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_model = t_model
        self.s_model = self.model
        self.writer = writer

        self.do_prune = self.args.prune_mode is not 'None'
        self.do_dist = self.args.distill is not 'None'
        self.do_soft_prune = self.args.soft_prune
        self.do_hard_prune = self.do_prune and not self.do_soft_prune

        self.criterion_cls = nn.CrossEntropyLoss()
        if self.do_dist:
            self.criterion_div = KLDistiller(self.args.kd_t)
            self.criterion_kd, self.is_group, self.is_block = self._init_kd(self.args.distill)

        self.s_pruner = FiltersPruner(
            self.s_model,
            self.optimizer,
            self.train_loader,
            self.logger,
            gamma=self.args.gamma,
            samp_batches=self.args.samp_batches,
            device=self.device,
            use_actPR=self.args.use_actPR,
            use_greedy=self.args.use_greedy
        )
        self.last_epoch = None

        self.t_model.eval()
        self.t_model = self.t_model.to(self.device)

    def _mask_pruned_filters_grad(self):
        conv_mask = self.s_pruner.get_conv_mask()
        for name, module in self.s_model.named_modules():
            if name in conv_mask:
                grad = module.weight.grad
                grad.data *= conv_mask[name]

    def _init_kd(self, method):
        is_group = False
        is_block = False
        if method == 'hap':
            is_block = True
            criterion = [HighAttenPreserving()]
        elif method == 'kd':
            is_group = True
            criterion = [KLDistiller(T=self.args.kd_t)]
        elif method == 'sp':
            is_group = True
            criterion = [Similarity()]
        elif method == 'at':
            is_group = True
            criterion = [Attention(dataset=self.args.dataset)]
        elif method == 'afd':
            is_block = True
            AFD = AFDBuilder()(args, t_model=self.t_model, s_model=self.s_model)
            criterion = [AFD]
            self.optimizer.add_param_group({'params': AFD.parameters()})
        else:
            raise NotImplementedError(method)
        return criterion, is_group, is_block

    def _get_dist_feat(self, method, s_feat, t_feat, s_logit, t_logit):
        t_feat = [f.detach() for f in t_feat]
        if method == 'hap':
            n = self.args.hap_ts
            s_f = [s_feat[1:-1]]
            t_f = [t_feat[-n-1:-1]]
        elif method == 'kd':
            s_f = [s_logit]
            t_f = [t_logit]
        elif method == 'at':
            if self.args.dataset == 'imagenet':
                s_f = [s_feat[-3:-1]]
                t_f = [t_feat[-3:-1]]
            else:
                s_f = [s_feat[1:-1]]
                t_f = [t_feat[1:-1]]
        elif method == 'sp':
            s_f = [[s_feat[-2]]]
            t_f = [[t_feat[-2]]]
        elif method == 'afd':
            s_f = [s_feat[1:-1]]
            t_f = [t_feat[1:-1]]
        else:
            raise NotImplementedError(method)
        return s_f, t_f

    def _get_loss_and_backward(self, batch):
        input, target = batch

        # Get the total_loss and backward
        if self.do_dist:
            # Do different kinds of distillation according to "args.distill"
            betas = self.args.betas
            s_feat, s_logit = self.s_model(input, is_group_feat=self.is_group, is_block_feat=self.is_block)
            t_feat, t_logit = self.t_model(input, is_group_feat=self.is_group, is_block_feat=self.is_block)
            s_f, t_f = self._get_dist_feat(self.args.distill, s_feat, t_feat, s_logit, t_logit)
            loss_cls = self.criterion_cls(s_logit, target)
            loss_div = self.criterion_div(s_logit, t_logit)
            loss_kd = sum([self.criterion_kd[i](s_f[i], t_f[i]) * betas[i] for i in range(len(s_f))])
            loss = loss_cls + loss_div * self.args.alpha + loss_kd
        else:
            # Normal training
            s_logit = self.s_model(input)
            loss_cls = self.criterion_cls(s_logit, target)
            loss_div = loss_kd = torch.zeros(1).to(self.device)
            loss = loss_cls
        loss.backward()

        # Set the gradient of the pruned weights to 0 if it's in the "hard prune mode"
        if self.do_hard_prune:
            self._mask_pruned_filters_grad()

        # Get performance metrics
        top1, top5 = accuracy(s_logit, target, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': loss.item(),
                'cls_loss': loss_cls.item(),
                'div_loss': loss_div.item(),
                'kd_loss': loss_kd.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, self.global_step
        )
        return loss, top1, top5

    def _evaluate(self, batch):
        input, target = batch
        logit = self.s_model(input)
        loss = self.criterion_cls(logit, target)
        top1, top5 = accuracy(logit, target, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}

    def _prune_s_model(self, do_prune):
        if not (do_prune and self.cur_epoch % self.args.prune_interval == 0):
            return
        self.s_pruner.prune(self.args.prune_mode, self.args.prune_rates)
        print_nonzeros(self.s_model)

    def _plot_feat(self, method):
        if method == 'hap':
            plotter = HAPPlotter()
        else:
            return
        for i, batch in enumerate(self.eval_loader):
            input, target = [t.to(self.device) for t in batch]
            s_feat, _ = self.s_model(input, is_group_feat=True, is_block_feat=False)
            t_feat, _ = self.t_model(input, is_group_feat=True, is_block_feat=False)
            s_f, t_f = self._get_dist_feat(self.args.distill, s_feat, t_feat, None, None)
            plotter.plot(s_f[0], t_f[0], input, target)
            break

    def train(self):
        """
        It’s a little different from FPGM (CVPR - 2019 oral), they do hard-prune after training an epoch,
        however, the experimental results are very close.
        Github: https://github.com/he-y/filter-pruning-geometric-median/blob/master/pruning_cifar10.py
        """
        self.model.train()  # Train mode
        self.model = self.model.to(self.device)
        best_top1 = 0.
        self.cur_lr = self.args.lr
        self.global_step = 0
        for epoch in range(self.args.n_epochs):
            self.cur_epoch = epoch
            self._prune_s_model(self.do_hard_prune)
            self._train_epoch()
            self._prune_s_model(self.do_soft_prune)
            # self._plot_feat(self.args.distill)
            eval_result = self._eval_epoch()
            if best_top1 < eval_result['top1']:
                best_top1 = eval_result['top1']
                save_model(self.model, self._get_save_model_path(), self.logger)


def main():
    set_seeds(args.seed)
    check_dirs_exist([args.save_dir])
    logger = Logger(args.log_path)
    device = get_device(args.dev_idx)
    if args.dataset not in dataset.__dict__:
        raise NameError
    if args.t_model not in models.__dict__:
        raise NameError
    if args.s_model not in models.__dict__:
        raise NameError
    train_loader, eval_loader, num_classes = dataset.__dict__[args.dataset](args.batch_size)
    t_model = models.__dict__[args.t_model](num_classes=num_classes)
    s_model = models.__dict__[args.s_model](num_classes=num_classes)
    load_model(t_model, args.t_path, logger, device)
    load_model(s_model, args.s_path, logger, device)
    optimizer = optim.SGD(
        s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    base_trainer_cfg = (args, s_model, train_loader, eval_loader, optimizer, args.save_dir, device, logger)
    writer = SummaryWriter(log_dir=args.log_dir)  # For tensorboardX
    trainer = PrunedModelTrainer(t_model, writer, *base_trainer_cfg)
    logger.log('\n'.join(map(str, vars(args).items())))
    if args.evaluate:
        trainer.eval()
    else:
        trainer.train()
        trainer.eval()


if __name__ == '__main__':
    main()
