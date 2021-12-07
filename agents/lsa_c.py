import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import torch.nn.functional as F
import numpy as np
import math
import copy

class LSA_C(ContinualLearner):
    def __init__(self, model, opt, params):
        super(LSA_C, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.data = params.data   
    
    # Calculate the loss of each class separately
    def Every_classify(self, logits, labels):
        loss_dict = dict()
        size = dict()
        for logit, label in zip(logits, labels):
            if label.item() not in loss_dict:
                loss_dict[label.item()] = 0.0
                size[label.item()] = 0
            logit = logit.view(-1, logit.size(-1))
            logit_logsoft = F.log_softmax(logit, dim=1)
            logit_logsoft = logit_logsoft.gather(1, label.view(-1, 1))
            loss = -torch.mul(1.0, logit_logsoft.view(-1).T)
            loss_dict[label.item()] += loss
            size[label.item()] += 1

        for key, value in loss_dict.items():
            loss_dict[key] = value / size[key]
        return loss_dict

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)
        
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                logits = self.model.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                if self.params.trick['kd_trick']:
                    loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                    self.kd_manager.get_kd_loss(logits, batch_x)
                if self.params.trick['kd_trick_star']:
                   loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                   (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                # update tracker
                acc_batch.update(correct_cnt, batch_y.size(0))
                losses_batch.update(loss, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                for j in range(self.mem_iters):
                    new_label = set(batch_y.cpu().numpy())
                    grad_list = [0.0] * 100
                    grad_list[batch_y[0].item()] += torch.norm(self.model.linear.weight.grad).item()   
                    
                    # mem update
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        loss_dict = self.Every_classify(logits=mem_logits, labels=mem_y)
                        for key, loss_memory in loss_dict.items():
                            loss_memory.backward(retain_graph=True)
                            grad_list[key] += torch.norm(self.model.linear.weight.grad).item()
                        self.opt.zero_grad()
                        all_loss = 0.0
                        for key, loss_memory in loss_dict.items():
                            if self.data == 'mnist' or self.data == 'cifar10':
                                all_loss += (loss_memory * grad_list[batch_y[0].item()] /grad_list[key])
                            elif  self.data == 'cifar100':
                                all_loss += (loss_memory * 1.01 * grad_list[batch_y[0].item()] /grad_list[key])
                            elif  self.data == 'mini_imagenet':
                                all_loss += (loss_memory * 1.0 /grad_list[key])
                        if self.params.trick['kd_trick']:
                            all_loss = 1 / (self.task_seen + 1) * all_loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        if self.params.trick['kd_trick_star']:
                            all_loss = 1 / ((self.task_seen + 1) ** 0.5) * all_loss + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        all_loss.backward()
                        
                        # update tracker
                        losses_mem.update(all_loss, mem_y.size(0))
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))
                        self.opt.step()
                    else:
                        pass
                    
                # update mem
                self.buffer.update(batch_x, batch_y)
                torch.cuda.empty_cache()

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
            self.after_train()




