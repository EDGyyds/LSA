import numpy as np
import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import torch.nn.functional as F

class Task_free_continual_learning(ContinualLearner):
    def __init__(self, model, opt, params,seed=123):
        super(Task_free_continual_learning, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.gradient_steps = self.epoch
        self.loss_window_length = 5
        self.loss_window_mean_threshold = 0.2
        self.loss_window_variance_threshold = 0.1
        self.MAS_weight = 0.5
        torch.manual_seed(seed)

    def train_learner(self, x_train, y_train, count_updates=0):
        self.before_train(x_train, y_train)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)
        self.model = self.model.train()
        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
        loss_window = []
        loss_window_means = []
        loss_window_variances = []
        update_tags = []
        new_peak_detected = True
        # MAS regularization: list of 3 weights vectors as there are 3 layers.
        star_variables = []
        omegas = []  # initialize with 0 importance weights
        last_loss_window_mean = 0.0
        last_loss_window_variance = 0.0
        # run few training iterations on the received batch.
        for gs in range(self.gradient_steps):
            # evaluate the new batch
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x)
                    total_loss = self.criterion(logits, batch_y)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    # replay set
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))
                        total_loss += self.criterion(mem_logits, mem_y)
                    if gs==0: first_train_loss=total_loss.cpu().detach().numpy()
                    # add MAS regularization to the training objective
                    if len(star_variables)!=0 and len(omegas)!=0:
                        for pindex, p in enumerate(self.model.parameters()):
                            
                            total_loss += self.MAS_weight / 2. * maybe_cuda(torch.sum(torch.from_numpy(omegas[pindex]).type(torch.float32)) * (p - star_variables[pindex]) ** 2)
                    losses.update(total_loss, mem_y.size(0))
                    # train self.model
                    self.opt.zero_grad()
                    torch.sum(total_loss).backward()
                    self.opt.step()

                    # add loss to loss_window and detect loss plateaus
                    loss_window.append(np.mean(first_train_loss))
                    if len(loss_window) > self.loss_window_length: del loss_window[0]
                    loss_window_mean = np.mean(loss_window)
                    loss_window_variance = np.var(loss_window)
                    # check the statistics of the current window

                    if not new_peak_detected and loss_window_mean > last_loss_window_mean + np.sqrt(last_loss_window_variance):
                        new_peak_detected = True
                    if loss_window_mean < self.loss_window_mean_threshold and loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        count_updates+=1
                        update_tags.append(0.01)
                        last_loss_window_mean=loss_window_mean
                        last_loss_window_variance=loss_window_variance
                        new_peak_detected=False

                        # calculate importance weights and update star_variables
                        gradients = [0 for p in self.model.parameters()]
                        for x, y in zip(mem_x, mem_y):
                            self.model.zero_grad()
                            y = maybe_cuda(y, self.cuda)
                            logits = self.model.forward(x)
                            torch.norm(logits, 2, dim=1).backward()
                            for pindex, p in enumerate(self.model.parameters()):
                                g = p.grad.data.clone().cpu().detach().numpy()
                                gradients[pindex] += np.abs(g)

                        # update the running average of the importance weights
                        omegas_old = omegas[:]
                        omegas = []
                        star_variables = []
                        for pindex, p in enumerate(self.model.parameters()):
                            if len(omegas_old) != 0:
                                omegas.append(1 / count_updates * gradients[pindex] + (1 - 1 / count_updates) * omegas_old[pindex])
                            else:
                                omegas.append(gradients[pindex])
                            star_variables.append(p.data.clone().detach())

                    else:
                        update_tags.append(0)
                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)
                    self.buffer.update(batch_x, batch_y)

                    if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                            'running train acc: {:.3f},' 
                            'running mem acc: {:.3f}'
                                .format(i, losses.avg(), acc_batch.avg(), acc_mem.avg())
                        )
                self.after_train()




