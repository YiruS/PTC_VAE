

import os
import sys
import time
import glog as logger
import numpy as np

import torch
from torch.autograd import Variable
from deepgeom.utils import check_exist_or_mkdirs


class Stats(object):
    def __init__(self):
        self.iter_loss = []

    def push_loss(self, iter, loss):
        self.iter_loss.append([iter, loss])

    def push(self, iter, loss):
        self.push_loss(iter, loss)

    def save(self, file):
        np.savez_compressed(
            file,
            iter_loss=np.asarray(self.iter_loss))


class TrainTester(object):

    def __init__(self, net, solver, total_epochs, cuda, log_dir, verbose_per_n_batch=1):
        self.net, self.solver, self.total_epochs, self.cuda = net, solver, total_epochs, cuda
        self.log_dir, self.verbose_per_n_batch = log_dir, verbose_per_n_batch
        check_exist_or_mkdirs(log_dir)

        self.done = False
        self.train_iter = 0
        self.stats_train_batch = Stats()
        self.stats_train_running = Stats()
        self.stats_test = Stats()
        self.running_loss = None
        self.running_factor = 0.9
        self.epoch_callbacks = [self.save_stats]


    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    logger.warn('epoch_callback[{}] failed.'.format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults['lr']
        lr = base_lr * (total_step - step + 1.) / total_step
        for param_group in self.solver.param_groups:
            param_group['lr'] = lr


    def train(self, epoch, loader, loss_fn):
        # import ipdb; ipdb.set_trace()
        self.net.train()
        total_step = self.total_epochs * len(loader)
        finished_step = (epoch-1) * len(loader)
        loss_sum, batch_loss = 0.0, 0.0
        for batch_idx, batch in enumerate(loader):
            # self.adjust_lr_linear(batch_idx + finished_step, total_step)
            data = batch['data']
            if self.cuda:
                data = data.cuda()
            data = Variable(data)

            # training step
            self.solver.zero_grad()
            output = self.net(data)
            loss = loss_fn(output, data)
            loss.backward()
            self.solver.step()

            # loss
            batch_len = len(data)
            batch_loss = loss.item()
            loss_sum += batch_loss
            if self.running_loss is None:
                self.running_loss = batch_loss
            else:
                self.running_loss = self.running_factor*self.running_loss \
                                    + (1-self.running_factor)*batch_loss

            # collect stats
            self.train_iter += 1
            self.stats_train_batch.push(self.train_iter, loss=batch_loss)
            self.stats_train_running.push(self.train_iter, loss=self.running_loss)

            # logger
            if self.verbose_per_n_batch>0 and batch_idx % self.verbose_per_n_batch==0:
                logger.info((
                    'Epoch={:<3d} [{:3.0f}% of {:<5d}] '+
                    'Loss(Batch,Running)={:.3f},{:.3f} ').format(
                    epoch, 100.*batch_idx/len(loader), len(loader.dataset),
                    batch_loss, self.running_loss))


        logger.info('Train set (epoch={:<3d}): Loss(LastBatch,Average)={:.3f},{:.3f}'.format(
            epoch, batch_loss, loss_sum / float(len(loader))))


    def test(self, epoch, loader, loss_fn):
        # import ipdb; ipdb.set_trace()
        self.net.eval()
        loss_fn.size_average = False
        test_loss = 0.
        counter = 0
        for batch in loader:
            data = batch['data']
            if self.cuda:
                data = data.cuda()
            data = Variable(data) # BxNxK

            output = Variable(self.net(data)) # BxNxK
            test_loss += loss_fn(output, data)
            counter += 1

        # import ipdb; ipdb.set_trace()
        test_loss = test_loss.cpu().item() / counter
        #import ipdb; ipdb.set_trace()
        self.stats_test.push(self.train_iter, loss=test_loss)
        logger.info('Test set  (epoch={:<3d}): AverageLoss={:.4f}'.format(epoch, test_loss))
        # import ipdb; ipdb.set_trace()
        loss_fn.size_average = True


    def save_stats(self):
        self.stats_train_running.save(os.path.join(self.log_dir, 'stats_train_running.npz'))
        self.stats_train_batch.save(os.path.join(self.log_dir, 'stats_train_batch.npz'))
        self.stats_test.save(os.path.join(self.log_dir, 'stats_test.npz'))


    def run(self, train_loader, test_loader, loss_fn):
        try:
            from deepgeom.visualize import make_dot
            y = self.net.forward(Variable(torch.from_numpy(test_loader.dataset[0:2]['data'])))
            g = make_dot(y)
            g.engine='dot'
            g.format='pdf'
            print(g.render(filename=os.path.join(self.log_dir, 'net.gv')))
        except:
            logger.warn('failed to draw net.')


        logger.check_eq(self.done, False, 'Done already!')
        if self.cuda:
            self.net.cuda()

        logger.info('Network Architecture:')
        print(str(self.net))
        sys.stdout.flush()

        logger.info('{} Hyperparameters:'.format(self.solver.__class__.__name__))
        print(str(self.solver.defaults))
        sys.stdout.flush()

        logger.info('Initial test with random initialized parameters:')
        self.test(epoch=0, loader=test_loader, loss_fn=loss_fn)
        for epoch in range(1, self.total_epochs+1):
            self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
            self.test(epoch=epoch, loader=test_loader, loss_fn=loss_fn)
            self.invoke_epoch_callback()
        self.save_stats()
        self.done=True
