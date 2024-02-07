import torch
import torch.nn as nn
from .vflbase import BaseVFL
import utils.datasets as datasets
import utils.models as models
import utils.losses as losses  # ExPLoit
import numpy as np  # ExPLoit
import os
import torch.nn.functional as F


class Flatten(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    

class Reconstruction(nn.Module):
    '''Reconstruction model.'''
    def __init__(self, input_size, num_classes):
        super(Reconstruction, self).__init__()
        self.reconstruction = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(input_size, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax(dim=1)
        )
        nn.init.xavier_uniform_(self.reconstruction[2].weight)
        print("Reconstruction Model", self.reconstruction)

    def forward(self, x):
        pred = self.reconstruction(x)
        return pred


class Attacker(BaseVFL):
    '''
    LIA using gradient reconstruction approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.reconstruction_model = []
        self.reconstruction_optimizer = []
        self.label_guess = []
        self.label_optimizer = []
        self.loss_pred = None
        self.loss_grad = None
        self.correct_list = [0] * args.num_passive
        print('Attacker: {}'.format(args.attack))

    def train(self):
        super().train()

    def test(self):
        super().test()

    def attack(self, init=False):
        '''
        Implement model reconstruction attack.
        '''
        # initialize reconstruction model
        if init:
            print('Initialize reconstruction model.')
            file_batch_0 = os.listdir(self.data_dir)[0]
            data = torch.load(os.path.join(self.data_dir, file_batch_0))
            tmp_emb, _, _ = data
            input_size_list = []
            for passive_id in range(self.args.num_passive):
                input_size_list.append(tmp_emb[passive_id].reshape(tmp_emb[passive_id].shape[0], -1).shape[1])
            num_classes = datasets.datasets_classes[self.args.dataset]
            for i in range(self.args.num_passive):
                self.reconstruction_model.append(Reconstruction(input_size_list[i], num_classes))
                self.reconstruction_optimizer.append(torch.optim.Adam(self.reconstruction_model[i].parameters(), lr=self.args.lr_attack_model))
            # self.loss_pred = torch.nn.CrossEntropyLoss()
            self.loss_pred = torch.nn.MSELoss()
            self.loss_grad = torch.nn.MSELoss()

        self._train_all()

    
    def _train_all(self):
        num_classes = datasets.datasets_classes[self.args.dataset]
        # train reconstruction model
        for passive_id in range(self.args.num_passive):
            self.reconstruction_model[passive_id].apply(models.weights_init)
            self.reconstruction_model[passive_id].train()
            scheduler = torch.optim.lr_scheduler.StepLR(self.reconstruction_optimizer[passive_id], step_size=10, gamma=0.1)
            for epoch in range(self.args.attack_model_epochs):
                # for batch_data in data:
                for batch_idx, batch_file in enumerate(os.listdir(self.data_dir)):
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file))
                    emb, grad, labels = batch_data
                    emb[passive_id].requires_grad = True
                    
                    if epoch == 0:
                        label_guess = torch.rand((labels.shape[0], num_classes), dtype=torch.float)
                        label_guess = F.softmax(label_guess, dim=1)
                    else:
                        label_guess = torch.load(os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))
                    label_guess.requires_grad = True
                    label_optimizer = torch.optim.Adam([label_guess], lr=self.args.lr_attack_model)

                    pred = self.reconstruction_model[passive_id](emb[passive_id])

                    # calculate loss
                    loss_pred = self.loss_pred(pred, label_guess)
                    weight_hat = torch.autograd.grad(loss_pred, emb[passive_id], create_graph=True)[0]

                    loss = self.loss_grad(weight_hat, grad[passive_id]) * 1e7

                    self.reconstruction_optimizer[passive_id].zero_grad()
                    label_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.reconstruction_optimizer[passive_id].step()
                    label_optimizer.step()

                    torch.save(label_guess, os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))
                scheduler.step()
                    
        # evaluate reconstruction model
        tot_acc = []
        for passive_id in range(self.args.num_passive):
            self.reconstruction_model[passive_id].eval()

            correct = 0
            with torch.no_grad():
                for batch_idx, batch_file in enumerate(os.listdir(self.data_dir)):
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file))
                    emb, _, labels = batch_data
                    # correct += self.reconstruction_model[passive_id](emb[passive_id]).argmax(dim=1).eq(labels).sum().item()
                    label_guess = torch.load(os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))
                    correct += label_guess.argmax(dim=1).eq(labels).sum().item()
            acc = 100. * correct / self.train_dataset_len
            tot_acc.append(acc)
            print('Average Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, acc))       
            
        self.metrics.attack_acc.append(tot_acc)
        self.metrics.write()


    def _exploit(self):
        num_classes = datasets.datasets_classes[self.args.dataset]

        ### ExPLoit ###
        label_prior = torch.ones([1, num_classes]) / num_classes
        p = 0.8918
        H_y = -p * np.log(p) - (1 - p) * np.log(1 - p)
        ###############
        
        tot_acc = [0] * self.args.num_passive
        for passive_id in range(self.args.num_passive):
            self.reconstruction_model[passive_id].train()
            for epoch in range(self.args.attack_model_epochs):
                for batch_idx, batch_file in enumerate(os.listdir(self.data_dir)):
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file))
                    batch_emb, batch_grad, batch_labels = batch_data
                    batch_emb = batch_emb[passive_id]
                    batch_emb.requires_grad = True
                    batch_grad = batch_grad[passive_id]

                    if epoch == 0:
                        label_guess = torch.zeros((batch_labels.shape[0], num_classes), dtype=torch.float)
                        label_guess = F.softmax(label_guess, dim=1)
                    else:
                        label_guess = torch.load(os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))
                    label_guess.requires_grad = True
                    label_optimizer = torch.optim.Adam([label_guess], lr=self.args.lr_attack_model)

                    pred = self.reconstruction_model[passive_id](batch_emb)
                    # H_label = losses.EntropyLoss(logits=False)(label_guess)

                    ce_loss = self.loss_pred(pred, label_guess)
                    weight_hat = torch.autograd.grad(ce_loss, batch_emb, create_graph=True)[0]
                    grad_loss = self.loss_grad(weight_hat, batch_grad) * 1e7
                    # ce_loss = losses.SoftCrossEntropyLoss(reduction='mean')(pred, label_guess)
                    # grad_approx = torch.autograd.grad(ce_loss, batch_emb, create_graph=True)[0]
                    # grad_norm_mean = batch_grad.norm(dim=-1).mean().item()
                    # grad_loss = ((batch_grad - grad_approx * self.args.batch_size).norm(dim=-1)).mean() / grad_norm_mean

                    acc_loss = ce_loss / H_y

                    kl_loss = losses.KlDivLoss(reduction='mean')(label_prior, label_guess.mean(dim=0, keepdim=True))

                    loss = grad_loss + acc_loss + kl_loss


                    # loss_pred = self.loss_pred(pred, label_guess)
                    # weight_hat = torch.autograd.grad(loss_pred, batch_emb, create_graph=True)[0]

                    # ### ExPLoit ###
                    # label_loss = losses.SoftCrossEntropyLoss(reduction='mean')(pred, label_guess)
                    # grad_norm_mean = batch_grad.norm(dim=-1).mean().item()
                    # grad_loss = (batch_grad - weight_hat).norm(dim=-1).mean() / grad_norm_mean
                    # kl_loss = losses.KlDivLoss(reduction='mean')(label_prior, pred.mean(dim=0))
                    # loss = grad_loss + label_loss / H_y + kl_loss
                    ###############

                    self.reconstruction_optimizer[passive_id].zero_grad()
                    label_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.reconstruction_optimizer[passive_id].step()
                    label_optimizer.step()

                    torch.save(label_guess, os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))

        # evaluate reconstruction model
        tot_acc = []
        for passive_id in range(self.args.num_passive):
            self.reconstruction_model[passive_id].eval()

            correct = 0
            with torch.no_grad():
                for batch_idx, batch_file in enumerate(os.listdir(self.data_dir)):
                    batch_data = torch.load(os.path.join(self.data_dir, batch_file))
                    emb, _, labels = batch_data
                    # correct += self.reconstruction_model[passive_id](emb[passive_id]).argmax(dim=1).eq(labels).sum().item()
                    label_guess = torch.load(os.path.join(self.label_guess_dir, "label_guess_{}_{}.pt".format(passive_id, batch_idx)))
                    correct += label_guess.argmax(dim=1).eq(labels).sum().item()
            acc = 100. * correct / self.train_dataset_len
            tot_acc.append(acc)
            print('Average Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, acc))       
            
        self.metrics.attack_acc.append(tot_acc)
        self.metrics.write()