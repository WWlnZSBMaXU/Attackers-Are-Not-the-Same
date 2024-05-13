import torch
from utils.metrics import Metrics
import time
import os
import numpy as np
import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import attackers.our as our


class BaseVFL(object):
    def __init__(self, args, entire_model, train_dataset, test_dataset):
        # setup arguments
        self.args = args

        # get data file path
        self.data_dir = os.path.join('./data', self.args.attack, self.args.dataset,
            'data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.num_passive,
                self.args.batch_size,
                self.args.attack_every_n_iter,
                self.args.epochs,
                self.args.attack_epoch, 
                self.args.attack_id,
                self.args.simple,
                self.args.padding_mode,
                self.args.as_order,
                self.args.balanced,
                self.args.division_mode))
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # get label guess file path
        self.label_guess_dir = os.path.join('./label_guess', self.args.attack, self.args.dataset,
            'data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.num_passive,
                self.args.batch_size,
                self.args.attack_every_n_iter,
                self.args.epochs,
                self.args.attack_epoch, 
                self.args.attack_id,
                self.args.simple,
                self.args.padding_mode,
                self.args.as_order,
                self.args.balanced,
                self.args.division_mode))
        if not os.path.exists(self.label_guess_dir):
            os.makedirs(self.label_guess_dir)
        
        # process dataset
        self._process_data(train_dataset, test_dataset)

        # setup entire model and optimizer
        self.model = entire_model
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer_entire = torch.optim.SGD(self.model.parameters(), lr=args.lr_active)
        self.optimizer_active = torch.optim.SGD(self.model.active.parameters(), lr=args.lr_active)
        self.optimizer_passive = []
        for i in range(args.num_passive):
            lr = args.lr_attack if i == args.attack_id else args.lr_passive
            self.optimizer_passive.append(torch.optim.SGD(self.model.passive[i].parameters(), lr=lr))

        # setup metrics
        self.metrics = Metrics(args)

        # record iteration
        self.iteration = None


    def _process_data(self, train_dataset, test_dataset):        
        if self.args.padding_mode:
            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data
                train_data_padded = []
                for i in range(self.args.num_passive):
                    if i == self.args.attack_id:
                        train_data_padded.append(data)
                    else:
                        train_data_padded.append(torch.rand(data.shape))  # generate padding data
                self.train_dataset.append([train_data_padded, labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                test_data_padded = []
                for i in range(self.args.num_passive):
                    if i == self.args.attack_id:
                        test_data_padded.append(data)
                    else:
                        test_data_padded.append(torch.rand(data.shape))
                self.test_dataset.append([test_data_padded, labels])
                self.test_dataset_len += len(data)
        elif self.args.division_mode == 'vertical':
            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data  # len(data) = batch_size 128
                # for mnist and fashionmnist: torch.Size([128, 1, 28, 28])
                # for cifar10: torch.Size([128, 3, 32, 32])
                if self.args.dataset == "criteo":
                    self.train_dataset.append([torch.chunk(data, self.args.num_passive, dim=1), labels])
                else:
                    self.train_dataset.append([torch.chunk(data, self.args.num_passive, dim=3), labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                if self.args.dataset == "criteo":
                    self.test_dataset.append([torch.chunk(data, self.args.num_passive, dim=1), labels])
                else:
                    self.test_dataset.append([torch.chunk(data, self.args.num_passive, dim=3), labels])
                self.test_dataset_len += len(data)
        elif self.args.division_mode == 'random':
            if self.args.dataset not in ['mnist', 'cifar10']:
                raise ValueError("Random division only supports MNIST and CIFAR-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset == "mnist" else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset == "mnist":
                    list_0 = [1, 4, 5, 7, 9, 11, 13, 14, 16, 18, 19, 23, 24, 26]
                    list_1 = [0, 2, 3, 6, 8, 10, 12, 15, 17, 20, 21, 22, 25, 27]
                elif self.args.dataset == "cifar10":
                    list_0 = [0, 3, 4, 5, 6, 7, 9, 14, 15, 16, 22, 23, 28, 29, 30, 31]
                    list_1 = [1, 2, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26, 27]
                sample_list.append(list_0)
                sample_list.append(list_1)
            elif self.args.num_passive == 4:
                if self.args.dataset == "mnist":
                    list_0 = [1, 2, 3, 17, 20, 23, 27]
                    list_1 = [6, 8, 11, 12, 13, 14, 24]
                    list_2 = [5, 7, 9, 10, 22, 25, 26]
                    list_3 = [0, 4, 15, 16, 18, 19, 21]
                elif self.args.dataset == "cifar10":
                    list_0 = [0, 4, 6, 14, 20, 24, 25, 28]
                    list_1 = [1, 3, 9, 15, 17, 19, 22, 27]
                    list_2 = [5, 7, 11, 16, 18, 21, 23, 31]
                    list_3 = [2, 8, 10, 12, 13, 26, 29, 30]
                sample_list.append(list_0)
                sample_list.append(list_1)
                sample_list.append(list_2)
                sample_list.append(list_3)

            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.train_dataset.append([data_list, labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.test_dataset.append([data_list, labels])
                self.test_dataset_len += len(data)
        elif self.args.division_mode == 'imbalanced':
            if self.args.dataset not in ['mnist', 'cifar10']:
                raise ValueError("Imbalance division only supports MNIST and CIFAR-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset == "mnist" else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset == "mnist":
                    # 20 & 8
                    list_0 = [0, 2, 3, 4, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 27]
                    list_1 = [1, 5, 6, 8, 13, 17, 20, 24]
                elif self.args.dataset == "cifar10":
                    # 20 & 12
                    list_0 = [0, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 18, 21, 22, 24, 25, 26, 28, 30, 31]
                    list_1 = [1, 2, 5, 6, 11, 12, 17, 19, 20, 23, 27, 29]
                sample_list.append(list_0)
                sample_list.append(list_1)
            elif self.args.num_passive == 4:
                if self.args.dataset == "mnist":
                    # 12 & 6 & 3 & 7
                    list_0 = [1, 3, 4, 5, 7, 11, 14, 15, 19, 21, 23, 27]
                    list_1 = [2, 6, 9, 10, 12, 22]
                    list_2 = [0, 13, 17]
                    list_3 = [8, 16, 18, 20, 24, 25, 26]
                elif self.args.dataset == "cifar10":
                    # 13 & 7 & 4 & 8
                    list_0 = [0, 3, 6, 7, 12, 14, 15, 16, 23, 27, 29, 30, 31]
                    list_1 = [1, 2, 10, 13, 19, 22, 24]
                    list_2 = [8, 9, 11, 26]
                    list_3 = [4, 5, 17, 18, 20, 21, 25, 28]
                sample_list.append(list_0)
                sample_list.append(list_1)
                sample_list.append(list_2)
                sample_list.append(list_3)

            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.train_dataset.append([data_list, labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.test_dataset.append([data_list, labels])
                self.test_dataset_len += len(data)

        print("Finish processing dataset.")

    
    def train(self):
        self.iteration = len(self.train_dataset)
        
        for epoch in range(self.args.epochs):
            # train entire model
            self.model.train()
            self.model.active.train()
            for i in range(self.args.num_passive):
                self.model.passive[i].train()

            dispersion_list = []

            critial_idx_list = []
            idx_count_list = [0] * self.args.num_passive

            # start train and attack
            for batch_idx, batch_data in enumerate(self.train_dataset):
                data, labels = batch_data

                emb, logit, pred = self.model(data)
                loss = self.loss(pred, labels)

                # zero grad for all optimizers
                self.optimizer_entire.zero_grad()
                self.optimizer_active.zero_grad()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].zero_grad()

                loss.backward(retain_graph=True)

                # our method
                grad = torch.autograd.grad(loss, emb, create_graph=True)
                
                if self.args.our:
                    if epoch == 0 and batch_idx <= 199:
                        cluster_acc_list = our._kmeans(self.args.num_passive, datasets.datasets_classes[self.args.dataset], emb, labels)
                        min_idx = np.argmin(cluster_acc_list)
                        if cluster_acc_list[min_idx] == 0:
                            cluster_acc_list[min_idx] = 1 / datasets.datasets_classes[self.args.dataset]
                        for i in range(self.args.num_passive):
                            if (cluster_acc_list[i] - cluster_acc_list[min_idx] > 5) or (cluster_acc_list[i] / cluster_acc_list[min_idx] > 2.0):
                                idx_count_list[i] += 1
                        if batch_idx == 199:
                            for i in range(self.args.num_passive):
                                if idx_count_list[i] >= 100:
                                    critial_idx_list.append(i)
                            print("critial_idx_list: {}".format(critial_idx_list))
                    else:
                        grad = our.our_defense(self.args, emb, grad, labels, critial_idx_list)
                        for i in critial_idx_list:
                            emb[i].backward(grad[i])


                # attack
                if self.args.attack in ['reconstruction', 'completion']:
                    # record embeddings, gradients, and labels
                    # process emb
                    passive_emb = []
                    for passive_id in range(self.args.num_passive):
                        passive_emb.append(emb[passive_id].clone().detach())

                    # process grad
                    # grad = torch.autograd.grad(loss, emb, create_graph=True)
                    if self.args.defense:
                        if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                            grad = self._dp_defense(grad)
                            emb[self.args.attack_id].backward(grad[self.args.attack_id])
                    if self.args.defense_all:
                        if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                            grad = self._dp_defense_all(grad)
                            for i in range(self.args.num_passive):
                                emb[i].backward(grad[i])
                    if self.args.dispersion:
                        if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                            dispersion_list.append(self.dispersion(emb, grad, labels))
                    passive_grad = []
                    for passive_id in range(self.args.num_passive):
                        passive_grad.append(grad[passive_id].clone().detach())
                    
                    # save the training data record to file to solve the lack of memory
                    torch.save([passive_emb, passive_grad, labels], os.path.join(self.data_dir, "data_{}.pt".format(batch_idx)))
                    del passive_emb, passive_grad, grad

                if (self.args.set_attack_epoch and epoch == self.args.attack_epoch) or not self.args.set_attack_epoch:
                    if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                        start_time = time.time_ns()

                        if self.args.attack == 'sign':
                            grad_logit = torch.autograd.grad(loss, logit, create_graph=True)
                            if self.args.defense:
                                grad = self._dp_defense(grad)
                                emb[self.args.attack_id].backward(grad[self.args.attack_id])
                            if self.args.defense_all:
                                grad = self._dp_defense_all(grad)
                                for i in range(self.args.num_passive):
                                    emb[i].backward(grad[i])
                            if self.args.dispersion:
                                self.dispersion(emb, grad, labels)
                            self.attack(grad_logit, labels, batch_idx)
                            del grad_logit, grad
                        elif self.args.attack == 'cluster':
                            # grad = torch.autograd.grad(loss, emb, create_graph=True)

                            if self.args.defense:
                                grad = self._dp_defense(grad)
                                emb[self.args.attack_id].backward(grad[self.args.attack_id])
                            if self.args.defense_all:
                                grad = self._dp_defense_all(grad)
                                for i in range(self.args.num_passive):
                                    emb[i].backward(grad[i])
                            if self.args.dispersion:
                                dispersion_list.append(self.dispersion(emb, grad, labels))

                            if self.args.use_emb:
                                self.attack(emb, labels, batch_idx)
                            else:                                
                                self.attack(grad, labels, batch_idx)

                            del grad

                        end_time = time.time_ns()
                        attack_nseconds = end_time - start_time
                        second = int(attack_nseconds / (1000 * 1000))
                        msecond = int(attack_nseconds / 1000) % 1000
                        nsecond = int(attack_nseconds) % 1000
                        print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                        self.metrics.attack_runtime.append(attack_nseconds)

                # update parameters for all optimizers
                self.optimizer_entire.step()
                self.optimizer_active.step()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].step()

                if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                    print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))

            # reconstruction attack after each epoch
            if self.args.attack in ['reconstruction', 'completion']:
                start_time = time.time_ns()

                if not self.args.set_attack_epoch:
                    if epoch == 0:
                        self.attack(init=True)
                    else:
                        self.attack()
                else:
                    if epoch + 1 < self.args.attack_epoch:
                        pass
                    elif epoch + 1 == self.args.attack_epoch:
                        self.attack(init=True)
                    else:
                        self.attack()

                end_time = time.time_ns()
                attack_nseconds = end_time - start_time
                second = int(attack_nseconds / (1000 * 1000))
                msecond = int(attack_nseconds / 1000) % 1000
                nsecond = int(attack_nseconds) % 1000
                print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                self.metrics.attack_runtime.append(attack_nseconds)

            # deal the dispersion list
            if self.args.attack in ['cluster', 'completion'] and self.args.dispersion:
                dispersion_list = np.array(dispersion_list)
                dispersion = dispersion_list.mean(axis=0)
                dispersion = dispersion.tolist()
                print("Dispersion: {}".format(dispersion))
                self.metrics.dispersion.append(dispersion)
                self.metrics.write()

            # evaluate the model each epoch
            self._evaluate()


    def test(self):
        print("\n============== Test ==============")
        self.iteration = len(self.test_dataset)

        # test entire model and show test loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_dataset):
                data, labels = batch_data
                emb, _, pred = self.model(data)

                # attack: test phase can only use "emb cluster", because "sign" and "reconstruction" need grad
                if (batch_idx + 1) % int(self.args.attack_every_n_iter / 5) == 0 or (batch_idx + 1) == self.iteration:
                    start_time = time.time_ns()

                    if self.args.attack == 'cluster' and self.args.use_emb:
                        self.attack(emb, labels)

                    end_time = time.time_ns()
                    attack_nseconds = end_time - start_time
                    second = int(attack_nseconds / (1000 * 1000))
                    msecond = int(attack_nseconds / 1000) % 1000
                    nsecond = int(attack_nseconds) % 1000
                    print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                    self.metrics.attack_runtime.append(attack_nseconds)

                test_loss += self.loss(pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(self.test_dataset)
        test_acc = 100. * correct / self.test_dataset_len
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, self.test_dataset_len, test_acc))        
        
        self.metrics.test_loss.append(test_loss)
        self.metrics.test_acc.append(test_acc)
        self.metrics.write()

        return test_acc
    

    def _evaluate(self):
        # evaluate entire model and show training loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        train_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_data in self.train_dataset:
                data, labels = batch_data  # data is tuple, len(data[0]) = batch_size
                _, _, pred = self.model(data)
                train_loss += self.loss(pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        train_loss /= len(self.train_dataset)
        train_acc = 100. * correct / self.train_dataset_len
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, self.train_dataset_len, train_acc))        
        
        self.metrics.train_loss.append(train_loss)
        self.metrics.train_acc.append(train_acc)
        self.metrics.write()

        return train_acc
    

    def attack(self, data, labels):
        pass

    
    def dispersion(self, emb, grad, labels):
        num_classes = datasets.datasets_classes[self.args.dataset]
        passive_emb = []
        for passive_id in range(len(emb)):
            passive_emb.append(emb[passive_id].reshape(emb[passive_id].shape[0], -1).detach().numpy())  # KMeans expected dim <= 2.
        passive_emb = np.array(passive_emb)

        dispersion_list = []
        for passive_id in range(self.args.num_passive):
            # algorithm{'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
            kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init='auto')
            kmeans.fit(passive_emb[passive_id])

            # calculate the closest point to the center
            dis = kmeans.transform(passive_emb[passive_id]).min(axis=1)  # n_samples * n_clusters
            dispersion = dis.sum()
            dispersion_list.append(dispersion)
            print("Passive {} dispersion: {}".format(passive_id, dispersion))
        return dispersion_list
    

    def _dp_defense(self, grad):
        '''add DP defense to the model'''
        # get sensitivity
        flatten_grad = grad[self.args.attack_id].flatten()
        grad_norm = abs(flatten_grad.norm(dim=0, p=2))
        # clip
        clip_grad = flatten_grad.clip(-grad_norm, grad_norm).reshape(grad[self.args.attack_id].shape)
        # add noise
        noise_grad = clip_grad + torch.tensor(np.random.laplace(loc=0, scale=grad_norm.detach().numpy() * 1.0 / self.args.epsilon, size=clip_grad.shape))

        noise_grad_list = []
        for i in range(self.args.num_passive):
            if i == self.args.attack_id:
                noise_grad_list.append(noise_grad)
            else:
                noise_grad_list.append(grad[i])
        noise_grad_list = tuple(noise_grad_list)

        return noise_grad_list
    
    
    def _dp_defense_all(self, grad):
        grad = torch.stack(list(grad), dim=0)
        flatten_grad = grad.flatten()
        grad_norm = abs(flatten_grad.norm(dim=0, p=2))
        clip_grad = flatten_grad.clip(-grad_norm, grad_norm).reshape(grad.shape)
        noise_grad = clip_grad + torch.tensor(np.random.laplace(loc=0, scale=grad_norm.detach().numpy() * 1.0 / self.args.epsilon, size=clip_grad.shape))
        return noise_grad