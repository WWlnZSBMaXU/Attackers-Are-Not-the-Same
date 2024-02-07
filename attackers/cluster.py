import torch
import numpy as np
from .vflbase import BaseVFL
import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class Attacker(BaseVFL):
    '''
    LIA using cluster approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.total_acc = [0] * self.args.num_passive
        self.round = 0
        print('Attacker: {}'.format(args.attack))

    def train(self):
        super().train()

    def test(self):
        super().test()

    def attack(self, data, labels, batch_idx):
        '''
        Implement cluster attack.
        '''
        
        num_classes = datasets.datasets_classes[self.args.dataset]

        if labels.shape[0] != self.args.batch_size:
            pass
        else:
            self.round += 1  # inplement attack once

            if self.args.use_emb:
                # process emb
                passive_emb = []
                if self.args.tsne:
                    for passive_id in range(len(data)):
                        passive_emb.append(self._tsne(data[passive_id].detach().numpy()))
                else:
                    for passive_id in range(len(data)):
                        passive_emb.append(data[passive_id].reshape(data[passive_id].shape[0], -1).detach().numpy())  # KMeans expected dim <= 2.
                # Using K-means to cluster embeddings for different passive parties.
                acc_list = self._kmeans(num_classes, passive_emb, labels)
            else:
                if len(data) != self.args.num_passive:
                    raise ValueError("The number of gradients is not equal to the number of passive parties.")
                
                # process grad
                passive_grad = []
                if self.args.tsne:
                    for passive_id in range(len(data)):
                        passive_grad.append(self._tsne(data[passive_id].detach().numpy()))
                    passive_grad = torch.Tensor(passive_grad)
                else:
                    for passive_id in range(len(data)):
                        passive_grad.append(data[passive_id].reshape(data[passive_id].shape[0], -1).detach())  # KMeans expected dim <= 2.

                # Using K-means to cluster gradients for different passive parties.
                acc_list = self._kmeans(num_classes, passive_grad, labels)

            # update total accuracy
            for passive_id in range(self.args.num_passive):
                self.total_acc[passive_id] += acc_list[passive_id]

        # calculate average accuracy and write metrics
        if batch_idx == self.iteration - 1:
            avg_acc = []
            for passive_id in range(self.args.num_passive):
                avg_acc.append(self.total_acc[passive_id] / self.round)
                print('Average Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, avg_acc[passive_id]))
            self.metrics.attack_acc.append(avg_acc)
            self.metrics.write()

            self.total_acc = [0] * self.args.num_passive
            self.round = 0


    def _kmeans(self, num_classes, data, labels):
        '''
        K-means clustering.
        '''
        # initialize the attack predicted labels
        cluster_labels = torch.randint(0, 9, labels.shape, dtype=torch.long)

        acc_list = [0] * self.args.num_passive
        for passive_id in range(self.args.num_passive):
            # algorithm{'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
            kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init='auto')
            kmeans.fit(data[passive_id])
            kmeans_labels = kmeans.predict(data[passive_id])

            # calculate the closest point to the center
            dis = kmeans.transform(data[passive_id]).min(axis=1)  # n_samples * n_clusters
            always_correct = 0
            for i in range(num_classes):
                i_idx = np.where(kmeans_labels == i)  # tuple, size=1
                if len(i_idx[0]) == 0:
                    continue
                always_correct += 1
                closest_idx = i_idx[0][dis[i_idx].argmin()]

                # update labels according to the closest point
                cluster_labels[i_idx] = labels[closest_idx]

            # calculate the accuracy
            correct = cluster_labels.eq(labels).sum().item()
            attack_acc = 100. * (correct - always_correct) / (labels.shape[0] - always_correct)
            acc_list[passive_id] = attack_acc

        return acc_list

    
    def _tsne(self, data):
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        tsne.fit_transform(data)

        return tsne.embedding_