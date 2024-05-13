import torch
import numpy as np
import utils.datasets as datasets
from sklearn.cluster import KMeans

def our_defense(args, emb, grad, labels, idx_list):
    num_classes = datasets.datasets_classes[args.dataset]
    if len(emb[0]) < num_classes:
        return grad

    cluster_acc_list = _kmeans(args.num_passive, num_classes, emb, labels)
    min_idx = np.argmin(cluster_acc_list)

    if cluster_acc_list[min_idx] == 0:
        cluster_acc_list[min_idx] = 1 / num_classes

    grad_new = []
    for idx in range(len(grad)):
        if (idx in idx_list) and ((cluster_acc_list[idx] - cluster_acc_list[min_idx] > 5) or (cluster_acc_list[idx] / cluster_acc_list[min_idx] > 2.0)):
            epsilon = cluster_acc_list[min_idx] / cluster_acc_list[idx] * 0.05
            grad_new.append(_dp_defense(grad[idx], epsilon))
        else:
            grad_new.append(grad[idx])
    
    return grad_new


def _kmeans(num_passive, num_classes, emb, labels):
    '''
    K-means clustering.
    '''
    # deal the data
    data = []
    for passive_id in range(len(emb)):
        data.append(emb[passive_id].reshape(emb[passive_id].shape[0], -1).detach())

    # initialize the attack predicted labels
    cluster_labels = torch.randint(0, 9, labels.shape, dtype=torch.long)

    acc_list = [0] * num_passive
    for passive_id in range(num_passive):
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


def _dp_defense(grad, epsilon):
        '''Add DP defense to the model'''
        # get sensitivity
        flatten_grad = grad.flatten()
        grad_norm = abs(flatten_grad.norm(dim=0, p=2))
        # clip
        clip_grad = flatten_grad.clip(-grad_norm, grad_norm).reshape(grad.shape)
        # add noise
        noise_grad = clip_grad + torch.tensor(np.random.laplace(loc=0, scale=grad_norm.detach().numpy() * 1.0 / epsilon, size=clip_grad.shape))

        return noise_grad