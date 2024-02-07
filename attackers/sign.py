import torch
from .vflbase import BaseVFL


class Attacker(BaseVFL):
    '''
    LIA using gradient sign approach.
    '''
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        self.total_acc = 0
        self.round = 0
        print('Attacker: {}'.format(args.attack))

    def train(self):
        super().train()

    def test(self):
        super().test()

    def attack(self, grad, labels, batch_idx):
        '''
        Implement gradient sign attack.

        Due to the property of gradient sign, the attack of each passive parties is equal, because they can only use the last layer's gradient.
        '''

        self.round += 1  # inplement attack once
        
        batch_grad = grad[0]

        attack_res = []
        for g in batch_grad:
            if torch.nonzero(g<0).shape[0] > 1:
                attack_res.append(torch.nonzero(g<0)[0].unsqueeze(0))
            elif torch.nonzero(g<0).shape[0] == 1:
                attack_res.append(torch.nonzero(g<0))  # only the ground-truth is negative
            else:
                attack_res.append(torch.randint(0, 10, (1, 1)))
        attack_res = torch.cat(attack_res, dim=0).squeeze()
        
        # the last batch may be smaller than batch_size, but gradient is padded to batch_size, so the attack_res may be larger than labels, even larger than batch_size
        attack_res = attack_res[:labels.shape[0]]
        correct = attack_res.eq(labels).sum().item()
        attack_acc = 100. * correct / labels.shape[0]
        self.total_acc += attack_acc
        print('Attack Accuracy: {}/{} ({:.2f}%)'.format(correct, labels.shape[0], attack_acc))
        
        if batch_idx == self.iteration - 1:
            print('Average Attack Accuracy (each epoch): {:.2f}%'.format(self.total_acc / self.round))
            self.metrics.attack_acc.append(self.total_acc / self.round)
            self.metrics.write()

            self.total_acc = 0
            self.round = 0