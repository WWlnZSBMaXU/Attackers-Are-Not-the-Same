import os
import json


class Metrics(object):
    def __init__(self, args):
        self.args = args        
        self.test_acc = []
        self.test_loss = []
        self.train_acc = []
        self.train_loss = []
        self.attack_acc = []
        self.dir = './log'
        self.attack_runtime = []
        self.dispersion = []

    def write(self):
        '''write existing history records into a json file'''
        metrics = {}
        metrics['attack'] = self.args.attack
        metrics['dataset'] = self.args.dataset
        metrics['epochs'] = self.args.epochs
        metrics['batch_size'] = self.args.batch_size
        metrics['lr_passive'] = self.args.lr_passive
        metrics['lr_active'] = self.args.lr_active
        metrics['lr_attack'] = self.args.lr_attack
        metrics['set_attack_epoch'] = self.args.set_attack_epoch
        metrics['attack_epoch'] = self.args.attack_epoch
        metrics['attack_id'] = self.args.attack_id
        metrics['num_passive'] = self.args.num_passive
        metrics['use_emb'] = self.args.use_emb
        metrics['attack_every_n_iter'] = self.args.attack_every_n_iter
        metrics['simple'] = self.args.simple
        metrics['padding_mode'] = self.args.padding_mode
        metrics['division_mode'] = self.args.division_mode
        metrics['tsne'] = self.args.tsne
        metrics['as_order'] = self.args.as_order
        metrics['lr_attack_model'] = self.args.lr_attack_model
        metrics['attack_model_epochs'] = self.args.attack_model_epochs
        metrics['balanced'] = self.args.balanced
        metrics['defense'] = self.args.defense
        metrics['defense_all'] = self.args.defense_all
        metrics['epsilon'] = self.args.epsilon
        metrics['round'] = self.args.round
        metrics['test_acc'] = self.test_acc
        metrics['test_loss'] = self.test_loss
        metrics['train_acc'] = self.train_acc
        metrics['train_loss'] = self.train_loss
        metrics['attack_acc'] = self.attack_acc
        metrics['attack_runtime'] = self.attack_runtime
        metrics['dispersion'] = self.dispersion
        metrics['our'] = self.args.our

        defense_mode = "no_defense"
        if self.args.defense:
            defense_mode = "defense"
        if self.args.defense_all:
            defense_mode = "defense_all"
        if self.args.our:
            defense_mode = "our"

        filename = "metrics_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(self.args.num_passive,
            self.args.batch_size,
            self.args.epochs,
            self.args.lr_passive, 
            self.args.lr_attack, 
            self.args.attack_epoch, 
            self.args.attack_id,
            self.args.use_emb,
            self.args.simple,
            self.args.division_mode,
            self.args.balanced,
            defense_mode,
            self.args.epsilon,
            self.args.round)
        if self.args.dataset == "criteo":
            filedir = "balanced" if self.args.balanced else "imbalanced"
            metrics_path = os.path.join(self.dir, self.args.attack, self.args.dataset, filedir, filename)
        else:
            metrics_path = os.path.join(self.dir, self.args.attack, self.args.dataset, filename)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
