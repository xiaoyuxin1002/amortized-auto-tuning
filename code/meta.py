import json
import math
import numpy as np
from collections import OrderedDict


class Meta:
    
    def __init__(self, args, info):
        
        self.database = args.database
        self.normalize_decimal = lambda x, l, u: (x-l)/(u-l)
        self.normalize_log2 = lambda x, l, u: self.normalize_decimal(math.log2(x), math.log2(l), math.log2(u))
        self.normalize_log10 = lambda x, l, u: self.normalize_decimal(math.log10(x), math.log10(l), math.log10(u))
        
        if self.database == info.DATABASE_HyperRec:
            self.set_HyperRec()
            self.convert_config = self.convert_HyperRec
        elif self.database == info.DATABASE_LCBench:
            self.set_LCBench()
            self.convert_config = self.convert_LCBench
            
        self.set_train_test_task_ids(args, info)
        
    
    def set_train_test_task_ids(self, args, info):
        
        train_test_pairs = json.load(open(info.FILE_PAIRS, 'r'))
        training_tasks, testing_task = train_test_pairs[args.database][args.pair_id]

        self.training_task_ids = set([self.TASK2ID[task] for task in training_tasks.split(',')])
        self.testing_task_id = self.TASK2ID[testing_task]
                    
            
    def set_HyperRec(self):
        
        self.NUM_PROFILE = 2 # 0: 43; 1: 12
        self.NUM_BASE_CONFIG = 100
        self.NUM_EPOCH = 75
                        
        self.TASK2ID = {'ACTION40':0, 'AWA2':1, 'BOOKCOVER30':2, 'CALTECH256':3, 'CARS196':4,
                   'CIFAR10':5, 'CIFAR100':6, 'CUB200':7, 'FLOWER102':8, 'FOOD101':9,
                   'IP102':10, 'ISR':11, 'OIPETS':12, 'PLANT39':13, 'RESISC45':14,
                   'SCENE15':15, 'SDD':16, 'SOP':17, 
                   'IMAGENET64SUB1':18, 'IMAGENET64SUB2':19, 'IMAGENET64SUB3':20,
                   'PLACE365SUB1':21, 'PLACE365SUB2':22, 'PLACE365SUB3':23,
                   'SUN397SUB1':24, 'SUN397SUB2':25, 'SUN397SUB3':26}
        self.ID2TASK = {v:k for k,v in self.TASK2ID.items()}
        
        self.INFO2ID = {'batchsize':0, 
                   'model':1,
                   'optimizer':2, 'lr':3, 'weight_decay':4, 'betas0':5, 'betas1':6, 'momentum':7, 
                   'scheduler':8, 'step_size':9, 'gamma':10, 'max_lr':11, 'step_size_up':12, 'T_0':13, 'T_mult':14, 'eta_min':15,
                   'task':16, 'epoch':17}
        self.ID2INFO = {v:k for k,v in self.INFO2ID.items()}
        
        self.TREE = OrderedDict({'root':['batchsize'],
                         'branches':OrderedDict({'model':OrderedDict({'resnet34':[], 
                                                       'resnet50':[]}),
                                         'optimizer':OrderedDict({'Adam':['lr', 'weight_decay', 'betas0', 'betas1'],
                                                          'SGD':['lr', 'weight_decay', 'momentum']}),
                                         'scheduler':OrderedDict({'StepLR':['step_size', 'gamma'],
                                                          'ExponentialLR':['gamma'],
                                                          'CyclicLR':['max_lr', 'step_size_up', 'gamma'],
                                                          'CosineAnnealingWarmRestarts':['T_0', 'T_mult', 'eta_min']})})})
        
        self.CONDITION2ID = {CONDI:
                         {BRANCH: BRANCH_ID for BRANCH_ID, BRANCH in enumerate(BRANCHES)}  
                      for CONDI, BRANCHES in self.TREE['branches'].items()}
        
        self.NESTED2ID = OrderedDict({
                      self.INFO2ID[CONDI]: OrderedDict({
                          BRANCH_ID: [self.INFO2ID[LEAF] for LEAF in LEAVES]
                      for BRANCH_ID, (BRANCH, LEAVES) in enumerate(BRANCHES.items())})
                    for CONDI, BRANCHES in self.TREE['branches'].items()})
        
        
    def convert_HyperRec(self, config):
        
        x = np.zeros(len(self.INFO2ID))
        
        x[self.INFO2ID['batchsize']] = self.normalize_decimal(config['batchsize'], 32, 128)
        x[self.INFO2ID['model']] = self.CONDITION2ID['model'][config['model']]
        x[self.INFO2ID['optimizer']] = self.CONDITION2ID['optimizer'][config['optimizer']]
        x[self.INFO2ID['scheduler']] = self.CONDITION2ID['scheduler'][config['scheduler']]
        
        for k, v in config['optimizer_param'].items():
            if k == 'lr':
                x[self.INFO2ID[k]] = self.normalize_log10(v, 1e-4, 1e-1)
            elif k == 'weight_decay':
                x[self.INFO2ID[k]] = self.normalize_log10(v, 1e-5, 1e-3)
            elif k == 'betas':
                x[self.INFO2ID[f'{k}0']] = self.normalize_log10(v[0], 0.5, 0.999)
                x[self.INFO2ID[f'{k}1']] = self.normalize_log10(v[1], 0.8, 0.999)
            elif k == 'momentum':
                x[self.INFO2ID[k]] = self.normalize_log10(v, 1e-3, 1)
            
        for k, v in config['scheduler_param'].items():
            if k == 'step_size':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 2, 20)
            elif k == 'gamma':
                if config['scheduler'] == 'StepLR':
                    x[self.INFO2ID[k]] = self.normalize_log10(v, 0.1, 0.5)
                elif config['scheduler'] == 'ExponentialLR':
                    x[self.INFO2ID[k]] = self.normalize_log10(v, 0.85, 0.999)
                elif config['scheduler'] == 'CyclicLR':
                    x[self.INFO2ID[k]] = self.normalize_log10(v, 0.1, 0.5)
            elif k == 'max_lr':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 1e-4*1.1, 1e-1*1.5)
            elif k == 'step_size_up':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 1, 10)
            elif k == 'T_0':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 2, 20)
            elif k == 'T_mult':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 1, 4)
            elif k == 'eta_min':
                x[self.INFO2ID[k]] = self.normalize_decimal(v, 1e-4*0.5, 1e-1*0.9)
            
        return x
            
    
    def set_LCBench(self):
        
        self.NUM_PROFILE = 1
        self.NUM_BASE_CONFIG = 100
        self.NUM_EPOCH = 52
                
        self.TASK2ID = {'APSFailure': 0, 'Amazon_employee_access': 1, 'Australian': 2, 'Fashion-MNIST': 3, 'KDDCup09_appetency': 4,
                   'MiniBooNE': 5, 'adult': 6, 'airlines': 7, 'albert': 8, 'bank-marketing': 9,
                   'blood-transfusion-service-center': 10, 'car': 11, 'christine': 12, 'cnae-9': 13, 'connect-4': 14,
                   'covertype': 15, 'credit-g': 16, 'dionis': 17, 'fabert': 18, 'helena': 19, 
                   'higgs': 20, 'jannis': 21, 'jasmine': 22, 'jungle_chess_2pcs_raw_endgame_complete': 23, 'kc1': 24,
                   'kr-vs-kp': 25, 'mfeat-factors': 26, 'nomao': 27, 'numerai28.6': 28, 'phoneme': 29,
                   'segment': 30, 'shuttle': 31, 'sylvine': 32, 'vehicle': 33, 'volkert': 34}
        self.ID2TASK = {v:k for k,v in self.TASK2ID.items()}
        
        self.INFO2ID = {'batch_size':0, 'learning_rate':1, 'momentum':2, 'weight_decay':3, 
                   'num_layers':4, 'max_units':5, 'max_dropout':6,
                   'task':7, 'epoch':8}
        self.ID2INFO = {v:k for k,v in self.INFO2ID.items()}
        
        self.TREE = OrderedDict({'root':['batch_size', 'learning_rate', 'momentum', 'weight_decay', 'num_layers', 'max_units', 'max_dropout'],
                         'branches':OrderedDict({})})
        
        self.CONDITION2ID = {CONDI:
                         {BRANCH: BRANCH_ID for BRANCH_ID, BRANCH in enumerate(BRANCHES)}  
                      for CONDI, BRANCHES in self.TREE['branches'].items()}
        
        self.NESTED2ID = OrderedDict({
                      self.INFO2ID[CONDI]: OrderedDict({
                          BRANCH_ID: [self.INFO2ID[LEAF] for LEAF in LEAVES]
                      for BRANCH_ID, (BRANCH, LEAVES) in enumerate(BRANCHES.items())})
                    for CONDI, BRANCHES in self.TREE['branches'].items()})
        
        
    def convert_LCBench(self, config):
        
        x = np.zeros(len(self.INFO2ID))
        
        x[self.INFO2ID['batch_size']] = self.normalize_log2(config['batch_size'], 16, 512)
        x[self.INFO2ID['learning_rate']] = self.normalize_log10(config['learning_rate'], 1e-4, 1e-1)
        x[self.INFO2ID['momentum']] = self.normalize_decimal(config['momentum'], 0.1, 0.99)
        x[self.INFO2ID['weight_decay']] = self.normalize_decimal(config['weight_decay'], 1e-5, 1e-1)
        
        x[self.INFO2ID['num_layers']] = self.normalize_decimal(config['num_layers'], 1, 5)
        x[self.INFO2ID['max_units']] = self.normalize_log2(config['max_units'], 64, 1024)
        x[self.INFO2ID['max_dropout']] = self.normalize_decimal(config['max_dropout'], 0, 1)
        
        return x