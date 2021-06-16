import os
import time
import math
import random
import dill as pk
import numpy as np
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def myprint(text, file):
    
    file = open(file, 'a')
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, file=file, flush=True)
    file.close()
    
    
def record_args(args, info, results):
    
    results[info.RESULT_ARGS]['Database'] = args.database
    results[info.RESULT_ARGS]['Profile ID'] = args.profile_id
    results[info.RESULT_ARGS]['Pair ID'] = args.pair_id
    
    results[info.RESULT_ARGS]['Num Inducing Points'] = args.num_inducing_point    
    results[info.RESULT_ARGS]['Batch Size'] = args.batch_size
    
    results[info.RESULT_ARGS]['Num Training Epoch'] = args.num_training_epoch
    results[info.RESULT_ARGS]['Learning Rate'] = args.lr
    results[info.RESULT_ARGS]['Momentum'] = args.momentum
    results[info.RESULT_ARGS]['Warmup Ratio'] = args.warmup_ratio    

    results[info.RESULT_ARGS]['Beta'] = args.beta
    results[info.RESULT_ARGS]['Num Tuning Budget'] = args.num_tuning_budget
    results[info.RESULT_ARGS]['Num Tuning Epoch'] = args.num_tuning_epoch
    
    
def print_hyperparameter(info, results):
    
    myprint('Trial Hyperparameters', info.FILE_STDOUT)
    for k, v in results[info.RESULT_ARGS].items():
        myprint(f'{k}: {v}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    
def print_model(info, model, likelihood):
    
    f = lambda para, value: myprint(f'Parameter name: {para} | value = {value.detach().cpu().numpy()}', info.FILE_STDOUT)

    myprint('Model Parameters', info.FILE_STDOUT)
    
    for i, mean in enumerate(model.mean_module):
        if i in model.meta.training_task_ids or i == model.meta.testing_task_id:
            f(f'mean_module.{i}.constant', mean.constant)
    
    f('covar_module_task.U', model.covar_module_task.U)
    f('covar_module_task.outputscale', model.covar_module_task.outputscale)
    f('covar_module_task.lengthscale', model.covar_module_task.lengthscale)

    f('covar_module_hyper.covar_module_hyper.offset', model.covar_module_hyper.covar_module_hyper.offset)

    f('covar_module_epoch.alpha', model.covar_module_epoch.alpha)
    f('covar_module_epoch.beta', model.covar_module_epoch.beta)
    f('covar_module_epoch.outputscale', model.covar_module_epoch.outputscale)
    
    f('likelihood.noise', likelihood.noise)
    
    myprint('-'*20, info.FILE_STDOUT)
    
    
def sample_inducing_points(base_X, num_inducing_points):
    
    index_X = np.arange(base_X.shape[0])
    np.random.shuffle(index_X)
    return base_X[index_X[:num_inducing_points], :]


def get_batch(X, Y, batch_size, if_train):
        
    batch_seq = np.arange(X.shape[0])
    if if_train: 
        np.random.shuffle(batch_seq)
        num_batch = X.shape[0] // batch_size
    else:
        num_batch = math.ceil(X.shape[0]/batch_size)
    
    for idx_batch in range(num_batch):
        
        batch_indices = batch_seq[idx_batch*batch_size:(idx_batch+1)*batch_size]
        batch_X, batch_Y = X[batch_indices], Y[batch_indices]
        
        yield batch_X, batch_Y
    

def reshape(info, input_tensor, mapping_matrix):
    
    return pad_sequence(torch.split(input_tensor, mapping_matrix.sum(1).tolist()), batch_first=True, padding_value=info.EXTREME_SMALL)
    
    
def get_query(info, meta, model, unobserved_mask, UCB):
    
    unobserved_UCB = UCB * unobserved_mask
    unobserved_UCB[torch.isnan(unobserved_UCB)] = info.EXTREME_SMALL
    
    query_configs = torch.unique(torch.where(unobserved_UCB==unobserved_UCB.max())[0])
    query_config = query_configs[torch.randint(query_configs.shape[0],(1,))[0]].item()
    query_epoch = np.argmin(model.covar_module_task.if_observed[meta.testing_task_id][query_config])

    model.covar_module_task.if_observed[meta.testing_task_id][query_config, query_epoch] = True
    unobserved_mask[query_config, query_epoch] = np.nan    
    
    return query_config, query_epoch


def get_max(mean, query_config):
    
    max_configs = torch.where(mean==mean.max())[0]
    if query_config in max_configs: max_config = query_config
    else: max_config = max_configs[torch.randint(max_configs.shape[0], (1,))[0]].item()    
    max_epochs = torch.where(mean[max_config]==mean[max_config].max())[0]
    max_epoch = max_epochs[torch.randint(max_epochs.shape[0], (1,))[0]].item()
    
    return max_config, max_epoch


def get_best(test_best):
    
    best_config, best_epoch = test_best['best_config'], test_best['best_epoch']    
    return best_config, best_epoch


def load_database(args, info, meta):

    configs = pk.load(open(info.FILE_CONFIG, 'rb'))
    profiles = pk.load(open(info.FILE_PROFILE, 'rb'))
    
    train_X, train_Y, test_X, test_Y = [], [], [], []
    train_rank, test_rank, all_observations, if_observed, if_hidden = defaultdict(list), {}, {}, {}, {}
    for task_id in range(len(meta.TASK2ID)):
        if task_id not in meta.training_task_ids and task_id != meta.testing_task_id: continue
        
        task_profile = profiles[args.profile_id][meta.ID2TASK[task_id]]
        all_observations[task_id] = np.zeros((meta.NUM_BASE_CONFIG, meta.NUM_EPOCH))
        if_observed[task_id] = np.full((meta.NUM_BASE_CONFIG, meta.NUM_EPOCH), False)
        if task_id == meta.testing_task_id: if_hidden[task_id] = np.full((meta.NUM_BASE_CONFIG, meta.NUM_EPOCH), False)
        
        config_performance = {}
        for config_id in range(meta.NUM_BASE_CONFIG):            
            if config_id in task_profile:
                available_epoch = len(task_profile[config_id])
                config_performance[config_id] = max(task_profile[config_id])
                all_observations[task_id][config_id, range(available_epoch)] = task_profile[config_id]
                                
                x = meta.convert_config(configs[config_id])
                x[meta.INFO2ID['task']] = task_id
                x = np.vstack([x] * available_epoch)
                x[:, meta.INFO2ID['epoch']] = (1 + np.arange(available_epoch)) / meta.NUM_EPOCH
                
                if task_id in meta.training_task_ids:
                    train_X.append(x); train_Y.append(task_profile[config_id])
                    if_observed[task_id][config_id, range(available_epoch)] = True
                elif task_id == meta.testing_task_id:
                    test_X.append(x); test_Y.append(task_profile[config_id])
                    if_hidden[task_id][config_id, range(available_epoch)] = True
                    
        config_performance = [(k,v) for k,v in sorted(config_performance.items(), key=lambda x: x[1], reverse=True)]
        for config_rank, (config_id, _) in enumerate(config_performance):
            if task_id in meta.training_task_ids:
                train_rank[config_id].append(config_rank)
            elif task_id == meta.testing_task_id:
                test_rank[config_id] = config_rank
                
        if task_id == meta.testing_task_id:
            test_best = {'best_value':config_performance[0][1], 'best_config':config_performance[0][0]}
            test_best['best_epoch'] = np.argmax(task_profile[test_best['best_config']])
                    
    train_X, test_X = map(lambda x: torch.vstack([torch.from_numpy(each) for each in x]).float().to(info.DEVICE), [train_X, test_X])
    train_Y, test_Y = map(lambda x: torch.cat([torch.from_numpy(each) for each in x]).float().to(info.DEVICE), [train_Y, test_Y])
        
    return train_X, train_Y, test_X, test_Y, train_rank, test_rank, test_best, all_observations, if_observed, if_hidden