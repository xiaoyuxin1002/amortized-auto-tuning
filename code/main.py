import json
import math
import argparse
import dill as pk
import numpy as np

import torch
import torch.optim as optim

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from transformers import get_linear_schedule_with_warmup

from info import Info
from meta import Meta
from model import MyGP
from util import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--database', type=str, default='HyperRec')
    parser.add_argument('--profile_id', type=int, default=0)
    parser.add_argument('--pair_id', type=int, default=0)
    
    parser.add_argument('--num_inducing_point', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    
    parser.add_argument('--num_training_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--num_tuning_budget', type=int, default=100)
    parser.add_argument('--num_tuning_epoch', type=int, default=3)

    return parser.parse_args()


def train(args, info, model, likelihood, optimizer, scheduler, mll, train_X, train_Y, if_offline):
    
    model.train()
    likelihood.train()
    
    report_batch = max(1, math.ceil(train_X.shape[0]/args.batch_size) // 5)
    for idx_epoch in range(args.num_training_epoch if if_offline else args.num_tuning_epoch):
                
        myprint(f'Start Training Epoch {idx_epoch}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)
        
        for idx_batch, (batch_X, batch_Y) in enumerate(get_batch(train_X, train_Y, args.batch_size, True)):
                        
            batch_output = model(batch_X)
            batch_loss = -mll(batch_output, batch_Y)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if idx_batch % report_batch == 0:
                myprint(f'Finish Training Epoch {idx_epoch} | Batch {idx_batch} | Loss {batch_loss.item():.4f}', info.FILE_STDOUT)
            
        scheduler.step()
        myprint('-'*20, info.FILE_STDOUT)
        
        
def test(args, info, meta, model, likelihood, unobserved_mask, test_X, test_Y, train_rank, test_rank, test_best, if_hidden):
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        
        mean, UCB = [], []
        for idx_batch, (batch_X, batch_Y) in enumerate(get_batch(test_X, test_Y, args.batch_size, False)):
            
            batch_output = model(batch_X)
            mean.append(batch_output.mean)
            UCB.append(batch_output.mean + args.beta*batch_output.variance.sqrt())
        
    mean, UCB, test_X, test_Y = map(lambda x: reshape(info, x, if_hidden[meta.testing_task_id]), [torch.cat(mean), torch.cat(UCB), test_X, test_Y])
                
    query_config, query_epoch = get_query(info, meta, model, unobserved_mask, UCB)
    max_config, max_epoch = get_max(mean, query_config)
    
    f = lambda message, config, epoch: myprint(f'{message}: Position ({config}, {epoch}) | Observation {test_Y[config, epoch].item():.4f} | Predicted Mean {mean[config, epoch].item():.4f} | UCB {UCB[config, epoch].item():.4f}', info.FILE_STDOUT)
        
    f('Query', query_config, query_epoch)
    myprint(f'Query Config Ranking: Train - {train_rank[query_config]}, Test - {test_rank[query_config]}', info.FILE_STDOUT)
    
    f('Max', max_config, max_epoch)
    myprint(f'Max Config Ranking: Train - {train_rank[max_config]}, Test - {test_rank[max_config]}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
        
    return test_X[query_config, query_epoch], test_Y[query_config, query_epoch], test_rank[max_config]
    
    
def main():
    
    args = parse_args()
    info = Info(args)
    meta = Meta(args, info)
    set_seed(info.SEED)
    
    myprint('='*20, info.FILE_STDOUT)
    myprint('Start Amortized Auto-Tuning', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)

    results = {info.RESULT_ARGS:{}, info.RESULT_QUERY_OBSERVATION:[], info.RESULT_MAX_RANKING:[]}
    record_args(args, info, results)
    print_hyperparameter(info, results)
    
    myprint('Load the Targeting Database', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    train_X, train_Y, test_X, test_Y, train_rank, test_rank, test_best, all_observations, if_observed, if_hidden = load_database(args, info, meta)
    inducing_points = sample_inducing_points(train_X, args.num_inducing_point)
    
    myprint('Start Training', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    model = MyGP(info, meta, all_observations, if_observed, inducing_points).to(info.DEVICE)
    likelihood = GaussianLikelihood().to(info.DEVICE)
    mll = VariationalELBO(likelihood, model, num_data=train_Y.shape[0])
    
    optimizer = optim.SGD([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=args.lr, momentum=args.momentum)    
    num_training_steps = args.num_training_epoch + args.num_tuning_budget * args.num_tuning_epoch
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    train(args, info, model, likelihood, optimizer, scheduler, mll, train_X, train_Y, True)
    
    myprint('Finish Offline Training', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    print_model(info, model, likelihood)
    
    myprint('Start Online Tuning', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    unobserved_mask = torch.from_numpy(~if_observed[meta.testing_task_id]).float().to(info.DEVICE)
    for idx_query in range(args.num_tuning_budget):
        
        myprint(f'Start Online Tuning Query {idx_query}', info.FILE_STDOUT)
        query_x, query_y, max_ranking = test(args, info, meta, model, likelihood, unobserved_mask, test_X, test_Y, train_rank, test_rank, test_best, if_hidden)
        results[info.RESULT_QUERY_OBSERVATION].append(query_y.item()); results[info.RESULT_MAX_RANKING].append(max_ranking)
        
        myprint(f'Incorporate New Observation from Query {idx_query}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)        
        train_X, train_Y = torch.vstack([train_X, query_x.reshape(1,-1)]), torch.cat([train_Y, query_y.unsqueeze(0)])        
        mll.num_data += 1
        train(args, info, model, likelihood, optimizer, scheduler, mll, train_X, train_Y, False)
        
    myprint('Finish Online Tuning', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    print_model(info, model, likelihood)
    
    myprint('Save the Results', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    json.dump(results, open(info.FILE_RESULT, 'w'))
    pk.dump(model, open(info.FILE_MODEL, 'wb'), -1)
    
    myprint('Finish Amortized Auto-Tuning', info.FILE_STDOUT)
    myprint('='*20, info.FILE_STDOUT)
        
    
if __name__=='__main__':
    main()