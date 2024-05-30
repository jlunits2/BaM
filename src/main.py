'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for training and evaluation of BaM.

Version: 1.0
***********************************************************************
'''


import os
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm

from utils import *
from MIND import MIND
from ComiRec import ComiRec


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='movies', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--interest_num', default=4, type=int)
parser.add_argument('--add_pos', default=True, type=bool)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--test_iter', default=1000, type=int)
parser.add_argument('--max_iter', default=1000, type=int)
parser.add_argument('--patience', default=15, type=int)
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--sampled_n', default=1280, type=int)
parser.add_argument('--path', default='run1', type=str)
parser.add_argument('--model', default='comirec', type=str)
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--selection', default='hard', type=str) # the method of interest selection, 'bam' for BaM

args = parser.parse_args()


if __name__ == '__main__':
    
    setup_seed(args.random_seed)
    best_model_path = './runs/' + args.dataset + '/' + args.path + '/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    ### PREPARE DATA ###
    dataset = data_partition(args.dataset)
    [train_data, valid_data, test_data, user_count, item_count] = dataset
    
    if args.dataset == 'movies':
        seq_len = 50
    elif args.dataset == 'books':
        seq_len = 50

    test_loader = get_DataLoader(test_data, user_count, item_count, 1, seq_len, train_flag=0)
    
    ### DEFINE MODEL ###
    if args.model == 'comirec':
        model = ComiRec(item_count, args.hidden_size, args.batch_size, args.selection, args.interest_num, seq_len, add_pos=args.add_pos).to(args.device)
    elif args.model == 'mind':
        model = MIND(item_count, args.hidden_size, args.batch_size, args.interest_num, seq_len, routing_times=3, relu_layer=False, selection=args.selection).to(args.device)
    model.set_device(args.device)
    model.set_sampler(args.sampled_n, device=args.device)

    ### TEST ONLY ###
    if args.test: 
        load_model(model, best_model_path)
        model.eval()
       
        metrics = evaluate(model, test_loader, args.device)
        print(', '.join(['test ' + key + '[@10, @20]: ' + str([round(i, 4) for i in value]) for key, value in metrics.items()]))
        print()
        
    ### TRAIN MODEL ###
    else:
        
        f = open(os.path.join('./runs/'+ args.dataset + '/' + args.path + '/log.txt'), 'w')

        train_loader = get_DataLoader(train_data, user_count, item_count, args.batch_size, seq_len)
        valid_loader = get_DataLoader(valid_data, user_count, item_count, 1, seq_len, train_flag=0, valid_flag=1)


        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

        trials = 0

        print('training begin')

        model.loss_fct = loss_fn
        try:
            total_loss = 0.0
            iter = 0
            best_metric = 0
            
            for i, (users, pos_items, items, mask) in tqdm(enumerate(train_loader), desc='train'):
                model.train()
                iter += 1
                optimizer.zero_grad()
                pos_items = to_tensor(pos_items, args.device)
                interests, selection = model(to_tensor(items, args.device), pos_items, to_tensor(mask, args.device))

                loss = model.calculate_sampled_loss(selection, pos_items)

                loss.backward()
                optimizer.step()

                total_loss += loss
                
                if iter%args.test_iter == 0:
                    model.eval()
                    metrics = evaluate(model, valid_loader, args.device, k=[10])
                    log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / args.test_iter)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(['valid ' + key + '[@10]: %.4f' % value if key!= 'ratio' else 'valid ' + key + ': ' + str(value) for key, value in metrics.items()])
                    print(log_str)
                    f.write(log_str+'\n')
                    f.flush()

                    recall = metrics['recall'][0]
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > args.patience:
                            print("early stopping!")
                            break
                    
                    total_loss = 0.0

                if iter >= args.max_iter * 1000:
                    break

        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

        load_model(model, best_model_path)
        model.eval()

        metrics = evaluate(model, test_loader, args.device) # evaluate trained model
        print(', '.join(['test ' + key + '[@10, @20]: ' + str([round(i, 4) for i in value]) for key, value in metrics.items()]))
        f.write(', '.join(['test ' + key + '[@10, @20]: ' + str([round(i, 4) for i in value]) for key, value in metrics.items()]))
        print()
    
        f.flush
