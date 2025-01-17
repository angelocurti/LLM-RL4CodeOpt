import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importa il DataLoader dal modulo Dataset
from Dataset.DataLoader import DataLoader
import torch
import numpy as np
import datetime
from Dataset.DataLoader import DataLoader
from reward_function import get_reward
from value_model import CodeT5HeadWithValueModel
from transformers import AutoTokenizer
from ppo import PPOTrainer
from itertools import cycle
from tqdm import tqdm
import argparse
from utils import respond_to_batch


parser = argparse.ArgumentParser()
## Required parameters  
parser.add_argument("--asp", default=2, type=int,
                    help="action space")  
parser.add_argument("--data_path", default=None, type=str,
                    help="data parent directory")  
parser.add_argument("--output_path", default="../output", type=str,
                    help="output directory")
parser.add_argument("--load_model_path", default="Salesforce/codet5p-770m", type=str,
                    help="path to load models")
parser.add_argument("--baseline_output_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--dataset_path", default="../Dataset/Code_Pairs.csv", type=str,
                    help="path to load dataset")
parser.add_argument("--max_source_length", default=256, type=int,
                    help="maximum source length")
parser.add_argument("--max_target_length", default=256, type=int,
                    help="maximum target length")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="train_batch_size")
parser.add_argument("--test_batch_size", default=48, type=int,
                    help="test_batch_size")
parser.add_argument("--train_epochs", default=100, type=int,
                    help="test_batch_size")
parser.add_argument("--run", default=1, type=int,
                    help="run ID")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--kl_coef", type=float, default=0.05, help="KL Coefficient")
parser.add_argument("--kl_target", type=float, default=1, help="Adaptive KL Target")
parser.add_argument("--vf_coef", type=float, default=1e-3, help="Coefficient of the Value Error")
  

args = parser.parse_args()
args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load models
#model to be trained
model = CodeT5HeadWithValueModel()
model.load_base_model(args.load_model_path)
model.to(args.device)

#ref model for KL divergence
model_ref = CodeT5HeadWithValueModel()
model_ref.load_base_model(args.load_model_path)
model_ref.to(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.load_model_path, truncation=True, padding='max_length', max_length=args.max_source_length)

#value model
value_model = CodeT5HeadWithValueModel()

#ppo trainer initialization
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":args.kl_target, "vf_coef":args.vf_coef}
ppo_trainer = PPOTrainer(model, model_ref, value_model, **ppo_config)

#load data 
data_loader = DataLoader()
data_loader.load(args.dataset_path)
#preparing dataset for training
data_loader.tokenize(tokenizer, max_length = 256)
data_loader.split_data()
#splitting the dataset
train_data = data_loader.tokenized_dataset["train"]
test_data = data_loader.tokenized_dataset["test"]
#batching the data to achieve better efficiency
batched_data = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

#training
nsteps = 0
total_rewards = 0
total_seen = 0

for ep in range(args.train_epochs):
    pbar = tqdm(batched_data, total=len(batched_data))
    for batch in pbar:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask,target_ids, _ = batch
    
        response_ids  = torch.clone(respond_to_batch(model, source_ids, source_mask, \
                                                    max_target_length=args.max_target_length, \
                                                    top_k=args.asp, top_p=1.0).detach()[:,1:])
        response_codes = tokenizer.batch_decode(response_ids, skip_special_tokens=True, \
                                                clean_up_tokenization_spaces=False)

        response_ids_ref  = torch.clone(respond_to_batch(model_ref, source_ids, source_mask, \
                                                    max_target_length=args.max_target_length, \
                                                    top_k=args.asp, top_p=1.0).detach()[:,1:])     

        reward,mean_rate,mean_ast_match,mean_dfg_match, num_errors,num_errors_ref, num_nodes,num_nodes_ref  = get_reward(lang = args.l2, code_ids=response_ids,code_ref_ids=response_ids_ref, gold_ids=target_ids, tokenizer=tokenizer)
        
        total_rewards += sum([reward.sum(axis=-1).tolist()])
        total_seen += len(source_ids)

        #PPO Step
        train_stats = ppo_trainer.step(source_ids, source_mask, response_ids, response_ids_ref, reward.to(args.device))
        
        
        mean_kl = train_stats['objective/kl']
        mean_entropy = train_stats['objective/entropy']
        loss, pg_loss, vf_loss = train_stats['ppo/loss/total'], train_stats['ppo/loss/policy'], train_stats['ppo/loss/value']
        mean_advg, mean_return,mean_val = train_stats['ppo/policy/advantages_mean'], train_stats['ppo/returns/mean'], train_stats['ppo/val/mean']

        nsteps += 1

        #save the results
        with open(args.output_path+'results/.csv', 'a') as f:
            f.write( datetime.datetime.now().strftime("%H:%M:%S") +  
                    ',' + str(args.run)+
                    ',' + str(args.train_batch_size)+
                    ',' + str(args.max_source_length)+
                    ',' + str(args.max_target_length)+
                    ',' + str(args.lr)+ 
                    ',' + str(ep)+ 
                    ',' + str(nsteps)+ 
                    ',' + str(round(sum([reward.sum(axis=-1).tolist()])/len(source_ids), 4))+
                    ',' + str(mean_kl) +
                    ',' + str(mean_entropy) + 
                    ',' + str(loss.item()) + 
                    ',' + str(pg_loss.item()) + 
                    ',' + str(vf_loss.item()) + 
                    ',' + str(mean_advg.item()) + 
                    ',' + str(mean_return.item()) + 
                    ',' + str(mean_val.item()) + 
                    ',' + str(mean_rate) +
                    ',' + str(mean_ast_match) +
                    ',' + str(mean_dfg_match)
                    + '\n')


path = args.output_path
path = os.path.join(path, 'checkpoints')
if not os.path.exists(path):
    os.makedirs(path)
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(path, "pytorch_model_ep%d.bin"%(ep))
torch.save(model_to_save.state_dict(), output_model_file)
