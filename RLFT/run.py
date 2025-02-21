import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importa il DataLoader dal modulo Dataset
from Dataset.Dataloader import Dataloader
import torch
import numpy as np
import datetime
from torch.utils.data import Dataset
from reward_function import get_reward
from value_model import CodeT5HeadWithValueModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModel
from ppo import PPOTrainer
from itertools import cycle
from tqdm import tqdm
import argparse
from utils import respond_to_batch
from huggingface_hub import login

parser = argparse.ArgumentParser()
## Required parameters  
parser.add_argument("--asp", default=2, type=int,
                    help="action space")  
parser.add_argument("--data_path", default=None, type=str,
                    help="data parent directory")  
parser.add_argument("--output_path", default="RLoutput/", type=str,
                    help="output directory")
parser.add_argument("--model_path", default="Salesforce/codet5p-770m", type=str,
                    help="path to load models")
parser.add_argument("--baseline_output_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--dataset_path", default="Dataset/Code_pairs_small.csv", type=str,
                    help="path to load dataset")
parser.add_argument("--max_source_length", default=256, type=int,
                    help="maximum source length")
parser.add_argument("--max_target_length", default=256, type=int,
                    help="maximum target length")
parser.add_argument("--train_batch_size", default=1, type=int,
                    help="train_batch_size")
parser.add_argument("--test_batch_size", default=4, type=int,
                    help="test_batch_size")
parser.add_argument("--train_epochs", default=1, type=int,
                    help="test_batch_size")
parser.add_argument("--run", default=1, type=int,
                    help="run ID")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--kl_coef", type=float, default=0.05, help="KL Coefficient")
parser.add_argument("--kl_target", type=float, default=1, help="Adaptive KL Target")
parser.add_argument("--vf_coef", type=float, default=1e-3, help="Coefficient of the Value Error")
  

args = parser.parse_args()
args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.version.cuda)

print("STARTED MODEL LOADING")
#load models
#model to be trained
num_gpus = torch.cuda.device_count()
print(f"Numero di GPU disponibili: {num_gpus}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation=True, padding='max_length', max_length=args.max_source_length)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)#, trust_remote_code=True, revision = "main")
model.to(args.device)
#model.config.decoder_start_token_id = 50256 
#model.config.pad_token_id = tokenizer.pad_token_id
#model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
print("MODEL LOADING OK")
print("STARTED REF MODEL LOADING")
#ref model for KL divergence
model_ref = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)#, trust_remote_code=True, revision = "main")
model_ref.to(args.device)
#model_ref.config.decoder_start_token_id = 50256 
#model_ref.config.pad_token_id = tokenizer.pad_token_id
#model = FSDP(model_ref, cpu_offload=CPUOffload(offload_params=True))
print("REF MODEL LOADING OK")
#value model
print("STARTED VALUE MODEL LOADING")
value_model = CodeT5HeadWithValueModel()
value_model.to(args.device)
print("VALUE MODEL LOADING OK")
print("STARTED PPO CONFIG")
#ppo trainer initialization
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":args.kl_target, "vf_coef":args.vf_coef}
ppo_trainer = PPOTrainer(model, model_ref, value_model, **ppo_config)
print("PPO CONFIG OK")
#load data 
print("STARTED DATA LOADING")
#data collator
def collate_fn(batch):
    """Converte liste in tensori per PyTorch DataLoader."""
    return {
        key: torch.tensor([item[key] for item in batch], dtype=torch.long)
        for key in batch[0].keys()
    }

data_loader = Dataloader(args.dataset_path)
data_loader.load(args.dataset_path)
#preparing dataset for training
data_loader.tokenize(tokenizer)
#splitting the dataset
data_loader.split_data()
train_data = data_loader.tokenized_dataset["train"]
test_data = data_loader.tokenized_dataset["test"]

# Verifica che la tokenizzazione sia stata eseguita correttamente
print("DATA LOADING OK")
print(data_loader.tokenized_dataset['train'].column_names)
#batching the data to achieve better efficiency
train_data = train_data.remove_columns([col for col in train_data.column_names if col not in ["input_ids", "attention_mask", "labels"]])
batched_data = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
#print(batched_data[0])  
#training
nsteps = 0
total_rewards = 0
total_seen = 0
device = args.device
print(device)
for ep in range(args.train_epochs):
    pbar = tqdm(batched_data, total=len(batched_data))
    for batch in pbar:
        #batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask,target_ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        
        #source_ids = source_ids.reshape(source_ids.size(0), -1)
        #source_ids.to(device)# Rimuove la dimensione 1
        #source_mask = source_ids.reshape(source_mask.size(0), -1)  # Rimuove la dimensione 1
        #target_ids = target_ids.reshape(target_ids.size(0), -1)  # Rimuove la dimensione 1 #synced_gpus=True
        source_ids = source_ids.reshape(source_ids.size(0), -1)
        source_mask = source_mask.reshape(source_mask.size(0), -1)  # Add this line
        target_ids = target_ids.reshape(target_ids.size(0), -1)  # Add this line
        
        # Print shapes for debugging
        
        # Proper model call
        '''outputs = model(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=target_ids
        )'''
        response_ids  = model.generate(source_ids, max_length=args.max_target_length).detach()[:,:]
        response_codes = tokenizer.batch_decode(response_ids, skip_special_tokens=True, \
                                                clean_up_tokenization_spaces=False)
        response_ids_ref  = torch.clone(model_ref.generate(source_ids, max_length=args.max_target_length).detach()[:,:])
        
        reward,mean_rate,mean_execution_time,mean_memory_usage  = get_reward(test_cases = None, code_ids=response_ids, base_ids=response_ids_ref, tokenizer=tokenizer)
        #print(reward.sum(axis=-1).tolist())
        print(source_mask.shape, response_ids.shape, response_ids_ref.shape)
        total_rewards += sum(reward.sum(axis=-1).tolist())
        total_seen += len(source_ids)
        print("STARTING PPO STEP")
        #PPO Step
        train_stats = ppo_trainer.step(source_ids, source_mask, response_ids, response_ids_ref, reward.to(args.device))
        print("PPO STEP OK")
        
        mean_kl = train_stats['objective/kl']
        mean_entropy = train_stats['objective/entropy']
        loss, pg_loss, vf_loss = train_stats['ppo/loss/total'], train_stats['ppo/loss/policy'], train_stats['ppo/loss/value']
        mean_advg, mean_return,mean_val = train_stats['ppo/policy/advantages_mean'], train_stats['ppo/returns/mean'], train_stats['ppo/val/mean']

        nsteps += 1

        #save the results
        with open(args.output_path+'results.csv', 'a') as f:
            f.write( datetime.datetime.now().strftime("%H:%M:%S") +  
                    ',' + str(args.run)+
                    ',' + str(args.train_batch_size)+
                    ',' + str(args.max_source_length)+
                    ',' + str(args.max_target_length)+
                    ',' + str(args.lr)+ 
                    ',' + str(ep)+ 
                    ',' + str(nsteps)+ 
                    ',' + str(round(sum(reward.sum(axis=-1).tolist())/len(source_ids), 4))+
                    ',' + str(mean_kl) +
                    ',' + str(mean_entropy) + 
                    ',' + str(loss.item()) + 
                    ',' + str(pg_loss.item()) + 
                    ',' + str(vf_loss.item()) + 
                    ',' + str(mean_advg.item()) + 
                    ',' + str(mean_return.item()) + 
                    ',' + str(mean_val.item()) + 
                    ',' + str(mean_rate) +
                    ',' + str(mean_execution_time) +
                    ',' + str(mean_memory_usage)
                    + '\n')


path = args.output_path
path = os.path.join(path, 'checkpoints')
if not os.path.exists(path):
    os.makedirs(path)
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(path, "pytorch_model_ep%d.bin"%(ep))
torch.save(model_to_save.state_dict(), output_model_file)
