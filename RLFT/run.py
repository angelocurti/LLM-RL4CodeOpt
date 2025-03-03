import os
import sys
from Dataset.Dataloader import Dataloader
import torch
import numpy as np
import datetime
from torch.utils.data import Dataset
from reward_function import get_reward
from utils import extract_useful_code
from value_model import ValueModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from ppo import PPOTrainer
from tqdm import tqdm
import argparse
from utils import respond_to_batch
from huggingface_hub import login

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login(token='...')

###PARSING ARGUMENTS  
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default="RLoutput/LLaMa_3.2/", type=str, help="output directory")
parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B", type=str, help="path to load models")
parser.add_argument("--baseline_output_path", default=None, type=str, help="path to load models")
parser.add_argument("--dataset_path", default="Dataset/Code_Pairs_small.csv", type=str, help="path to load dataset")
parser.add_argument("--max_source_length", default=256, type=int, help="maximum source length")
parser.add_argument("--max_target_length", default=256, type=int, help="maximum target length")
parser.add_argument("--train_batch_size", default=8, type=int, help="train_batch_size")
parser.add_argument("--test_batch_size", default=4, type=int, help="test_batch_size")
parser.add_argument("--train_epochs", default=1, type=int, help="test_batch_size")
parser.add_argument("--run", default=1, type=int, help="run ID")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--kl_coef", type=float, default=0.05, help="KL Coefficient")
parser.add_argument("--kl_target", type=float, default=1, help="Adaptive KL Target")
parser.add_argument("--vf_coef", type=float, default=1e-3, help="Coefficient of the Value Error")
  
args = parser.parse_args()

###DATA LOADING
print("STARTED DATA LOADING")
data_loader = Dataloader(args.dataset_path)
data_loader.load(args.dataset_path)
#preparing dataset for training
data_loader.tokenize(tokenizer)
#splitting the dataset
train_data = data_loader.tokenized_dataset["train"]
print("DATA LOADING OK")

###MODEL LOADING
print("STARTED MODEL LOADING")
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
#model.to(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, return_tensors = 'pt', truncation=True, padding='max_length', max_length=args.max_source_length, padding_side = "left")
tokenizer.pad_token = tokenizer.eos_token
print("MODEL LOADING OK")

###REF MODEL LOADING
print("STARTED REF MODEL LOADING")
#ref model for KL divergence
model_ref = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
print("REF MODEL LOADING OK")

###VALUE MODEL LOADING
print("STARTED VALUE MODEL LOADING")
#value model
value_model = ValueModel(model_name=checkpoint).to(device)
print("VALUE MODEL LOADING OK")

###PPOCONFIG
print("STARTED PPO CONFIG")
#ppo trainer initialization
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":args.kl_target, "vf_coef":args.vf_coef}
ppo_trainer = PPOTrainer(model, model_ref, value_model, **ppo_config)
print("PPO CONFIG OK")

#helpers
n_element = 0
n_tot = len(train_data)
nsteps = 0
total_rewards = 0
total_seen = 0
pad_token = 0
eos_index = 255
i = 0

for ep in range(args.train_epochs):
    while i < len(train_data):
        
        print("STARTING STEP")
        #batch of data
        batch = train_data[i:i+args.train_batch_size]
        
        #loading inputs to fed model ###FIX (organizing in batch)
        source_ids,source_mask,target_ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        source_mask = source_mask.squeeze(1) #to have consistent dimensions
        #loading problem_id for test cases
        problem_ids = [batch[i]['problem_id'] for i in range(args.train_batch_size)]
        
        #model generation
        response_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)#, do_sample=True, temperature=0.7, top_p=0.92, top_k=50, repetition_penalty=1.2, no_repeat_ngram_size=3, num_return_sequences=1, length_penalty=1.0, num_beams=1).detach()[:,:]
        
        #padding to ensure consistency ###FIX(better handling possible with eos token)
        pad_length = 2*args.max_source_length - response_ids.shape[1]
        if pad_length > 0:
            padding = torch.full((1, pad_length), pad_token, dtype=torch.long).to(device)
            response_ids = torch.cat([response_ids, padding], dim=1)
        
        #taking only the generated code
        generated_ids = response_ids[:,eos_index:-1] #padding
        resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, \
                                                clean_up_tokenization_spaces=False)

        #filtering responses to have executable snippet of code
        resp_filtered = [extract_code(resp[i]) for i in range(args.train_batch_size)]

        #reward calculation
        reward,mean_rate,mean_execution_time,mean_memory_usage  = get_reward(test_cases = problem_ids[0], code_ids=generated_ids, codes = resp_filtered, tokenizer = tokenizer)
        
        #rewards stat
        total_rewards += sum(reward.sum(axis=-1).tolist()) #rewards stat
        total_seen += len(source_ids)
        
        ###PPO STEP
        print("STARTING PPO STEP")
        train_stats = ppo_trainer.step(source_ids, source_mask, generated_ids, reward.to(args.device))
        print("PPO STEP OK")
        
        #step statistics
        mean_kl = train_stats['objective/kl']
        mean_entropy = train_stats['objective/entropy']
        loss, pg_loss, vf_loss = train_stats['ppo/loss/total'], train_stats['ppo/loss/policy'], train_stats['ppo/loss/value']
        mean_advg, mean_return,mean_val = train_stats['ppo/policy/advantages_mean'], train_stats['ppo/returns/mean'], train_stats['ppo/val/mean']

        nsteps += 1
        i = min(i + args.train_batch_size, n_tot)

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
        
        print("STEP OK")

#load final model
path = args.output_path
path = os.path.join(path, 'checkpoints')
if not os.path.exists(path):
    os.makedirs(path)
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(path, "pytorch_model_ep%d.bin"%(ep))
torch.save(model_to_save.state_dict(), output_model_file)
