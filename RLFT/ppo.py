__all__ = ['AdaptiveKLController', 'FixedKLController', 'PPOTrainer']

import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torch.optim as optim
import torch
import collections
import time
import random
from transformers import RobertaTokenizer
from utils import (logprobs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix)


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

        
class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

    
class PPOTrainer:

    #Default training parameters
    default_params = {
        "lr": 1e-5,
        "adap_kl_ctrl": True, 
        "init_kl_coef": 100, 
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":0.1, 
        "batch_size": 16,
        "forward_batch_size": 4,
        "ppo_epochs": 4,
        "device": torch.device("cuda"),
        'adam_eps': 1e-8
    }

    def __init__(self, model, ref_model, value_model, **ppo_params):
        
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        #models
        self.ref_model = ref_model
        self.model = model
        self.value_model = value_model
        #optimizer
        self.optimizer = AdamW(model.parameters(), lr=self.ppo_params['lr'], eps=self.ppo_params['adam_eps'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor =1. / np.cbrt(2), patience= 100, verbose = True)
        self.metric = 0

        #selecting controller type
        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

    # this function implements a training step
    def step(self, source_ids, source_mask, response_ids, scores):
        
        bs = source_ids.size()[0] #batch_size
        timing = dict()
        #measuring time
        t0 = time.time()
        t = time.time()
        #forward pass with no grad, used only to gather logprobs and values
        logprobs, ref_logprobs, values = self.batched_forward_pass(source_ids, source_mask, response_ids)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        #the reward is composed by combining kl component rewards and execution reward(scores)
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        max_target_len = response_ids.size()[1]

        #minibatch training, as usual in PPO algorithms
        for i in range(bs):
            idx = idxs[i]
            curr_len = (np.array(response_ids.cpu()[idx,:])==self.ppo_params['eos_token_id']).argmax() + 1
            train_stats = self.train_minibatch(logprobs[idx:idx+1, :], values[idx:idx+1, :],
                                                rewards[idx:idx+1, :], source_ids[idx:idx+1],
                                                source_mask[idx:idx+1], response_ids[idx:idx+1,:])
            all_stats.append(train_stats)
        
        #collecting statistics
        timing['time/ppo/optimize_step'] = time.time()-t
        t = time.time()
        train_stats = stack_dicts(all_stats)
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef, response_ids = response_ids)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        #updating controller
        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats


    def batched_forward_pass(self, source_ids, source_mask, response_ids):

        with torch.no_grad():
            model_output = self.model(input_ids=source_ids, attention_mask=source_mask, labels=response_ids)
            ref_model_output = self.ref_model(input_ids=source_ids, attention_mask=source_mask, labels=response_ids)
            values = self.value_model(input_ids=source_ids, attention_mask=source_mask, labels=response_ids)
        values = values.detach()
        logprobs = logprobs_from_logits(model_output.logits, response_ids).detach()
        ref_logprobs = logprobs_from_logits(ref_model_output.logits, response_ids).detach()
        
        return logprobs, ref_logprobs, values


    def train_minibatch(self, logprobs, values, rewards, source_ids, source_mask, response_ids): 
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, source_ids, source_mask, response_ids)
        loss = loss_p + loss_v 
        self.optimizer.zero_grad()
        loss.backward() #backward pass
        self.optimizer.step()
        self.scheduler.step(self.metric)
        return train_stats


    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs #kl rewards
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        print ('kl reward', rewards.mean(axis=-1))
        rewards += scores
        print ('score reward', scores.sum(axis=-1))
        return rewards, non_score_reward, self.kl_ctl.value

    #loss function
    def loss(self, old_logprobs, values, rewards, source_ids, source_mask, response_ids):
        
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response_ids.size()[1]
        
        #calculating advantages
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        
        #returns
        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()
        
        #forward passes to obtain new logprobs
        model_output = self.model.forward(input_ids=source_ids, attention_mask=source_mask, labels=response_ids)
        vpred = self.value_model.forward(input_ids=source_ids, attention_mask=source_mask, labels=response_ids)
        logprob = logprobs_from_logits(model_output.logits, response_ids)
        
        #clipping values to ensure stability
        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        #value model loss calculation
        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        #policy model loss calculation
        ratio = torch.exp(logprob - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        #final loss
        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        #statistics and useful measurement
        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)

        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),)

        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_kl = torch.max(-mean_kl,mean_kl)
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }
        

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats