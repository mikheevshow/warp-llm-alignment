"""
This file containce an implementation of WARP alignment method presented in
the following article https://arxiv.org/pdf/2406.16768
"""

import copy
import torch

from typing import List, Tuple
from torch.utils.data import (
    DataLoader
)
from torch.optim import Adam

from warp_config import WARPConfig
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    sentime
)

from datasets import Dataset, load_metric

class WARPTrainer:
    """
    WARP Trainer implementation class
    """
    def __init__(self, 
                 config: WARPConfig,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 reward_model: PreTrainedModel,
                 reward_tokenizer: PreTrainedTokenizer,
                 train_dataset: Dataset,
                 eval_dataset: Dataset):
        
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer

        self.santiment_classifier = pipeline(
            task = "sentiment-analysis",
            model=self.reward_model, 
            tokenizer=self.reward_tokenizer)

        self.train_dataloader = DataLoader(
            dataset=train_dataset
        )

        self.eval_dataloader = DataLoader(
            dataset=eval_dataset
        )

    def _get_generation_score(self, responses: List[str]) -> torch.FloatTensor:
        """
        Generates scores of provided texts
        """
        def get_score(item) -> float:
            score = item['score']
            if item['label'] == 'LABEL_1':
                return score
            else:
                return -score

        sentiment_classifier = self.santiment_classifier

        sentiment_responses = sentiment_classifier(responses)
        scores_list = list(map(get_score, sentiment_responses))

        return torch.FloatTensor(scores_list)
    
    def _process_logitss(self, encoded_responses: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        generated_token_probs = []
        for i in range(log_probs.size(0)):
            sequence_probs = []
            for j in range(log_probs.size(1)):
                generated_token_id = encoded_responses[i, j].item()
                token_prob = log_probs[i, j, generated_token_id - 1].item()
                sequence_probs.append(token_prob)
            generated_token_probs.append(sequence_probs)

        return torch.Tensor(generated_token_probs)
    
    def _generate(self, 
                  model: PreTrainedModel, 
                  tokenizer: PreTrainedTokenizer,
                  promts: List[str],
                  generation_config: GenerationConfig,
                  pad_token_id) -> Tuple[List[str], torch.Tensor, torch.FloatTensor]:
        with torch.no_grad():

            encoded_promts = tokenizer(promts, padding=True, truncation=True, return_tensors="pt")
            outputs = model.generate(imputs=encoded_promts['input_ids'], 
                                    generation_config=generation_config,
                                    attention_mask=encoded_promts != pad_token_id)
            
            decoded_responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs['sequences']]
            encoded_responses: torch.Tensor = outputs['sequences']
            logitss = torch.stack(outputs['logits'], 1)

            return decoded_responses, encoded_responses, logitss
        
    def _slerp(self, model: PreTrainedModel, models: List[PreTrainedModel], l: float) -> PreTrainedModel:
        """
        This function do spherical interpolation. 
        """
        pass

    def train(self):
        config = self.config

        generation_config = GenerationConfig(
            max_new_tokens=30,
            min_new_tokens=30,
            temperature=config.temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_logits=True,
        )

        model_init = copy.deepcopy(self.model)

        def repeat_generator():
            while True:
                yield from self.train_dataloader

        iter_dataloader = iter(repeat_generator())        

        for _ in range(config.iterations):
            rl_models = []
            for _ in range(config.rl_runs):

                model_m = copy.deepcopy(model_init)
                model_ema = copy.deepcopy(model_init)

                optimizer = Adam(model_m.parameters())

                for t in range(config.training_steps):
                    decoded_responses_m, encoded_responses_m, logitss_m = self._generate(
                        model=model_m,
                        tokenizer=self.tokenizer,
                        promts=data,
                        generation_config=generation_config,
                        pad_token_id=model_m.config.pad_token_id
                    )
        
                    _, encoded_responses_ema, logits_ema = self._generate(
                        model=model_ema, 
                        okenizer=self.tokenizer,
                        promts=data,
                        generation_config=generation_config,
                        pad_token_id=model_ema.config.pad_token_id)
                    
                    logprobs_m = torch.log_softmax(logitss_m, dim=-1)

                    pol_logprobs_m = self._process_logitss(encoded_responses_m, logitss_m)
                    pol_logprobs_ema = self._process_logitss(encoded_responses_m, logitss_m)

                    kl = pol_logprobs_m - pol_logprobs_ema
                    non_score_reward = (-config.kl_penalty * kl).sum(1)
                    rewards_of_generated_texts = self._get_generation_score(decoded_responses_m)
                    total_reward = (rewards_of_generated_texts + non_score_reward).unsqueeze(-1).unsqueeze(-1)

                    loss = (total_reward * logprobs_m).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        for model_m_ema_param, model_m_param in zip(model_ema.parameters(), model_m.parameters()):
                            mu = config.ema_update_rate
                            model_m_ema_param.data = (1 - mu) * model_m_ema_param.data + mu * model_m_param.data

            rl_models.append(model_m)

            model_slerp = self._slerp(model=model_init, models=rl_models, l=1/len(rl_models))
            with torch.no_grad():
                for model_init_param, model_slerp_param in zip(model_init.parameters(), model_slerp.parameters()):
                    eta = config.liti_update_rate
                    model_init_param.data = (1 - eta) * model_init_param.data + eta * model_slerp_param.data
