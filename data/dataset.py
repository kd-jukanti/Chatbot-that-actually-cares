# script for the custom dataset for the project

from torch.utils.data import Dataset
from itertools import chain

import torch
import copy
import pickle  
from src.utils import switch_speaker     
        

class ED_dataset(Dataset):
    def __init__(self, prefix, config, utter_len,logger):

        #loading the tokenized dialogues
        with open(f"{config['dataset']['data_dir']}/{prefix}_ids_{config['model']['model']}.pickle", 'rb') as f:
            dialogues_ids = pickle.load(f)
        
        if(prefix == 'valid'):
            dialogues_ids = dialogues_ids[:100]
        
        self.input_ids = []
        self.token_type_ids = []
        self.labels = []
        total_seq_ids = []
        # loop to concatenate alternative speaker's dialogues with the special speaker 1 and 2 special tokens
        for d, dialogue_ids in enumerate(dialogues_ids):
            cur_speaker = 'a'
            hists = []
            for t, token_ids in enumerate(dialogue_ids):
                if cur_speaker == 'a':
                    speaker_id = config["specialTokens_id"]["s1_id"]
                else:
                    speaker_id = config["specialTokens_id"]["s2_id"]
                    
                if len(hists) < config["hyperparameters"]["max_diag_hist"]:
                    hists.append([speaker_id] + token_ids)
                else:
                    hists = hists[1:] + [[speaker_id] + token_ids]
                    
                cur_speaker = switch_speaker(cur_speaker)
                total_seq_ids.append(copy.deepcopy(hists))
        
        # loop to set the inputs to the GPT2LMHeadModel 
        # input_ids  : Tokens of the dialouges
        # token_type_ids : Segment tokens to distinguish between diferent speakers
        # labels  : label for the language model
        for s, seq_ids in enumerate(total_seq_ids):
            if len(seq_ids) > 1 and seq_ids[-1][0] == config["specialTokens_id"]["s2_id"]:
                seq_ids[0] = [config["specialTokens_id"]["bos_id"]] + seq_ids[0]
                seq_ids[-1] = seq_ids[-1] + [config["specialTokens_id"]["eos_id"]]
                input_id = list(chain.from_iterable(seq_ids))
                total_len = 0
                for token_ids in seq_ids:
                    total_len += len(token_ids)
                    
                if total_len > config["hyperparameters"]["max_len"]:
                    seq_ids = [token_ids[:utter_len] for token_ids in seq_ids]
                    seq_ids[-1][-1] = config["specialTokens_id"]["eos_id"]
                    
                token_type_id = [[token_ids[0]] * len(token_ids) if t != 0 else [token_ids[1]] * len(token_ids) for t, token_ids in enumerate(seq_ids)]
                token_type_id = list(chain.from_iterable(token_type_id))

                lm_label = [[-100] * len(token_ids) if t != len(seq_ids)-1 else token_ids for t, token_ids in enumerate(seq_ids)]
                lm_label = list(chain.from_iterable(lm_label))
                
                assert len(input_id) == len(lm_label) and len(input_id) == len(token_type_id)
                
                self.input_ids.append(input_id)
                self.token_type_ids.append(token_type_id)
                self.labels.append(lm_label)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]
 
