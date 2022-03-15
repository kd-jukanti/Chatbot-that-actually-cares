#Script for dataloaders

import torch
from torch.utils.data import DataLoader
from data.dataset import ED_dataset
from torch.nn.utils.rnn import pad_sequence

class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id
        
    def pad_collate(self, batch):
        input_ids, token_type_ids, labels =[], [], []
        for idx, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[2]))
            
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
        return input_ids, token_type_ids, labels

def get_dataloader(config, utter_len, logger):

    ppd = PadCollate(pad_id=config["specialTokens_id"]["pad_id"])
    train_dataset = ED_dataset('train', config, utter_len, logger)
    valid_dataset = ED_dataset('valid', config, utter_len, logger)
    
    train_loader = DataLoader(train_dataset, 
                                    collate_fn=ppd.pad_collate, 
                                    shuffle=True, 
                                    batch_size=config["hyperparameters"]["batch_size"])
    valid_loader = DataLoader(valid_dataset, 
                                    collate_fn=ppd.pad_collate,
                                    batch_size=config["hyperparameters"]["batch_size"])

    return train_loader, valid_loader
