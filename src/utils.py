import pickle, random
import logging
import subprocess
import os
from src.logger import create_logger
import torch
import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to process the dialogue (simple pre-processing such as making sure of sentence ending and begining)
def process_token_list(special_characters, token_list):

    token_list[0] = token_list[0].capitalize()
    for i, token in enumerate(token_list):
        if token in special_characters['end']:
            if i<len(token_list)-1:
                if token_list[i+1][0] != special_characters['space']:
                    token_list[i+1] = special_characters['space'] + token_list[i+1].capitalize()
                else:
                    token_list[i+1] = special_characters['space'] + token_list[i+1][1:].capitalize()
                
    _token_list = [token for token in token_list if token != special_characters['space'] and len(token)>0]
    if _token_list[-1] not in special_characters['end']:
        _token_list.append(special_characters['end'][-1])
        
    return _token_list

# function to save as pickle files
def save_as_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# function to switch speaker while appending into list of conversations
def switch_speaker(cur_speaker):
    if(cur_speaker == 'a'):
        return 'b'
    if(cur_speaker == 'b'):
        return 'a'

# Function to intialize a run
# creates a experiment id for saving to unique folder for each run
# creates experiment folder to save the logs
def initialize_exp(config):

    if not os.path.exists(config["experiment"]["dump_path"]):
        subprocess.Popen("mkdir -p %s" % config["experiment"]["dump_path"], shell=True).wait()

    
    if config["experiment"]["exp_id"] == None:
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(5))
                if not os.path.isdir(os.path.join(config["experiment"]["dump_path"], exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        config["experiment"]["exp_id"] = exp_id 

    if not os.path.exists(os.path.join(config["experiment"]["dump_path"], config["experiment"]["exp_id"])):
        subprocess.Popen("mkdir -p %s" % os.path.join(config["experiment"]["dump_path"], config["experiment"]["exp_id"]), shell=True).wait()
    checkpoint_folder = os.path.join(config["model"]["model_loc"], config["model"]["model"],config["experiment"]["exp_id"])
    if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
    
    logger = create_logger(os.path.join(config["experiment"]["dump_path"], config["experiment"]["exp_id"], 'logger.log'), rank=0)
    logger.info("============ Initialized logger ============")
    return logger, config
