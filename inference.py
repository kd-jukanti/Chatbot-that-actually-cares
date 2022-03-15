import os, yaml, random
import numpy as np
from tqdm import tqdm
import torch
from transformers import *
from src.utils import initialize_exp
from itertools import chain
import copy
from torch.nn import functional as F
from src.utils import switch_speaker

# hyperparameters loaded from config file
with open("/cephyr/users/anjalip/Alvis/project_dml/inference_config.yaml") as file:
    config = yaml.safe_load(file)

logger, config = initialize_exp(config)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

# code for top-p sampling
def top_p_sampling(input_ids_list, token_type_ids_list, next_speaker_id, utter_len):
    output_id = []
    res_id = [next_speaker_id]
    res_type_id = [next_speaker_id]
    for _ in range(utter_len):
        input_ids = list(chain.from_iterable(input_ids_list)) + res_id
        token_type_ids = list(chain.from_iterable(token_type_ids_list)) + res_type_id
        input_len = len(input_ids)
        
        remaining_index = config["hyperparameters"]["max_len"] - len(input_ids)
        input_ids += [config["specialTokens_id"]["pad_id"]] * remaining_index
        token_type_ids += [config["specialTokens_id"]["pad_id"]] * remaining_index

        assert len(input_ids) == len(token_type_ids), "Error, incorrect lengths"
        
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)  
        token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
        
        # taking the linear layer output for words in vocabulary (this is just the raw numbers from linear head layer)
        output = model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, input_len-1]
        # sort the outputs and  applying softmax to get probabilities        
        
        sorted_probs, sorted_idxs = torch.sort(output, descending=True)
        sorted_probs = F.softmax(sorted_probs, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # applying the top-p sampling algo by removing those words with cummulative probability greater than p
        idx_remove = cumsum_probs > config["hyperparameters"]["top_p"]

        # right shifted indices and making first token above the threshold
        idx_remove[:, 1:] = idx_remove[:, :-1].clone()
        idx_remove[:, 0] = 0

        indices_to_remove = sorted_idxs[idx_remove]
        sorted_probs[indices_to_remove] = 1e-8          
        
        # sample the tokens from the filtered distribution and check the terminating condition
        probs = F.softmax(sorted_probs, dim=-1)
        idx = torch.multinomial(probs, 1).squeeze(-1).squeeze(0).item()
        
        if len(output_id) == utter_len or idx == config["specialTokens_id"]["eos_id"]:
            break
        else:
            output_id.append(idx)
            res_id.append(idx)
            res_type_id.append(next_speaker_id)
            
    return output_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(config["mode"]["seed"])

# loading tokenizer
logger.info("Loading the tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["model"])

# loading the model
logger.info("Loading the model...")
model = GPT2LMHeadModel.from_pretrained(config["model"]["model"]).to(device)

# specifying special tokens and adding them to vocabulary
special_tokens = {
    'bos_token': config["specialTokens"]["bos_token"],
    'eos_token': config["specialTokens"]["eos_token"],
    'pad_token': config["specialTokens"]["pad_token"],
    'additional_special_tokens': [config["specialTokens"]["s1_token"], config["specialTokens"]["s2_token"]]
}
num_new_tokens = tokenizer.add_special_tokens(special_tokens)

vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
model.resize_token_embeddings(vocab_size)

config["specialTokens_id"] = {}
config["specialTokens_id"]["bos_id"] = vocab[config["specialTokens"]["bos_token"]]
config["specialTokens_id"]["eos_id"] = vocab[config["specialTokens"]["eos_token"]]
config["specialTokens_id"]["pad_id"] = vocab[config["specialTokens"]["pad_token"]]
config["specialTokens_id"]["s1_id"] = vocab[config["specialTokens"]["s1_token"]]
config["specialTokens_id"]["s2_id"] = vocab[config["specialTokens"]["s2_token"]]

# loading the checkpoint from which the model should infer
if os.path.exists(f"{config['model']['model_loc']}/{config['model']['ckpt']}"):
    logger.info("Loading the trained checkpoint...")
    ckpt = torch.load(f"{config['model']['model_loc']}/{config['model']['ckpt']}")
    model.load_state_dict(ckpt['model_state_dict'])

utter_len = (config["hyperparameters"]["max_len"]-config["hyperparameters"]["max_diag_hist"]-2) // config["hyperparameters"]["max_diag_hist"]

logger.info(f'Starting the conversation\nIf you want to abort, type {config["hyperparameters"]["end"]}')
# starting the evaluation
model.eval()
with torch.no_grad():
    current_speaker = 'a'
    input_ids_list = []
    token_type_ids_list = []
    t = 0
    output_id = None
    
    while True:
        if current_speaker == 'a':
            user_input = input("You: ") 
            if user_input == config["hyperparameters"]["end"]:
                print("Bot: Good bye.")
                break
            
            current_speaker_id = config["specialTokens_id"]["s1_id"]  
            # tokenizing the user input and adding the speaker 1 token
            input_id = [current_speaker_id] + tokenizer.encode(user_input)
            
            if t == 0:
                input_id = [config["specialTokens_id"]["bos_id"]] + input_id
        else:
            current_speaker_id = config["specialTokens_id"]["s2_id"]
            input_id = copy.deepcopy(output_id)
        
        # encoding the token type for model input
        token_type_id = [current_speaker_id] * len(input_id)
        
        # checking if the sentence ended to append to the complete lists
        if input_id[-1] == config["specialTokens_id"]["eos_id"]:
            input_id = input_id[:-1]
            token_type_id = token_type_id[:-1] 
        
        input_ids_list.append(input_id)
        token_type_ids_list.append(token_type_id)
        
        if t >= config["hyperparameters"]["max_diag_hist"]:
            input_ids_list = input_ids_list[1:]
            token_type_ids_list = token_type_ids_list[1:]
        
        # changing the speaker
        next_speaker = switch_speaker(current_speaker)
        if next_speaker == 'a':
            next_speaker_id = config["specialTokens_id"]["s1_id"]
        else:
            next_speaker_id = config["specialTokens_id"]["s2_id"]
        
        # calling the top-p sampling method to generate the bot response and decoding the tokens generated
        if current_speaker == 'a':
            output_id = top_p_sampling(input_ids_list, token_type_ids_list, next_speaker_id, utter_len)
            bot_response = tokenizer.decode(output_id)

            print(f"Bot: {bot_response}")
        
        current_speaker = next_speaker
        t += 1
