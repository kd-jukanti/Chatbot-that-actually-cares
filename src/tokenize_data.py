# script for tokenizing the dialouges and saving into pickle files

from transformers import *
import argparse
import pandas as pd
from utils import process_token_list, save_as_pickle


special_characters = {'space' : 'Ġ',
                      'end' : [',','!','?','...','.'],
                      'comma' : "_comma_"}

def load_data(tokenizer, path):
    #load cleaned csv files
    dataset = pd.read_csv(path)
    utterance = dataset['utterance']
    conv_ids = dataset['conv_id']
    speaker_ids = dataset['speaker_idx']

    conv_dict = {}
    current_speaker = -1
    # make list of lists for each conversation
    # Lists of conversation -> 
    # Each conversation is a list where each adjacent list of tokens represent dialogue from alternative speakers
    for i, utter in enumerate(utterance):
        conv_id = conv_ids[i]
        speaker_idx = speaker_ids[i]
        
        
        modified = utter.strip().replace(special_characters['comma'], ',')
        tokenized = tokenizer.tokenize(modified)

        token_list = process_token_list(special_characters, tokenized)

        text = tokenizer.convert_tokens_to_string(token_list)

        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            current_speaker = -1

        if current_speaker != speaker_idx:
            conv_dict[conv_id].append(text)
            current_speaker = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {text}"

    utterance_count = 0
    dialogues = []
    for utter_list in conv_dict.values():
        utterance_count += len(utter_list)
        dialogues.append(utter_list)
    
    return dialogues, utterance_count

def save_tokenized(prefix, data_dir, dialogues, tokenizer):

    # Tokenize and save as pickle files
    ids = []
    for dialogue in dialogues:
        dialogue_ids = []
        for utter in dialogue:
            tokens = tokenizer.tokenize(utter)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            dialogue_ids.append(token_ids)
        ids.append(dialogue_ids)
        
    assert len(ids) == len(dialogues)

    # save_as_pickle(dialogues, f"{data_dir}/{prefix}_utters_{args.model}.pickle")
    save_as_pickle(ids, f"{data_dir}/{prefix}_ids_{args.model}.pickle")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/cephyr/users/anjalip/Alvis/project_dml/data", help="directory where .csv are stored")
    parser.add_argument('--prefix', type=str, default="train", help="part to be processed (train/valid/test)")
    parser.add_argument('--model', type=str, default="gpt2", help="name of the model")
    
    args = parser.parse_args()
    
    assert args.prefix in ["train","valid","test"]
    
    print("Loading the tokenizer...")
    
    #using GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)


    dialogues, utterance_count = load_data(tokenizer, f"{args.data_path}/{args.prefix}_clean.csv")

    save_tokenized(args.prefix, args.data_path, dialogues, tokenizer)