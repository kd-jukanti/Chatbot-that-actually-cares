# main script for training the model

import os, yaml, random
import numpy as np
from tqdm import tqdm
import torch
import pickle

from transformers import *
from src.dataloader import get_dataloader
from src.utils import initialize_exp
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)    

# the arguments are given from a config file
with open("/cephyr/users/anjalip/Alvis/project_dml/config.yaml") as file:
    config = yaml.safe_load(file)

logger, config = initialize_exp(config)

# logging the metrics to wandb
if(config["experiment"]["wandb_log"]):
    os.environ["WANDB_API_KEY"] = ''
    os.environ['WANDB_ENTITY']  = ''
    wandb.init(entity='', name=config["experiment"]["wandb_name"], project='GPT_Chatbot', dir='/cephyr/users/anjalip/Alvis/project_dml/wandb_logs')
    logger.info(f'wandb run id {wandb.run.id}')
    logger.info(f'wandb run name {wandb.run.name}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(config["mode"]["seed"])

#loading the tokenizer
logger.info("Loading the tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["model"])

#loading the GPT2LMHeadModel model with 'gpt2' pretrained weights
logger.info("Loading the model...")
model = GPT2LMHeadModel.from_pretrained(config["model"]["model"]).to(device)

#initializing special tokens and adding them to the model vocab
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

# specifying the maximum utterance length depending on the max history dialougues and maximum lnegth of the GPT2 model
utter_len = (config["hyperparameters"]["max_len"]-config["hyperparameters"]["max_diag_hist"]-2) // config["hyperparameters"]["max_diag_hist"]

# loading dataloaders
logger.info("Loading train and valid dataloaders...")
train_dataloader, val_dataloader = get_dataloader(config, utter_len,logger)

# specifying the optimizer and scheduler (to decay the learning rate)
logger.info("Loading the optimizer...")
t_total = len(train_dataloader) // config["hyperparameters"]["gradient_accumulation_steps"] *  config["hyperparameters"]["epochs"]
optim = torch.optim.AdamW(model.parameters(), lr=float(config["hyperparameters"]["lr"]), eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=t_total
    )

checkpoint_folder = os.path.join(config["model"]["model_loc"], config["model"]["model"],config["experiment"]["exp_id"])
last_epoch = 0
start_epoch = 0
decrease_counts = 0
best_loss = np.inf

# start of training loop
for epoch in range(start_epoch, config["hyperparameters"]["epochs"]):
    model.train()

    logger.info(f"#"*20 + f"    Epoch: {epoch}  " + "#"*20)
    train_losses = []
    valid_losses = []
    # start of a epoch
    for i, batch in enumerate(tqdm(train_dataloader)):
        # loading the batch
        input_ids, token_type_ids, lm_labels = batch

        input_ids, token_type_ids, lm_labels = \
            input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
        
        # getting the model output (Language modeling loss)
        outputs = model(
            input_ids=input_ids,
            token_type_ids = token_type_ids,
            labels = lm_labels
        )
        
        loss = outputs[0]

        # optimizing the loss by firts clipping it and using accumulating gradients
        # (general technique for NLP problems to avoid exploding gradients or vanishing gradients)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["hyperparameters"]["max_norm"])
        if epoch % config["hyperparameters"]["gradient_accumulation_steps"] == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()
        
        train_losses.append(loss)

    train_losses = [loss.item() for loss in train_losses]
    train_loss = np.mean(train_losses)
    logger.info(f"Train loss: {train_loss}")

    last_epoch += 1
    
    # validation on val_dataloader
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            
            input_ids, token_type_ids, lm_labels = batch
            input_ids, token_type_ids, lm_labels = \
                input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                labels = lm_labels
            )
            
            loss = outputs[0]
            
            valid_losses.append(loss)
        
        valid_losses = [loss.item() for loss in valid_losses]
        valid_loss = np.mean(valid_losses)

    logger.info(f"Valid loss: {valid_loss}")
    # dictionary for logging to wandb
    log_dict = {}
    log_dict['train_loss'] = train_loss
    log_dict['valid_loss'] = valid_loss
    if(config["experiment"]["wandb_log"]):
        logger.info(f"Logging metrics to wandb")        
        wandb.log(log_dict, step=epoch)

    # saving the best model where validation loss decreases
    if valid_loss < best_loss:
        best_loss = valid_loss
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': best_loss,
            'epoch': last_epoch
        }
        
        torch.save(state_dict, f"{checkpoint_folder}/best_model_train.ckpt")
        logger.info("*"*10 + "Current best checkpoint is saved." + "*"*10)
    # Stopping criterion
    # else:
    #     logger.info("Not a better validation score (%i / %i)."
    #                         % (decrease_counts, config["hyperparameters"]["stopping_criterion_decrease_counts_max"]))
    #     decrease_counts += 1

    # if decrease_counts > config["hyperparameters"]["stopping_criterion_decrease_counts_max"]:
    #     logger.info("Stopping criterion has been below its best value for more than %i epochs. Ending the experiment..." % config["hyperparameters"]["stopping_criterion_decrease_counts_max"])
    #     exit()
        
    logger.info(f"Best valid loss: {best_loss}")
        
logger.info("Training finished!")



