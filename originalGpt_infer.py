# original GPT2 chatbot script using dialoGPT
# dialoGPT has GPT2 architecture but trained with conversational data
# code taken from
# https://www.machinecurve.com/index.php/2021/03/16/easy-chatbot-with-dialogpt-machine-learning-and-huggingface-transformers/

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_response(tokenizer, model, chat_round, chat_history_ids):

  # tokenizing the input given by user and adding eos token
  user_inpt_ids = tokenizer.encode(input('user: ') + tokenizer.eos_token, return_tensors='pt')

  # chat history updated with the user input
  bot_input_ids = torch.cat([chat_history_ids, user_inpt_ids], dim=-1) if chat_round > 0 else user_inpt_ids

  # dialoGPT has generate function that generates the output response given the input and the max length
  chat_history_ids = model.generate(bot_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)

  # decoding the tokens given by the model by skipping the special tokens
  response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
  print("bot: {}".format(response))

  return chat_history_ids


def chat_for_n_rounds(n=5, model="microsoft/DialoGPT-large"):

  # loading the tokenizer and model for DialoGPT-large
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModelForCausalLM.from_pretrained(model)
  print('#' * 20 + ' Start the Chatting ' + '#' * 20)

  chat_history_ids = None

  for chat_round in range(n):
    chat_history_ids = generate_response(tokenizer, model, chat_round, chat_history_ids)


if __name__ == '__main__':
  chat_for_n_rounds(5)