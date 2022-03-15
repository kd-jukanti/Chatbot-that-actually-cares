# Chatbot-that-actually-cares
With the recent progress in machine learning and natural language processing techniques, people have found new ways to utilize them. Having said that, in the recent years, the aim for providing more deeper and emotionally stocked conversations to users has become popular. Therefore, the motivation for this project is to build an empathetic chatbot having the competence to listen to people, understand the user’s state and respond empathetically.

## Methodology
The work is done by considering a pre-trained GPT2 model and fine tune it on an empathetic dataset and check if the model improves its empathetic capability. One of the major challenges for this project is the evaluation metrics. Since there is no specific metric for evaluating the empathetic capability, the evaluation is done by letting people subjectively rate their experience with the chatbot.

GPT2 is a language model structured by stacking transformer decoder blocks. This is a generative model that is used to predict and generate text based on the previous words. This helps us to use GPT2 language models because the main aim is to generate text (generate responses based on the user input).


## Evaluation

Therefore, the current best method to analyze and evaluate a chatbot is by human judgement. This was done by conducting a survey to about 50 people and determine the chatbot’s performance. People are given a situation like an incident and are asked to convey their emotions with the chatbot. The user then analyzes the responses from the chatbot and rate it accordingly. To make it possible to rate on different aspects, some metrics are designed to evaluate the empathetic quality of the
chatbot. The metrics used for this evaluation are dialogue quality, human-likeness, fluency. These set of metrics constitutes the overall performance of the chatbot. Each of these metrics are rated between the range of 1 to 5. (1: very bad, 2: bad, 3:neutral, 4:good, 5: very good)

1. Fluency: One of the major factors for evaluating a conversation agent such as a chatbot is based on how well the chatbot is expressing itself easily and articulately. This can be determined by judging on whether the user has understood the responses and if the language seemed accurate.

2. Dialogue Quality or Relevance: This metric is used to measure the quality of the chatbot’s response. This can be determined by understanding if the response was a “sensible” reply or “a strange but understandable” reply or a totally “non-sensical” reply and rate accordingly.

3. Human-likeness: Human likeness is basically to determine if the responses from the chatbots manifests the understanding of the emotions of the person who is talking about their feelings or experience. In simple words, this metric is used to evaluate on how well the chatbots shows empathy.

## Results 

![image](https://user-images.githubusercontent.com/101395346/158382524-908c61f6-5ef8-4ea3-b531-a46815a48c46.png)

## Conclusion

This work is focused on building a chatbot with the ability to respond to a user in an empathetic manner by finetuning the GPT2 model with the empathetic dataset and evaluate the empathetic capability based on human judgement. The survey and the results showed that the fine-tuned GPT2 model yielded better empathetic capability when compared to the original GPT2 model.

## Instructions

-  Downloading the data: 
    - Download the train and valid csv files from the ED dataset homepage https://paperswithcode.com/dataset/empatheticdialogues
-  Preprocessing the csv files
    - This script outputs a cleaned version of train and valid csv files. Run the script: 
      ```
       python data/clean_csv.py --data_path '<path for the files>'
      ``` 
- Tokenization and saving the data as pickle files
  - This script provides the .pickle files for train and valid dialouges token ids. Run the script:      
      ```
        python src/tokenize_data.py --data_path '<path for the cleaned csv files>' --prefix '<train/valid>' --model '<gpt2 model need to be specified>'   
      ``` 
- Check the arguments for training in config.yaml
- Run the training file to start training
  - Run the script: 
      ```
        python trainer.py
      ```   
  - Best model will be saved in models directory, logger will be saved in experiments folder and if wandb logging is enabled, wandb logs the metrics
- Check the arguments for inference in inference_config.yaml and specify the checkpoint      
- Run the inference code
  - Run the script and you can start chatting with the chatbot
      ```
        python inference.py
      ```   
      
