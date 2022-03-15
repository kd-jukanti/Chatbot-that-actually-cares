# Chatbot-that-actually-cares
With the recent progress in machine learning and natural language processing techniques, people have found new ways to utilize them. Having said that, in the recent years, the aim for providing more deeper and emotionally stocked conversations to users has become popular. Therefore, the motivation for this project is to build an empathetic chatbot having the competence to listen to people, understand the user’s state and respond empathetically.

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
      
