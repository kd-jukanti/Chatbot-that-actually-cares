# Script to clean the csv files downloaded from
#https://paperswithcode.com/dataset/empatheticdialogues
# Few rows in the dataset are inconsistent with irregular number of ',' seperated items.
# this has been understood and rows are processed separately according to their lengths

import pandas as pd
import numpy as np
from csv import reader
import argparse

def process_rowLength_10(row):
    others = row[5].split('\n')
    new_data = []
    new_data.append(row[:5])
    new_data[0].extend(others[0].split(','))

    for i in others[1:-1]:
        new_data.append(i.split(','))

    k = others[-1].split(',')
    k.extend(row[6:])
    new_data.append(k)
    return new_data

def process_rowLength_12(row):
    others = row[5].split('\n')
    new_data = []
    new_data.append(row[:5])
    new_data[0].extend(others[0].split(','))

    for i in others[1:-1]:
        new_data.append(i.split(','))

    
    k = others[-1].split(',')
    k.append(row[6])
    k1 = row[7].split('\n')
    k.extend(k1[0].split(','))
    new_data.append(k)

    k = k1[1].split(',')
    k.extend(row[8:])
    new_data.append(k)
    return new_data

def clean_csv(path):
    data = []
    with open(path, 'r') as read_obj:
        
        csv_reader = reader(read_obj)
        for row in csv_reader:

            if(len(row) == 8):
                data.append(row)

            elif(len(row) == 10):
                data.extend(process_rowLength_10(row))

            elif(len(row) == 12):
                data.extend(process_rowLength_12(row))

            elif(len(row) == 9):
                data.extend([row[:7]+['']])
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./", help="Directory where .csv are stored")
    
    args = parser.parse_args()

    prefix = ['train','valid']
    for p in prefix:
        path = f"{args.data_path}/{p}.csv"
        save_path = f"{args.data_path}/{p}_clean.csv"
        data = clean_csv(path)
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_csv(save_path)         