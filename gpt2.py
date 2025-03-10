import pandas as pd
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

def pred_class(tokenizers, USP_model, stg_list, text):
    inputs = tokenizers(text, return_tensors='pt').to(device)  
    USP_model.to(device)   
    outputs = USP_model(**inputs)  
    logits = outputs.logits.squeeze().tolist()   
    pred_class_index = outputs.logits.argmax(dim=-1).item()
    return stg_list[pred_class_index], logits   

def USP_ee_pred(text):
    ee_list = open("EE_stg_index.txt", "r").read().splitlines() 
    tokenizer_EE = GPT2Tokenizer.from_pretrained('trained_models/USP_ee_model')
    USP_model_EE = GPT2ForSequenceClassification.from_pretrained('trained_models/USP_ee_model', from_tf=False)
    return pred_class(tokenizer_EE, USP_model_EE, ee_list, text)   

def USP_er_pred(text):
    er_list = open("ER_stg_index.txt", "r").read().splitlines()  
    tokenizer_ER = GPT2Tokenizer.from_pretrained('trained_models/USP_er_model')
    USP_model_ER = GPT2ForSequenceClassification.from_pretrained('trained_models/USP_er_model', from_tf=False)
    return pred_class(tokenizer_ER, USP_model_ER, er_list, text)