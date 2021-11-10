# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import subprocess
subprocess.call(["pip", "install", "transformers==4.4.2"])
import logging
import json
import torch 

from transformers import BertTokenizer
from modeling_cpt import CPTModel, CPTForConditionalGeneration

logger = logging.getLogger(__name__)

tokenizer = ''

def model_fn(model_dir):
    logger.info(model_dir)
    model_path = f'{model_dir}/cpt_model/'
    logger.info(model_path)
    tokenizer_path = f'{model_dir}/cpt_tokenizer/'
    logger.info(tokenizer_path)
    device = get_device()
    model = CPTForConditionalGeneration.from_pretrained(model_path).to(device)
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    logger.info('model loaded')
    return model

def input_fn(json_request_data, content_type='application/json'):  
    input_data = json.loads(json_request_data)
    text_to_summarize = input_data['text']
    return text_to_summarize

def predict_fn(text_to_summarize, model):
    device = get_device()

    inputs = tokenizer(text_to_summarize, return_tensors="pt", max_length=512).to(device)
    outputs = model.generate(inputs['input_ids'], max_length=64, top_p=0.95)
    summary_txt = tokenizer.decode(outputs[0])
    return summary_txt
    
def output_fn(summary_txt, accept='application/json'):
    return json.dumps(summary_txt), accept

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device