from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

import numpy as np
import evaluate

import os
import torch
import json


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
else:
    print ("MPS device not found.")

print("Instantiating model...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
# Use a pipeline as a high-level helper
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("fpianz/roberta-english-book-reviews-sentiment", num_labels = 3)

review_text = []
review_rating = []

print("Loading dataset...")
# Open file and append reviews into seperate var
with open('review_text.json', 'r') as openfile:
    # Reading from json file
    data = json.load(openfile)

true_labels = {}
for i in range(len(data)):
    review_rating.append(data[i]['label'])
    review_text.append(data[i]['text'])

    label = data[i]['label']
    if label in true_labels: 
        true_labels[label] = true_labels[label] + 1
    else:
        true_labels[label] = 1

    if(data[i]['label'] == 'positive'):
        data[i]['label'] = 2
    elif(data[i]['label'] == 'neutral'):
        data[i]['label'] = 1
    elif(data[i]['label'] == 'negative'):
        data[i]['label'] = 0
    data[i].update(tokenizer(data[i]['text'], padding="max_length", truncation=True, max_length=512))

num_reviews = len(data)

split = 0.9

train_dataset = data[:int(num_reviews*split)]
eval_dataset = data[int(num_reviews*split):]

print("training model...")
# Begin training model
training_args = TrainingArguments(output_dir="test_trainer", per_device_train_batch_size=1,
        per_device_eval_batch_size=1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(

    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,

)

trainer.train()