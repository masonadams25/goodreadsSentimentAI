from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from transformers import pipeline
from transformers import AutoTokenizer

import json
 
# Insantiaze the sentiment model from hugging face
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
# Use a pipeline as a high-level helper
#sentiment_task = pipeline("text-classification", model="fpianz/roberta-english-book-reviews-sentiment", tokenizer=model_path)
sentiment_task = pipeline("text-classification", model="goodreads_model", tokenizer=model_path)

review_text = []
review_rating = []

# Open 
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

num_reviews = len(data)

# Create a tokenizer so we can be sure we do not feed the model too long strings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
scores = 0
labels = {}
pred_rating = []
num = len(review_text)
count = 0

# Iterate through each review and begin to determine the sentiment of each
for review in review_text:
    # Check length of string, and only move forward if it is short enough
    tokenized_sentence = tokenizer.tokenize(review)
    if(len(tokenized_sentence) < 450):
        # Model determines sentiment
        try:
            out = sentiment_task(review)
        except:
            out = sentiment_task(review[:100])
        #print(out[0])

        # Add labels to dict and scores to a var
        label = str(out[0]['label'])
        pred_rating.append(label)
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + out[0]['score']

        scores = scores + out[0]['score']
        print('label: ' + label + " score: " + str(out[0]['score']) + "  " + str(count) + "/" + str(num_reviews))

    # If it is too long, split it up and taken the avg of each segment 
    else:
        start = 0
        mean_scores = 0
        mean_labels = {}

        mod_token = len(tokenized_sentence) / 400
        mod = int(len(review) / 1000)

        for i in range(mod):
            out = sentiment_task(review[start:mod])

            mean_scores = (mean_scores * i + out[0]['score']) / (i+1)

            label = str(out[0]['label'])
            if label in mean_labels:
                mean_labels[label] = mean_labels[label] + 1
            else:
                mean_labels[label] = 1
                
        final_label = max(mean_labels, key=mean_labels.get, default="neutral")
        label = final_label
        pred_rating.append(label)
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + mean_scores
        print('label: ' + str(final_label) + " score: " + str(mean_scores) + "  " + str(count) + "/" + str(num_reviews))

        #if(label == "N/A"):
        #    review_rating.insert(count, "N/A")
    count += 1

target_names = ['negative', 'neutral', 'positive']

print("Number of reviews: " + str(num_reviews))
print("Acc: " + str(balanced_accuracy_score(review_rating, pred_rating))) 
print(classification_report(review_rating, pred_rating, target_names=target_names))
print("Pred: " + str(labels))
print("True: " +str(true_labels))

file2 = open("predictions.txt", "a") 
file2.write("Real: " + str(review_rating))
file2.write("\nPred: " + str(pred_rating))