import requests
from bs4 import BeautifulSoup
import re
import torch

from transformers import pipeline
from transformers import AutoTokenizer

book_code = "44492285"
url = "https://www.goodreads.com/book/show/" + book_code + "/reviews?"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml') 

#review_links = soup.find_all('span', {'class': 'Text Text__body3'})
review_text = soup.find_all('span', {'class': 'Formatted'})

file = open("review_text.txt", "a") 

for review in review_text:
    file.write(review.text)
    file.write("\n\n\n")

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
scores = 0
labels = {}
num = 0

for review in review_text:
    tokenized_sentence = tokenizer.tokenize(review.text)
    if(len(tokenized_sentence) < 450):
        out = sentiment_task(review.text)
        print(out[0])
        label = str(out[0]['label'])
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + out[0]['score']
        num += 1
    
print("\n\n\n")
print("Avg score: " + str(scores/num))
print(str(labels))

        

    






