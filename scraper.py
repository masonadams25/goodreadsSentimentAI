import requests
from bs4 import BeautifulSoup
import re
import torch
import os.path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import geckodriver_autoinstaller

from transformers import pipeline
from transformers import AutoTokenizer

geckodriver_autoinstaller.install()

# Set up Firefox driver, plus a small wait to allow things to load
driver = webdriver.Firefox()
driver.implicitly_wait(0.5)

# Enter book code and then use Selennium to get page source
book_code = "44492285"
url = "https://www.goodreads.com/book/show/" + book_code + "/reviews?"
driver.get(url)
source = driver.page_source

# Legacy, initially used Beautiful Soup but needed Selenium to press buttons
#response = requests.get(url)
#soup = BeautifulSoup(response.content, 'lxml') 
#soup = BeautifulSoup(source, 'lxml')

# Search for text and find button element to press by moving up the tree, loads 30 more reviews
wait = WebDriverWait(driver, 10)
button_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Show more reviews')]")
button_label = button_text.find_element(By.XPATH, '..')
more_reviews_button = button_label.find_element(By.XPATH, '..')
more_reviews_button.send_keys(Keys.ENTER)

# Wait for button to load and then press it again, loads 90 total reviews
wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more reviews')]")))
more_reviews_button.send_keys(Keys.ENTER)

# Legacy, for beaufitul soup
#review_links = soup.find_all('span', {'class': 'Text Text__body3'})
#review_text = soup.find_all('span', {'class': 'Formatted'}) #soup

# Wait until review text loads and then find all reviews on page
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "Formatted")))
review_text = driver.find_elements(By.CLASS_NAME, "Formatted") #selenium

# If file does not exist, save all reviews to text file
if(~os.path.exists("review_text.txt")):
    file = open("review_text.txt", "a") 

    # Write all reviews to a file for safe keeping
    for review in review_text:
        file.write(review.text)
        file.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Insantiaze the sentiment model from hugging face
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Create a tokenizer so we can be sure we do not feed the model too long strings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
scores = 0
labels = {}
num = len(review_text)

# Iterate through each review and begin to determine the sentiment of each
for review in review_text:
    # Check length of string, and only move forward if it is short enough
    tokenized_sentence = tokenizer.tokenize(review.text)
    if(len(tokenized_sentence) < 450):
        # Model determines sentiment
        out = sentiment_task(review.text)
        #print(out[0])

        # Add labels to dict and scores to a var
        label = str(out[0]['label'])
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + out[0]['score']
        print('label: ' + label + " score: " + str(out[0]['score']))
    # If it is too long, split it up and taken the avg of each segment 
    else:
        start = 0
        mean_scores = 0
        mean_labels = {}

        mod_token = len(tokenized_sentence) / 400
        mod = int(len(review.text) / 1000)

        for i in range(mod):
            out = sentiment_task(review.text[start:mod])

            mean_scores = (mean_scores * i + out[0]['score']) / (i+1)

            label = str(out[0]['label'])
            if label in mean_labels:
                mean_labels[label] = mean_labels[label] + 1
            else:
                mean_labels[label] = 1
                
        final_label = max(mean_labels, key=mean_labels.get, default="N/A")
        label = final_label
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + mean_scores
        print('label: ' + str(final_label) + " score: " + str(mean_scores))


print("\n\n\n")
print("Avg score: " + str(scores/num))
print(str(labels))

        

    






