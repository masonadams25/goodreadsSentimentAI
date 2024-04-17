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
from selenium.common.exceptions import NoSuchElementException
import geckodriver_autoinstaller

from sklearn.metrics import balanced_accuracy_score

from transformers import pipeline
from transformers import AutoTokenizer

##################
### Global Var ###
num_pages_of_reviews = 10 # 30 reviews per page
num_reviews = num_pages_of_reviews * 30

##################

geckodriver_autoinstaller.install()

# Set up Firefox driver, plus a small wait to allow things to load
driver = webdriver.Firefox()
driver.implicitly_wait(0.5)

# Enter book code and then use Selennium to get page source
book_code = "44767458"
url = "https://www.goodreads.com/book/show/" + book_code + "/reviews?"
driver.get(url)
source = driver.page_source

# Legacy, initially used Beautiful Soup but needed Selenium to press buttons
#response = requests.get(url)
#soup = BeautifulSoup(response.content, 'lxml') 
#soup = BeautifulSoup(source, 'lxml')

### Logic to open more pages ###
# Search for text and find button element to press by moving up the tree, loads 30 more reviews
if(num_pages_of_reviews >= 2):
    wait = WebDriverWait(driver, 10)
    button_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Show more reviews')]")
    button_label = button_text.find_element(By.XPATH, '..')
    more_reviews_button = button_label.find_element(By.XPATH, '..')
    more_reviews_button.send_keys(Keys.ENTER)

# Wait for button to load and then press it again, loads 90 total reviews
for i in range(num_pages_of_reviews - 2):
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more reviews')]")))
    more_reviews_button.send_keys(Keys.ENTER)

# Legacy, for beaufitul soup
#review_links = soup.find_all('span', {'class': 'Text Text__body3'})
#review_text = soup.find_all('span', {'class': 'Formatted'}) #soup

# Wait until review text loads and then find all reviews on page
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "Formatted")))
review_text = driver.find_elements(By.CLASS_NAME, "Formatted") #selenium

# Wait until review starts loads and then grab all reviews
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "RatingStars__small")))
review_rating = driver.find_elements(By.CLASS_NAME, "RatingStars__small")

# Remove reviews without a star rating
bad_reviews = []
for i in range(len(review_text)):
    trunc_content_text = review_text[i].find_element(By.XPATH, '..')
    truncated_content = trunc_content_text.find_element(By.XPATH, '..')
    review_text_content = truncated_content.find_element(By.XPATH, '..')
    review_text_section = review_text_content.find_element(By.XPATH, '..')
    review_card_content = review_text_section.find_element(By.XPATH, '..')
    try:
        review_card_content.find_element(By.CLASS_NAME, "RatingStars__small")
    except NoSuchElementException:
        bad_reviews.append(i)

# Pop ratingless reviews
bad_reviews.reverse()
for index in bad_reviews:
    review_text.pop(index)
        

# Converts rating text into a label
true_labels = {}
for i in range(len(review_rating)):
    if(review_rating[i].get_attribute("aria-label") == "Rating 4 out of 5" or
       review_rating[i].get_attribute("aria-label") == "Rating 5 out of 5"):
        review_rating[i] = "positive"
    elif(review_rating[i].get_attribute("aria-label") == "Rating 3 out of 5"):
        review_rating[i] = "neutral"
    elif(review_rating[i].get_attribute("aria-label") == "Rating 2 out of 5" or
       review_rating[i].get_attribute("aria-label") == "Rating 1 out of 5"):
        review_rating[i] = "negative"

    label = review_rating[i]
    if label in true_labels:
        true_labels[label] = true_labels[label] + 1
    else:
        true_labels[label] = 1
        

# If file does not exist, save all reviews to text file
try:
    if(~os.path.exists("review_text.txt")):
        file = open("review_text.txt", "a") 

        # Write all reviews to a file for safe keeping
        for i in range(len(review_text)):
            file.write("~~~\n" + review_rating[i] + "\n~~~\n")
            file.write(review_text[i].text)
            file.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
except:
    pass

# Insantiaze the sentiment model from hugging face
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

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
    tokenized_sentence = tokenizer.tokenize(review.text)
    if(len(tokenized_sentence) < 450):
        # Model determines sentiment
        try:
            out = sentiment_task(review.text)
        except:
            out = sentiment_task(review.text[:100])
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
                
        final_label = max(mean_labels, key=mean_labels.get, default="neutral")
        label = final_label
        pred_rating.append(label)
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1
        scores = scores + mean_scores
        print('label: ' + str(final_label) + " score: " + str(mean_scores))

        #if(label == "N/A"):
        #    review_rating.insert(count, "N/A")
    count += 1

num_wrong_pos = abs(labels['positive'] - true_labels['positive'])
num_wrong_neu = abs(labels['neutral'] - true_labels['neutral'])
num_wrong_neg = abs(labels['negative'] - true_labels['negative'])

num_wrong_pos = abs(labels['positive'] - true_labels['positive'])
num_wrong = num_wrong_pos + num_wrong_neu + num_wrong_neg
num_right = num_reviews-num_wrong

print("Number of reviews: " + str(num_reviews))
print("Acc: " + str(num_right/num_reviews))
print("Balanced acc: " + str(balanced_accuracy_score(review_rating, pred_rating))) # Unbalenced for some reason?
print("Pred: " + str(labels))
print("True: " +str(true_labels))

file2 = open("predictions.txt", "a") 
file2.write("Real: " + str(review_rating))
file2.write("\nPred: " + str(pred_rating))

driver.quit()