import re
import os.path
import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import geckodriver_autoinstaller

##################
### Global Var ###
num_pages_of_reviews = 3 # 30 reviews per page
##################

geckodriver_autoinstaller.install()

# Set up Firefox driver, plus a small wait to allow things to load
driver = webdriver.Firefox()
driver.implicitly_wait(0.5)


# Delete previous data, if it exists
try:
    if(~os.path.exists("review_text.json")):
            os.remove("review_text.json") 
except:
    pass

# Enter book code and then use Selennium to get page source
book_file = open('books.txt', 'r')
titles_temp = book_file.readlines()
 
titles = []
# Strips the newline character
for title in titles_temp:
    titles.append(title)

data = []
for title in titles:
    print("Scraping: " + str(title))
    book_code = (re.search("(.*?)-", title)).group(1)
    url = "https://www.goodreads.com/book/show/" + str(book_code) + "/reviews?"

    driver.get(url)
    source = driver.page_source

    # Close popup if it appears
    try:
        wait = WebDriverWait(driver, 2)
        wait.until(EC.presence_of_element_located((By.CLASS, "Button Button--transparent Button--small Button--rounded")))
        close_popup_button = driver.find_element(By.CLASS, "Button Button--transparent Button--small Button--rounded")
        close_popup_button.click()
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "Formatted")))
    except:
        pass

    ### Logic to open more pages ###
    # Search for text and find button element to press by moving up the tree, loads 30 more reviews
    if(num_pages_of_reviews >= 2):
        button_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Show more reviews')]")
        button_label = button_text.find_element(By.XPATH, '..')
        more_reviews_button = button_label.find_element(By.XPATH, '..')
        more_reviews_button.send_keys(Keys.ENTER)

    wait = WebDriverWait(driver, 10)
    # Wait for button to load and then press it again, loads 90 total reviews
    for i in range(num_pages_of_reviews - 2):
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more reviews')]")))
        more_reviews_button.send_keys(Keys.ENTER)
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more reviews')]")))
    

    # Wait until review text loads and then find all reviews on page
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "Formatted")))
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "RatingStars__small")))
    time.sleep(2)
    review_text = driver.find_elements(By.CLASS_NAME, "Formatted") #selenium

    # Wait until review starts loads and then grab all reviews
    #review_rating = driver.find_elements(By.CLASS_NAME, "RatingStars__small")

    # Remove reviews without a star rating
    bad_reviews = []
    review_rating = []
    for i in range(len(review_text)):
        trunc_content_text = review_text[i].find_element(By.XPATH, '..')
        truncated_content = trunc_content_text.find_element(By.XPATH, '..')
        review_text_content = truncated_content.find_element(By.XPATH, '..')
        review_text_section = review_text_content.find_element(By.XPATH, '..')
        review_card_content = review_text_section.find_element(By.XPATH, '..')
        try:
            rating = review_card_content.find_element(By.CLASS_NAME, "RatingStars__small")
            review_rating.append(rating)
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
            
    for i in range(len(review_text)):
        data.append({'label': review_rating[i], 'text': review_text[i].text})



# Save all reviews to file
with open('review_text.json', 'a') as filehandle:
    json_object = json.dumps(data)
    filehandle.write(json_object)

print("Reviews scraped: " + str(len(data)))

driver.quit()