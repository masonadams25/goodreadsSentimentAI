import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import geckodriver_autoinstaller
import re

book_code = "44492285"
url = "https://www.goodreads.com/book/show/" + book_code + "/reviews?"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml') 

review_links = soup.find_all('span', {'class': 'Text Text__body3'})
review_text = soup.find_all('span', {'class': 'Formatted'})

file = open("review_text.txt", "a") 

for review in review_text:
    file.write(review.text)
    file.write("\n\n\n")
    






