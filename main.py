from time import sleep
import time
import subprocess

print("\n*** running scraper ***\n")
start_scraper = time.time()
subprocess.run(["python3", "scraper.py"])
end_scraper = time.time()

print("Scraper ran in " + str(end_scraper - start_scraper) + " seconds")
sleep(1)


print("\n*** running model ***\n")
start_model = time.time()
subprocess.run(["python3", "model.py"])
end_model = time.time()
print("Model ran in " + str(end_model - start_model) + " seconds")

print("\nDone!")
print("Program ran in " + str(end_model - start_scraper) + " seconds")
