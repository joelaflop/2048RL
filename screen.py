import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

'''
going to need to brew install selenium, (cask) chromedriver
'''

# Using Chrome to access web
driver = webdriver.Chrome()
driver.implicitly_wait(5) # seconds
# Open the website
driver.get('https://play2048.co/')

cookies = driver.find_element_by_xpath("/html/body/div[@class = 'cookie-notice']/a[@class = 'cookie-notice-dismiss-button']")
cookies.click()


score = driver.find_element_by_class_name("score-container")

game = driver.find_element_by_tag_name('body')

print(score.text)

game.send_keys(Keys.ARROW_UP)
time.sleep(1)
game.send_keys(Keys.ARROW_DOWN)
time.sleep(1)
game.send_keys(Keys.ARROW_UP)
time.sleep(1)
game.send_keys(Keys.ARROW_DOWN)
time.sleep(1)
game.send_keys(Keys.ARROW_UP)
time.sleep(1)
game.send_keys(Keys.ARROW_DOWN)
time.sleep(1)

print(score.text)


driver.close()
