import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np

'''
going to need to brew install selenium, (cask) chromedriver
'''
class myWebDriver():

    def __init__(self):
        # Using Chrome to access web
        self.driver = webdriver.Chrome() #argument to path of chromedriver
        #self.driver.implicitly_wait(5) # seconds
        # Open the website
        self.driver.get('https://play2048.co/')
        cookies = self.driver.find_element_by_xpath("/html/body/div[@class = 'cookie-notice']/a[@class = 'cookie-notice-dismiss-button']")
        cookies.click()

        self.directionMap = [Keys.ARROW_UP, Keys.ARROW_DOWN, Keys.ARROW_RIGHT, Keys.ARROW_LEFT]

        self.game = self.driver.find_element_by_tag_name('body')

    def getScore(self):
        e = self.driver.find_element_by_class_name("score-container").text
        if '\n' not in e:
            return int(e)
        else:
            return int(e.split('\n')[0])

    def getBoard(self):
        b = np.zeros((4,4))
        tiles = self.driver.find_element_by_class_name("tile-container").find_elements_by_css_selector("*")
        for t in tiles:

            name = t.get_attribute("class")

            if 'tile tile' in name:
                name = name.replace('tile tile-','').replace(' tile-position-',':-:').replace(' tile-new', '').replace(' tile-merged','')
                value, location = name.split(':-:')
                column,row = location.split('-')
                b[int(row)-1, int(column)-1] = int(value)
        return b

    def getInfo(self):
        return None

    def move(self, d):

        oldScore = self.getScore()
        self.game.send_keys(self.directionMap[d])
        newScore = self.getScore()
        time.sleep(.05)

        return oldScore, newScore

    def close(self):
        self.driver.close()
