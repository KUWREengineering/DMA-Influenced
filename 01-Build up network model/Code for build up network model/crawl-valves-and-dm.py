from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import geopandas as gpd

from secrets import *

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
valves = gpd.read_file('data/valves.kml', driver='KML').to_crs("EPSG:32647")

service = Service('/opt/chromedriver/chromedriver-v97')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
driver.get('https://extranet.mwa.co.th')

# check if we have been redirected to the agreement page
if driver.current_url == 'https://extranet.mwa.co.th/dana-na/auth/url_default/welcome.cgi':
    sleep(1)
    agree_button = driver.find_element(By.ID, 'sn-preauth-proceed_2')
    agree_button.click()

input_login = driver.find_element(By.ID, 'username')
input_password = driver.find_element(By.ID, 'password')
input_captcha = driver.find_element(By.ID, 'txtInput')
btn_submit = driver.find_element(By.ID, 'btnSubmit_6')
txt_captcha = driver.find_element(By.ID, 'txtCaptchaDiv')

sleep(1)
input_login.send_keys(MWA_EXTRANET_USER)
sleep(1)
input_password.send_keys(MWA_EXTRANET_PASS)
sleep(1)
input_captcha.send_keys(txt_captcha.text)
sleep(1)
btn_submit.click()

sleep(2)
driver.get('https://extranet.mwa.co.th/dana/home/launch.cgi?url=http%3A%2F%2F172.16.193.162%2Fsmartmap%2Findex.php')
input_username = driver.find_element(By.ID, 'username')
input_password = driver.find_element(By.ID, 'password')
btn_submit = driver.find_element(By.XPATH, "//input[@value='Submit']")
sleep(1)
input_username.send_keys(MWA_EXTRANET_USER)
sleep(1)
input_password.send_keys(MWA_EXTRANET_PASS)
sleep(1)
btn_submit.click()

sleep(5)
# start getting 'valve' info
for vid in valves.Description:
    print("Downloading HTML page for valve {}".format(vid))
    driver.get('https://extranet.mwa.co.th:11001/smartmap/valve_menu.php?id={}&work_id='.format(vid))
    with open('data/valve-html/{}.html'.format(vid), 'w') as f:
        f.write(driver.page_source)
    sleep(1)
#for id in idlist:
#    driver.get('https://extranet.mwa.co.th:11001/smartmap/valve_menu.php?id={}&work_id='.format(id))

# retrieve 'DM' info
driver.get('https://extranet.mwa.co.th:11001/smartmap/kml/dm_point.geojson')
with open('data/dm_point.geojson', 'w') as f:
    f.write(driver.page_source)
