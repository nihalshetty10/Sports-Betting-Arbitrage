import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_prizepicks_props():
    # Set up the Selenium driver (make sure chromedriver is in your PATH)
    driver = webdriver.Chrome()
    driver.get("https://app.prizepicks.com/")
    time.sleep(5)
    # Scroll to load more content
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    wait = WebDriverWait(driver, 20)
    # Wait for any player prop card to appear using a broader selector
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li[aria-label]")))

    data = []
    cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
    print(f"Found {len(cards)} cards")
    for card in cards:
        try:
            more_btn = card.find_element(By.CSS_SELECTOR, "button#test-more")
            less_btn = card.find_element(By.CSS_SELECTOR, "button#test-less")
            player = card.find_element(By.CSS_SELECTOR, "h3#test-player-name").text
            team = card.find_element(By.CSS_SELECTOR, "div#test-team-position").text
            stat_value = card.find_element(By.CSS_SELECTOR, "div.heading-md").text
            stat_type = card.find_element(By.CSS_SELECTOR, "span.break-words").text
            game_info = card.find_element(By.CSS_SELECTOR, "time[aria-label='Start Time']").text
            data.append({
                "player": player,
                "team": team,
                "prop_type": stat_type,
                "line": stat_value,
                "game_info": game_info,
                "more": more_btn.text,
                "less": less_btn.text
            })
        except Exception:
            continue
    driver.quit()
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = scrape_prizepicks_props()
    print(df.head())
#"hhsssss"
print("hello")
print("hello")
