import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sys

print(f"🚨 Running Python version: {sys.version}")

def scrape_prizepicks_props():
    driver = uc.Chrome(headless=False)

    print("🧭 Opening PrizePicks...")
    driver.get("https://app.prizepicks.com/")

    time.sleep(5)
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    print("⏬ Scrolling to load all props...")
    for _ in range(10): 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    wait = WebDriverWait(driver, 20)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li[aria-label]")))
    except:
        print("❌ Timeout: No player cards found.")
        driver.quit()
        return pd.DataFrame()

    data = []
    cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
    print(f"✅ Found {len(cards)} prop cards")

    for card in cards:
        try:
            player = card.find_element(By.ID, "test-player-name").text
            team = card.find_element(By.ID, "test-team-position").text
            stat_type = card.find_element(By.CSS_SELECTOR, "span.break-words").text
            stat_value = card.find_element(By.CSS_SELECTOR, "div[class*='heading-md']").text
            game_info = card.find_element(By.CSS_SELECTOR, "time[aria-label='Start Time']").text

            data.append({
                "player": player,
                "team": team,
                "prop_type": stat_type,
                "line": stat_value,
                "game_info": game_info,
                "odds": -119,
                "implied_prob": round(100 / (abs(-119) + 100), 4)
            })

        except Exception:
            continue

    driver.quit()

    df = pd.DataFrame(data)
    def extract_opponent(game_info):
        try:
            return game_info.split()[1] 
        except:
            return None

    df["opponent"] = df["game_info"].apply(extract_opponent)

    mlb_stat_types = {
        "Hits", "Total Bases", "Home Runs",
        "RBIs", "Runs", "Pitcher Strikeouts", "Hitter Strikeouts"
    }

    df = df[df["prop_type"].isin(mlb_stat_types)].copy()

    return df[["player", "team", "opponent", "prop_type", "line", "odds"]]

if __name__ == '__main__':
    df = scrape_prizepicks_props()
    if df.empty:
        print("❌ No MLB props scraped.")
    else:
        print(df.head())
