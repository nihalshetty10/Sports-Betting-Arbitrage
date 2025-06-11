import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sys

def scrape_prizepicks_props():
    driver = uc.Chrome(headless=False)
    driver.get("https://app.prizepicks.com/")
    time.sleep(5)

    # for human test
    input("If CAPTCHA appears, complete it in the browser, then press ENTER here to continue...")

    try:
        wait = WebDriverWait(driver, 10)
        mlb_span = wait.until(EC.presence_of_element_located((By.XPATH, "//span[@class='name' and text()='MLB']")))
        mlb_button = mlb_span.find_element(By.XPATH, "..")
        mlb_button.click()
        print("Clicked MLB tab.")  # make sure on mlb and not nba
        time.sleep(4)
    except Exception as e:
        print("❌ Could not find or click MLB tab.")
        driver.quit()
        return pd.DataFrame()

    # Scroll down
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(20):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "li[aria-label]"))
        )
    except:
        print("❌ Timeout: No player cards found.")
        driver.quit()
        return pd.DataFrame()

    data = []
    cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
    print(f"Found {len(cards)} prop cards")

    for card in cards:
        try:
            player = card.find_element(By.ID, "test-player-name").text.strip()
            team = card.find_element(By.ID, "test-team-position").text.strip()
            stat_value = card.find_element(By.CSS_SELECTOR, "div[class*='heading-md']").text.strip()
            game_info = card.find_element(By.CSS_SELECTOR, "time[aria-label='Start Time']").text.strip()

            try:
                stat_type = card.find_element(By.CSS_SELECTOR, "span.break-words").text.strip()
            except:
                stat_type = ""

            if not stat_type:
                spans = card.find_elements(By.TAG_NAME, "span")
                for span in spans:
                    text = span.text.strip()
                    if text and any(k in text for k in ["Strikeout", "Hits", "Runs", "Bases", "RBIs", "Home"]):
                        stat_type = text
                        break

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
    if df.empty:
        print("❌ No props scraped.")
        return df

    df["prop_type"] = df["prop_type"].str.strip()

    # Opponent extraction
    def extract_opponent(game_info):
        try:
            return game_info.split()[1]
        except:
            return None

    df["opponent"] = df["game_info"].apply(extract_opponent)

    # MLB 7 stats
    mlb_stat_types = {
        "Hits", "Total Bases", "Home Runs",
        "RBIs", "Runs", "Pitcher Strikeouts", "Hitter Strikeouts"
    }

    df = df[df["prop_type"].isin(mlb_stat_types)].copy()

    if df.empty:
        print("❌ No MLB props scraped.")
    else:
        print("MLB props scraped!")

    return df[["player", "team", "opponent", "prop_type", "line", "odds"]]

if __name__ == '__main__':
    df = scrape_prizepicks_props()
    if not df.empty:
        print(df.head())
