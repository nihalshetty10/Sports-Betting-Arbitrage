import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Tabs to scrape
stat_categories = [
    "Hits",
    "Total Bases",
    "Runs",
    "Pitcher Strikeouts",
    "Hitter Strikeouts",
    "Earned Runs Allowed"
]

def scrape_prizepicks_props():
    driver = uc.Chrome(headless=False)
    driver.get("https://app.prizepicks.com/")
    time.sleep(5)

    input("⚠️ If CAPTCHA appears, complete it in the browser, then press ENTER here to continue...")

    # Click MLB tab
    try:
        wait = WebDriverWait(driver, 15)
        mlb_span = wait.until(EC.presence_of_element_located((By.XPATH, "//span[text()='MLB']")))
        mlb_button = mlb_span.find_element(By.XPATH, "..")
        mlb_button.click()
        print(" Clicked MLB tab.")
        time.sleep(4)
    except:
        print("❌ Could not find or click MLB tab.")
        driver.quit()
        return pd.DataFrame()

    all_data = []

    for stat in stat_categories:
        print(f"\n Clicking '{stat}' ")
        try:
            filter_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[.//text()[contains(.,'{stat}')]]"))
            )
            driver.execute_script("arguments[0].click();", filter_button)
            time.sleep(2)
        except:
            print(f"⚠️ Could not find filter for '{stat}' — skipping.")
            continue

        # Scroll to load all props
        for _ in range(25):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.2)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li[aria-label]"))
            )
        except:
            print(f"⚠️ No props found for '{stat}' — skipping.")
            continue

        cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
        print(f" Found {len(cards)} cards under '{stat}'")

        valid_count = 0

        for card in cards:
            try:
                buttons = card.find_elements(By.CSS_SELECTOR, "button")
                has_more = any("More" in b.text for b in buttons)
                has_less = any("Less" in b.text for b in buttons)
                if not (has_more and has_less):
                    continue

                valid_count += 1

                player = card.find_element(By.ID, "test-player-name").text.strip()
                team = card.find_element(By.ID, "test-team-position").text.strip()
                stat_value = card.find_element(By.CSS_SELECTOR, "div[class*='heading-md']").text.strip()
                game_info = card.find_element(By.CSS_SELECTOR, "time[aria-label='Start Time']").text.strip()

                # ✅ Assign stat_type based on the tab
                stat_type = stat

                all_data.append({
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

        print(f" {valid_count} props with both Less and More under '{stat}'")

    driver.quit()

    df = pd.DataFrame(all_data)
    if df.empty:
        print("❌ No props scraped.")
        return df

    def extract_opponent(game_info):
        try:
            return game_info.split()[1]
        except:
            return None

    df["opponent"] = df["game_info"].apply(extract_opponent)

    print(f"\n✅ MLB props scraped: {len(df)}")
    df.reset_index(drop=True, inplace=True)

    with pd.option_context('display.max_rows', None):
        print(df)

    return df[["player", "team", "opponent", "prop_type", "line", "odds"]]

if __name__ == '__main__':
    df = scrape_prizepicks_props()
