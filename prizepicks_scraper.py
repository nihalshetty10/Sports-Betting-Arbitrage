import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# MLB props
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

    try:
        wait = WebDriverWait(driver, 15)
        mlb_span = wait.until(EC.presence_of_element_located((By.XPATH, "//span[text()='MLB']")))
        mlb_button = mlb_span.find_element(By.XPATH, "..")
        mlb_button.click()
        print("✅ Clicked MLB tab.")
        time.sleep(4)
    except:
        print("❌ Could not find or click MLB tab.")
        driver.quit()
        return pd.DataFrame()

    all_data = []

    for stat in stat_categories:
        print(f"\n🔍 Clicking '{stat}' filter...")
        try:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            filter_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[normalize-space(text())='{stat}']"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", filter_button)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", filter_button)
            time.sleep(3.5)
        except:
            print(f"⚠️ Could not find filter for '{stat}' — skipping.")
            continue

        print("⏬ Scrolling to load all props...")
        prev_count = 0
        scroll_attempts = 0

        while scroll_attempts < 40:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
            current_count = len(cards)

            if current_count == prev_count:
                scroll_attempts += 1
            else:
                scroll_attempts = 0

            prev_count = current_count

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li[aria-label]"))
            )
        except:
            print(f"⚠️ No props found for '{stat}' — skipping.")
            continue

        cards = driver.find_elements(By.CSS_SELECTOR, "li[aria-label]")
        print(f"🧾 Found {len(cards)} cards under '{stat}'")

        valid_count = 0

        for card in cards:
            try:
                buttons = card.find_elements(By.CSS_SELECTOR, "button")
                labels = [b.text.strip() for b in buttons if b.text.strip() in ["Less", "More"]]
                if not ("Less" in labels and "More" in labels):
                    continue

                player = card.find_element(By.ID, "test-player-name").text.strip()
                if "\n" in player or len(player.splitlines()) > 1:
                    continue

                team = card.find_element(By.ID, "test-team-position").text.strip()
                position = team.split(" - ")[-1].strip()

                if stat in ["Hits", "Total Bases", "Runs", "Hitter Strikeouts"] and position == "P":
                    continue
                if stat == "Pitcher Strikeouts" and position != "P":
                    continue

                stat_value = card.find_element(By.CSS_SELECTOR, "div[class*='heading-md']").text.strip()
                game_info = card.find_element(By.CSS_SELECTOR, "time[aria-label='Start Time']").text.strip()

                all_data.append({
                    "player": player,
                    "team": team,
                    "prop_type": stat,
                    "line": stat_value,
                    "game_info": game_info
                })

                valid_count += 1

            except Exception:
                continue

        print(f"✅ {valid_count} props with both Less and More under '{stat}'")

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

    print(f"\n✅ MLB props scraped! Total: {len(df)}")
    df.reset_index(drop=True, inplace=True)

    with pd.option_context('display.max_rows', None):
        print(df)

    df_output = df[["player", "team", "opponent", "prop_type", "line"]]
    df_output.to_csv("mlb_prizepicks_props.csv", index=False)
    print("📁 Output saved to 'mlb_prizepicks_props.csv'")

    return df_output

if __name__ == '__main__':
    df = scrape_prizepicks_props()
