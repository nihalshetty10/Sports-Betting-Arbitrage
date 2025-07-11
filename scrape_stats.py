import sys
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

def get_final_game_links(driver, date_obj):
    url = f"https://www.baseball-reference.com/boxes/?date={date_obj.strftime('%Y-%m-%d')}"
    print(f"\U0001F310 Opening: {url}")
    driver.get(url)
    time.sleep(3)

    td_cells = driver.find_elements(By.CSS_SELECTOR, "td.gamelink")
    print(f"\U0001F50D Found {len(td_cells)} <td class='gamelink'> elements")

    links = []
    for td in td_cells:
        if "Final" in td.text:
            try:
                a = td.find_element(By.TAG_NAME, "a")
                href = a.get_attribute("href")
                links.append(href)
            except:
                continue

    print(f"\U0001F517 Found {len(links)} final game links")
    return links

def normalize_name(name):
    return re.sub(r"\s+[A-Z]{1,2}$", "", name)

def parse_total_bases(driver):
    tb_dict = {}
    try:
        notes = driver.find_element(By.XPATH, "//*[contains(text(), 'TB:')]").text
        tb_line = next((line.strip() for line in notes.splitlines() if line.startswith("TB:")), "")
        tb_data = tb_line[3:].strip()
        entries = tb_data.split(";")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            match = re.match(r"([A-Za-z \.'\-]+)\s*(\d*)", entry)
            if match:
                name = match.group(1).strip().rstrip(".")
                tb = int(match.group(2)) if match.group(2).isdigit() else 1
                tb_dict[name] = tb
    except:
        pass
    return tb_dict

def scrape_game_stats(driver, url):
    print(f"➡️ Scraping: {url}")
    driver.get(url)
    time.sleep(3)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    tb_dict = parse_total_bases(driver)

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    print(f"\U0001F9FE Found {len(rows)} rows in stat tables")
    stats = []
    current_section = ""

    for row in rows:
        try:
            if 'thead' in row.get_attribute("outerHTML"):
                continue

            section_check = row.find_elements(By.CSS_SELECTOR, "th[data-stat='player']")
            if section_check and len(section_check) == 1:
                section_text = section_check[0].text.strip()
                if section_text in ["Batting", "Pitching"]:
                    current_section = section_text
                    continue

            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 2:
                continue

            name = row.find_element(By.TAG_NAME, "th").text.strip()
            values = [c.text.strip() for c in cols]
            if not any(values):
                continue

            norm_name = normalize_name(name)
            player_stats = {
                "player": norm_name,
                "R": int(values[1]) if values[1].isdigit() else 0,
                "H": int(values[4]) if values[4].isdigit() else 0,
                "Hitter SO": int(values[6]) if current_section == "Batting" and values[6].isdigit() else 0,
                "Pitcher SO": int(values[4]) if current_section == "Pitching" and values[4].isdigit() else 0,
                "ER": int(values[2]) if current_section == "Pitching" and values[2].isdigit() else 0,
                "TB": tb_dict.get(norm_name, 0),
            }
            stats.append(player_stats)
        except:
            continue

    print(f"✅ Scraped {len(stats)} player rows from this game")
    return stats

def run_scraper():
    if len(sys.argv) == 2:
        input_date = sys.argv[1]
    else:
        input_date = datetime.today().strftime("%m-%d-%Y")
        print(f"No date provided, defaulting to today: {input_date}")

    try:
        date_obj = datetime.strptime(input_date, "%m-%d-%Y")
    except ValueError:
        print("Invalid date format. Use MM-DD-YYYY.")
        return

    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    try:
        game_links = get_final_game_links(driver, date_obj)
        if not game_links:
            print("⚠️ No games found. Exiting early.")
            return

        all_stats = []
        for url in game_links:
            game_stats = scrape_game_stats(driver, url)
            all_stats.extend(game_stats)

        if not all_stats:
            print("❌ No player stats found in any game.")
            return

        df = pd.DataFrame(all_stats)
        df = df.groupby("player", as_index=False).sum(numeric_only=True)

        output_file = f"game_stats_{date_obj.strftime('%Y-%m-%d')}.csv"
        df.to_csv(output_file, index=False)
        print(f"📁 Saved to {output_file}")

    finally:
        driver.quit()

if __name__ == "__main__":
    run_scraper()
