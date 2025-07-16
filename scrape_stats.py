import sys
import time
import re
import unicodedata
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from collections import defaultdict

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

def clean_name_for_matching(name):
    name = unicodedata.normalize('NFKD', name)
    name = name.replace('\xa0', ' ')  # Convert non-breaking space to space
    name = re.sub(r'[^\w\s\.\'\-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_name(name):
    name = name.replace('\xa0', ' ')
    name = unicodedata.normalize('NFKD', name)

    # Remove comma-based suffixes like ", W (10-6)", ", H (2)", etc.
    name = re.sub(r",[^\n]*", "", name)

    # Remove common trailing positions (PH, LF, C, 3B, RF, etc.)
    name = re.sub(r"\b(PH|PR|LF|RF|CF|C|1B|2B|3B|SS|DH|P|SP|RP|UTIL|OF)\b", "", name)

    # Remove any trailing multi-position formats like "PH-CF", "LF-DH"
    name = re.sub(r"\s+[A-Z]{1,3}([\-\/][A-Z]{1,3}){0,2}$", "", name)

    # Final cleanup: strip extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def parse_total_bases(driver):
    tb_dict = {}
    try:
        full_text = driver.find_element(By.TAG_NAME, "body").text
        tb_lines = re.findall(r"TB:\s*(.*?)(?=\n[A-Z]{2,4}:|\nTeam LOB:|\nWith RISP:|\Z)", full_text, re.DOTALL)

        for tb_data in tb_lines:
            entries = tb_data.split(";")
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                # Match: name followed optionally by a number
                match = re.match(r"^(.*?)(?:\s+(\d+))?$", entry)
                if match:
                    raw_name = match.group(1).strip()
                    tb = int(match.group(2)) if match.group(2) else 1
                    norm_name = normalize_name(raw_name)
                    tb_dict[norm_name] = tb
    except Exception as e:
        print("TB parsing error:", e)
    return tb_dict


def scrape_game_stats(driver, url):
    print(f"➡️ Scraping: {url}")
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
    )
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    tb_dict = parse_total_bases(driver)

    stats_dict = defaultdict(lambda: {
        "player": "",
        "R": 0,
        "H": 0,
        "Hitter SO": 0,
        "Pitcher SO": 0,
        "TB": 0,
        "ER": 0
    })

    tables = driver.find_elements(By.CSS_SELECTOR, "table")

    for table in tables:
        try:
            table_id = table.get_attribute("id")
            if not table_id:
                continue

            is_batting = "batting" in table_id
            is_pitching = "pitching" in table_id

            if not (is_batting or is_pitching):
                continue

            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")

            for row in rows:
                if "thead" in row.get_attribute("outerHTML"):
                    continue

                try:
                    name_cell = row.find_element(By.CSS_SELECTOR, "th[data-stat='player']")
                    raw_name = name_cell.text.strip()
                    if not raw_name:
                        continue
                    norm_name = normalize_name(raw_name)
                except:
                    continue

                cells = {c.get_attribute("data-stat"): c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")}
                if not cells:
                    continue

                player_stat = stats_dict[norm_name]
                player_stat["player"] = norm_name

                if is_batting:
                    player_stat["R"] += int(cells.get("R", 0) or 0)
                    player_stat["H"] += int(cells.get("H", 0) or 0)
                    player_stat["Hitter SO"] += int(cells.get("SO", 0) or 0)
                    if norm_name in tb_dict:
                        player_stat["TB"] += tb_dict[norm_name]
                        print(f"✅ Matched TB for '{norm_name}': {tb_dict[norm_name]}")
                    else:
                        print(f"❌ No TB match for '{norm_name}'")
                elif is_pitching:
                    player_stat["Pitcher SO"] += int(cells.get("SO", 0) or 0)
                    player_stat["ER"] += int(cells.get("ER", 0) or 0)

        except Exception:
            continue

    final_stats = []
    for player, stat in stats_dict.items():
        if any(stat[k] > 0 for k in ["R", "H", "Hitter SO", "Pitcher SO", "TB", "ER"]):
            final_stats.append(stat)

    print(f"✅ Scraped {len(final_stats)} player rows from this game")
    return final_stats

def run_scraper():
    if len(sys.argv) != 2:
        print("Usage: python scrape_game_stats.py MM-DD-YYYY")
        return

    input_date = sys.argv[1]
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
