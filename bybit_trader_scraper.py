"""
Bybit Copy Trading Trader Profile Scraper

This script scrapes trader profiles from Bybit Copy Trading platform.
It first fetches all trader IDs from the API, then visits each profile
page to extract detailed information.

Requirements:
    pip install playwright requests pandas
    playwright install chromium
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api2.bybit.com/fapi/beehive/public/v1/common/master-trader/list"
PROFILE_BASE_URL = "https://www.bybit.com/copyTrade/lead-trader/detail"
LISTING_PAGE_URL = "https://www.bybit.com/copyTrade/"

OUTPUT_CSV = "bybit_traders_full.csv"
OUTPUT_JSON = "bybit_traders_full.json"
ERROR_LOG = "scrape_errors.log"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

PAGE_SIZE = 20  # Traders per API page
MAX_PAGES = 10  # Maximum pages to fetch (adjust as needed)
MIN_DELAY = 2  # Minimum delay between profile visits (seconds)
MAX_DELAY = 4  # Maximum delay between profile visits (seconds)
MAX_RETRIES = 3  # Maximum retry attempts per trader

# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging():
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(ERROR_LOG, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# =============================================================================
# STEP 1: FETCH TRADER IDs FROM API
# =============================================================================


def fetch_trader_ids() -> list[dict]:
    """
    Fetch all trader IDs and basic info from Bybit API.
    
    Returns:
        List of trader dictionaries with at least 'masterTraderId' key.
    """
    all_traders = []
    
    for page in range(1, MAX_PAGES + 1):
        params = {
            "page": page,
            "limit": PAGE_SIZE
        }
        
        try:
            response = requests.get(
                API_BASE_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle different possible response structures
            traders = []
            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], list):
                    traders = data["result"]
                elif "data" in data and isinstance(data["data"], list):
                    traders = data["data"]
                elif "retObj" in data and isinstance(data["retObj"], list):
                    traders = data["retObj"]
            elif isinstance(data, list):
                traders = data
            
            if not traders:
                logger.info(f"No more traders found at page {page}, stopping pagination.")
                break
            
            all_traders.extend(traders)
            logger.info(f"Fetched page {page}: {len(traders)} traders (total: {len(all_traders)})")
            
            # Small delay between API calls
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed at page {page}: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response at page {page}: {e}")
            break
    
    logger.info(f"Total traders fetched: {len(all_traders)}")
    return all_traders


def extract_trader_id(trader: dict) -> str | None:
    """
    Extract trader ID from trader dictionary.
    
    Handles different possible key names for trader ID.
    """
    possible_keys = [
        "masterTraderId",
        "masterTraderID",
        "traderId",
        "traderID",
        "id",
        "userId",
        "userID",
        "leaderId",
        "leaderID"
    ]
    
    for key in possible_keys:
        if key in trader and trader[key]:
            return str(trader[key])
    
    return None

# =============================================================================
# STEP 2: SCRAPE PROFILE PAGE WITH PLAYWRIGHT
# =============================================================================


def scrape_trader_profile(page, trader_id: str) -> dict | None:
    """
    Scrape a single trader's profile page.
    
    Args:
        page: Playwright page object
        trader_id: The trader's unique ID
        
    Returns:
        Dictionary with all extracted fields, or None on failure.
    """
    profile_url = f"{PROFILE_BASE_URL}?id={trader_id}"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Navigate to profile page
            page.goto(profile_url, wait_until="networkidle", timeout=30000)
            
            # Wait for profile content to load (adjust selector as needed)
            page.wait_for_selector(".profile-card, .trader-info, [class*='trader'], [class*='profile']", 
                                   timeout=15000)
            
            # Additional wait for dynamic content
            page.wait_for_timeout(2000)
            
            # Extract all fields
            profile_data = extract_profile_data(page, trader_id, profile_url)
            
            return profile_data
            
        except PlaywrightTimeout as e:
            logger.warning(f"Timeout on attempt {attempt}/{MAX_RETRIES} for trader {trader_id}: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"Max retries reached for trader {trader_id}")
                return None
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error scraping trader {trader_id} (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2)
    
    return None


def extract_profile_data(page, trader_id: str, profile_url: str) -> dict:
    """
    Extract all profile fields from the loaded page.
    
    Args:
        page: Playwright page object with profile loaded
        trader_id: The trader's ID
        profile_url: Full profile URL
        
    Returns:
        Dictionary with all extracted fields.
    """
    
    def safe_text(selector: str, default: str = "") -> str:
        """Safely extract text content from a selector."""
        try:
            element = page.query_selector(selector)
            if element:
                text = element.inner_text().strip()
                return text if text else default
            return default
        except Exception:
            return default
    
    def safe_text_all(selectors: list[str], default: str = "") -> str:
        """Try multiple selectors, return first non-empty result."""
        for selector in selectors:
            result = safe_text(selector)
            if result:
                return result
        return default
    
    # Trader name - main heading
    trader_name = safe_text_all([
        "h1.trader-name",
        "h1[class*='name']",
        ".profile-title",
        ".trader-name",
        "[class*='trader-name']",
        "h1"
    ])
    
    # Badge - Silver, Gold, etc.
    badge = safe_text_all([
        ".badge",
        "[class*='badge']",
        ".trader-badge",
        ".level-badge",
        ".rank-badge"
    ])
    
    # Followers count
    followers = safe_text_all([
        "[class*='follower']",
        ".follower-count",
        ".stats-follower"
    ])
    # Clean up the text (remove "Followers" label if present)
    followers = ''.join(c for c in followers if c.isdigit() or c in ',.').strip()
    
    # Trading days
    trading_days = safe_text_all([
        "[class*='trading-day']",
        ".trading-days",
        ".stats-days"
    ])
    trading_days = ''.join(c for c in trading_days if c.isdigit() or c in ',.').strip()
    
    # Stability index (e.g., "4.0/5.0" or just "4.0")
    stability_index = safe_text_all([
        "[class*='stability']",
        ".stability-index",
        ".stability-score"
    ])
    
    # 7-Day views
    views_7d = safe_text_all([
        "[class*='view']",
        ".views-7d",
        ".weekly-views"
    ])
    views_7d = ''.join(c for c in views_7d if c.isdigit() or c in ',.').strip()
    
    # Bio/description (highlighted section)
    bio_description = safe_text_all([
        ".bio",
        ".description",
        "[class*='bio']",
        "[class*='description']",
        "[class*='intro']",
        ".trader-intro"
    ])
    
    # AUM in USDT
    aum_usdt = safe_text_all([
        "[class*='aum']",
        ".aum-value",
        ".stats-aum"
    ])
    
    # Total assets (may be hidden as *****)
    total_assets_usdt = safe_text_all([
        "[class*='total-asset']",
        ".total-assets",
        ".assets-total"
    ])
    
    # Profit sharing percentage
    profit_sharing_pct = safe_text_all([
        "[class*='profit-sharing']",
        "[class*='profitShare']",
        ".profit-share",
        ".sharing-rate"
    ])
    
    # Slots available
    slots_available = safe_text_all([
        "[class*='slot']",
        ".slots-available",
        ".available-slots"
    ])
    
    # Trading style tags (join with |)
    tags = []
    try:
        tag_elements = page.query_selector_all("[class*='tag'], .trading-style, .style-tag, [class*='style-tag']")
        for tag_elem in tag_elements:
            tag_text = tag_elem.inner_text().strip()
            if tag_text and len(tag_text) < 50:  # Avoid capturing large text blocks
                tags.append(tag_text)
    except Exception:
        pass
    trading_style_tags = " | ".join(tags) if tags else ""
    
    # Build result dictionary
    result = {
        "trader_id": trader_id,
        "trader_name": trader_name,
        "badge": badge,
        "followers": followers,
        "trading_days": trading_days,
        "stability_index": stability_index,
        "views_7d": views_7d,
        "bio_description": bio_description,
        "aum_usdt": aum_usdt,
        "total_assets_usdt": total_assets_usdt,
        "profit_sharing_pct": profit_sharing_pct,
        "slots_available": slots_available,
        "trading_style_tags": trading_style_tags,
        "profile_url": profile_url,
        "scraped_at": datetime.now().isoformat()
    }
    
    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main entry point for the scraper."""
    print("=" * 60)
    print("Bybit Copy Trading Trader Scraper")
    print("=" * 60)
    
    # Step 1: Fetch all trader IDs from API
    print("\n[Step 1] Fetching trader IDs from API...")
    traders = fetch_trader_ids()
    
    if not traders:
        logger.error("No traders found. Check API endpoint or network connection.")
        return
    
    # Extract IDs
    trader_ids = []
    for trader in traders:
        tid = extract_trader_id(trader)
        if tid:
            trader_ids.append(tid)
    
    print(f"Found {len(trader_ids)} unique trader IDs")
    
    if not trader_ids:
        logger.error("Could not extract any trader IDs from API response")
        return
    
    # Step 2: Scrape each profile with Playwright
    print(f"\n[Step 2] Scraping {len(trader_ids)} trader profiles...")
    print(f"Output: {OUTPUT_CSV}, {OUTPUT_JSON}")
    print(f"Errors logged to: {ERROR_LOG}")
    print("-" * 60)
    
    all_profiles = []
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu"
            ]
        )
        
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080}
        )
        
        page = context.new_page()
        
        # Scrape each trader
        for idx, trader_id in enumerate(trader_ids, start=1):
            total = len(trader_ids)
            
            # Random delay between requests
            if idx > 1:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
            
            # Scrape profile
            profile_data = scrape_trader_profile(page, trader_id)
            
            if profile_data:
                all_profiles.append(profile_data)
                trader_name = profile_data.get("trader_name", "Unknown")
                print(f"Scraped trader {idx}/{total}: {trader_name}")
            else:
                print(f"Failed to scrape trader {idx}/{total}: ID={trader_id}")
                logger.error(f"Failed to scrape profile for trader ID: {trader_id}")
        
        # Cleanup
        context.close()
        browser.close()
    
    # Step 3: Save results
    print("\n" + "=" * 60)
    print("[Step 3] Saving results...")
    
    if all_profiles:
        # Create DataFrame
        df = pd.DataFrame(all_profiles)
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved {len(all_profiles)} records to {OUTPUT_CSV}")
        
        # Save to JSON
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_profiles)} records to {OUTPUT_JSON}")
        
        # Print summary
        print("\n" + "-" * 60)
        print("Scraping Summary:")
        print(f"  Total traders scraped: {len(all_profiles)}")
        print(f"  Success rate: {len(all_profiles) / len(trader_ids) * 100:.1f}%")
        print(f"  Errors logged: Check {ERROR_LOG}")
    else:
        print("No profiles were successfully scraped!")
        logger.error("Scraping completed with zero successful results")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
