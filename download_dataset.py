"""
Download Hampi monument images from Wikimedia Commons.

Creates: data/test_images/<MonumentName>/*.jpg

Uses Special:FilePath redirect to avoid CDN rate limits.
"""

import os
import json
import time
import subprocess
import urllib.request
import urllib.parse
from typing import Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "HampiMonumentIdentifier/1.0 "
    "(academic project; Python/3.x)"
)
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

IMAGES_PER_MONUMENT = 5
DELAY_API = 2
DELAY_DOWNLOAD = 5

CATEGORY_MAP = {
    "Virupaksha Temple": "Virupaksha_Temple,_Hampi",
    "Vittala Temple": "Vittala_Temple,_Hampi",
    "Lotus Mahal": "Lotus_Mahal,_Hampi",
    "Elephant Stables": "Elephant_Stables,_Hampi",
    "Hazara Rama Temple": "Hazara_Rama_Temple,_Hampi",
    "Achyutaraya Temple": "Achyutaraya_Temple,_Hampi",
    "Matanga Hill": "Matanga_Hill,_Hampi",
    "Underground Shiva Temple": "Prasanna_Virupaksha_Temple,_Hampi",
    "Queen's Bath": "Queen%27s_Bath,_Hampi",
    "Hampi Bazaar": "Hampi_Bazaar",
}


def api_request(params):
    """Wikimedia API request with retries."""
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if "429" in str(e):
                time.sleep((attempt + 1) * 10)
            else:
                raise
    raise Exception("API failed after retries")


def get_file_titles(category, search_query, limit=15):
    """Get file titles from category + search."""
    titles = []

    # Category members
    cat_name = urllib.parse.unquote(category)
    try:
        data = api_request({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat_name}",
            "cmtype": "file",
            "cmlimit": str(limit),
        })
        for m in data.get("query", {}).get("categorymembers", []):
            t = m.get("title", "")
            if any(t.lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp")):
                titles.append(t)
    except Exception as e:
        print(f"   Category failed: {e}")

    time.sleep(DELAY_API)

    # Search fallback
    if len(titles) < limit:
        try:
            data = api_request({
                "action": "query",
                "list": "search",
                "srsearch": f"filetype:bitmap {search_query}",
                "srnamespace": "6",
                "srlimit": str(limit),
            })
            existing = set(titles)
            for r in data.get("query", {}).get("search", []):
                t = r.get("title", "")
                if t not in existing and any(t.lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp")):
                    titles.append(t)
        except Exception as e:
            print(f"   Search failed: {e}")

    return titles


def download_via_filepath(file_title, dest_path, width=800):
    """
    Download using Special:FilePath which bypasses CDN rate limits.
    URL format: https://commons.wikimedia.org/wiki/Special:FilePath/<filename>?width=800
    """
    # Extract just the filename (remove "File:" prefix)
    filename = file_title.replace("File:", "").strip()
    encoded = urllib.parse.quote(filename)
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width={width}"

    try:
        result = subprocess.run(
            [
                "curl", "-s", "-L",
                "-o", dest_path,
                "-w", "%{http_code}",
                "-H", f"User-Agent: {BROWSER_UA}",
                "-H", "Referer: https://commons.wikimedia.org/",
                "--max-time", "60",
                url,
            ],
            capture_output=True, text=True, timeout=90,
        )
        status = result.stdout.strip()

        if status == "200" and os.path.exists(dest_path) and os.path.getsize(dest_path) > 2000:
            return True
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if status != "200":
                print(f"      HTTP {status}")
            return False
    except Exception as e:
        print(f"      Error: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def sanitize_filename(title):
    name = title.replace("File:", "").strip()
    for ch in ["?", "#", "%", "&", "\\", "/"]:
        name = name.replace(ch, "_")
    if len(name) > 80:
        ext = os.path.splitext(name)[1] or ".jpg"
        name = name[:75] + ext
    return name


def main():
    print("=" * 60)
    print("  Hampi Monument Dataset Downloader")
    print("  Method: Special:FilePath redirect")
    print(f"  Target: {TEST_IMAGES_DIR}")
    print(f"  Images per monument: {IMAGES_PER_MONUMENT}")
    print("=" * 60)

    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    total = 0

    for monument_name, wiki_category in CATEGORY_MAP.items():
        print(f"\n📍 {monument_name}")

        monument_dir = os.path.join(TEST_IMAGES_DIR, monument_name)
        os.makedirs(monument_dir, exist_ok=True)

        existing = [f for f in os.listdir(monument_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if len(existing) >= IMAGES_PER_MONUMENT:
            print(f"   ✓ Already has {len(existing)} images")
            total += len(existing)
            continue

        needed = IMAGES_PER_MONUMENT - len(existing)
        print(f"   Need {needed} more (have {len(existing)})")

        # Get file titles from API
        titles = get_file_titles(wiki_category, f"{monument_name} Hampi")
        print(f"   Found {len(titles)} candidates")
        time.sleep(DELAY_API)

        downloaded = len(existing)
        for title in titles:
            if downloaded >= IMAGES_PER_MONUMENT:
                break

            filename = sanitize_filename(title)
            dest = os.path.join(monument_dir, filename)
            if os.path.exists(dest):
                downloaded += 1
                continue

            print(f"   ↓ {filename[:55]}...")
            if download_via_filepath(title, dest):
                downloaded += 1
                print(f"     ✓ OK ({downloaded}/{IMAGES_PER_MONUMENT})")
            time.sleep(DELAY_DOWNLOAD)

        total += downloaded
        print(f"   Result: {downloaded}/{IMAGES_PER_MONUMENT}")

    # Create virupaksha.jpg for single-image test cell
    vdir = os.path.join(TEST_IMAGES_DIR, "Virupaksha Temple")
    if os.path.isdir(vdir):
        imgs = [f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if imgs:
            import shutil
            dst = os.path.join(TEST_IMAGES_DIR, "virupaksha.jpg")
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(vdir, imgs[0]), dst)
                print(f"\n✓ Created virupaksha.jpg")

    print(f"\n{'=' * 60}")
    print(f"  Total: {total} images")
    print("\nDataset structure:")
    for m in CATEGORY_MAP:
        d = os.path.join(TEST_IMAGES_DIR, m)
        if os.path.isdir(d):
            c = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
            s = "✓" if c >= IMAGES_PER_MONUMENT else "⚠"
            print(f"  {s} {m}: {c}/{IMAGES_PER_MONUMENT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
