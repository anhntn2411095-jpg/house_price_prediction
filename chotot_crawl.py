import asyncio
import aiohttp
import csv
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


# Config

API_LISTING_URL = "https://gateway.chotot.com/v1/public/ad-listing"

HANOI_REGION = 12

APARTMENT_CATEGORY_ID = 1010

SALE_STATUS = "s,k"

PAGE_SIZE = 20

TARGET_TOTAL = 2000
MAX_PAGES = 120

CONCURRENCY = 2
REQUEST_TIMEOUT = 20
MIN_DELAY = 2.0
MAX_DELAY = 5.0
RETRIES = 3

FETCH_DETAIL = False

OUTPUT_CSV = "chotot_hanoi_apartment_sale_final.csv"
RAW_SAMPLE_JSON = "raw_ad_samples_debug.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Student academic project; "
        "Hanoi apartment price prediction; contact: your_email@example.com)"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://nha.chotot.com/",
    "Origin": "https://nha.chotot.com",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


USE_SUPABASE = True

SUPABASE_URL = ""
SUPABASE_KEY = ""
SUPABASE_TABLE = 'housing_chotot'

if USE_SUPABASE:
    if not SUPABASE_URL:
        raise ValueError("Missing SUPABASE_URL. Please set it as an environment variable.")

    if not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_KEY. Please set it as an environment variable.")


# Data model

@dataclass
class Listing:
    # Identifier
    id: int
    source: str
    source_url: Optional[str]

    # Target variable
    price_vnd: float
    price_billion: float

    # Basic property information
    category: Optional[str]
    is_rent: bool
    title: Optional[str]
    description: Optional[str]
    area_m2: Optional[float]
    price_per_m2: Optional[float]  

    # Location
    region_name: Optional[str]
    district: Optional[str]
    ward: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]

    # Structured property features
    legal_document: Optional[str]
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    floors: Optional[int]
    furnishing: Optional[str]
    property_status: Optional[str]
    seller_type: Optional[str]
    list_time: Optional[int]

    # Text-derived keyword features
    has_full_furniture: int
    has_balcony: int
    has_lake_view: int
    has_city_view: int
    has_red_book: int
    has_car_access: int
    has_luxury_keyword: int
    has_new_keyword: int
    has_corner_keyword: int
    has_near_center: int

    def to_row(self) -> dict:
        return asdict(self)


# Helper functions

def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return None

    return text


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

# Return the first non-empty value from a list of possible field names

def get_first(ad: dict, keys: list[str]) -> Any:
    for key in keys:
        value = ad.get(key)
        if value not in [None, "", [], {}]:
            return value

    return None

# Extract integer values from title/description.
def extract_int_from_text(text: Optional[str], patterns: list[str]) -> Optional[int]:
    if not text:
        return None

    lower_text = text.lower()

    for pattern in patterns:
        match = re.search(pattern, lower_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue

    return None


def keyword_flag(text: Optional[str], keywords: list[str]) -> int:
    if not text:
        return 0

    lower_text = text.lower()

    for keyword in keywords:
        if keyword.lower() in lower_text:
            return 1

    return 0

# Generic reference URL
def build_source_url(list_id: int) -> str:
    return f"https://www.chotot.com/{list_id}.htm"


def validate_hanoi(ad: dict) -> bool:
    region = ad.get("region")
    region_name = str(ad.get("region_name", "")).lower()

    if region is not None:
        try:
            return int(region) == HANOI_REGION
        except (TypeError, ValueError):
            return False

    return "hà nội" in region_name or "ha noi" in region_name


def normalize_legal_document(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None

    return str(value)

# Convert one raw listing JSON object into a clean Listing row
def parse_listing(ad: dict) -> Optional[Listing]:

    if not validate_hanoi(ad):
        return None

    list_id = get_first(ad, ["list_id", "ad_id", "id"])
    list_id = safe_int(list_id)

    if list_id is None:
        return None

    price_vnd = safe_float(get_first(ad, ["price", "price_vnd"]))

    if price_vnd is None or price_vnd <= 0:
        return None

    area_m2 = safe_float(get_first(ad, [
        "size",
        "area",
        "area_m2",
        "living_area",
        "square",
    ]))

    title = clean_text(get_first(ad, [
        "subject",
        "title",
        "ad_subject",
    ]))

    description = clean_text(get_first(ad, [
        "body",
        "description",
        "ad_body",
    ]))

    combined_text = " ".join(
        part for part in [title, description] if part
    )

    # Bedrooms
    bedrooms = safe_int(get_first(ad, [
        "rooms",
        "bedrooms",
        "bedroom",
        "number_of_bedrooms",
        "rooms_nb",
        "room",
    ]))

    if bedrooms is None:
        bedrooms = extract_int_from_text(
            combined_text,
            [
                r"(\d+)\s*phòng ngủ",
                r"(\d+)\s*pn\b",
                r"(\d+)\s*bedroom",
            ],
        )

    bathrooms = safe_int(get_first(ad, [
        "toilets",
        "bathrooms",
        "bathroom",
        "wc",
        "number_of_bathrooms",
        "toilet",
    ]))

    if bathrooms is None:
        bathrooms = extract_int_from_text(
            combined_text,
            [
                r"(\d+)\s*phòng tắm",
                r"(\d+)\s*wc\b",
                r"(\d+)\s*toilet",
                r"(\d+)\s*bathroom",
            ],
        )

    floors = safe_int(get_first(ad, [
        "floors",
        "floor",
        "number_of_floors",
        "floors_nb",
    ]))

    if floors is None:
        floors = extract_int_from_text(
            combined_text,
            [
                r"(\d+)\s*tầng",
                r"(\d+)\s*floor",
            ],
        )

    legal_document = normalize_legal_document(get_first(ad, [
        "property_legal_document",
        "legal_document",
        "legal_document_name",
        "property_legal_document_name",
    ]))

    furnishing = clean_text(get_first(ad, [
        "furnishing_sell",
        "furnishing",
        "furniture_status",
        "furnishing_name",
    ]))

    property_status = clean_text(get_first(ad, [
        "property_status",
        "property_status_name",
        "condition",
        "condition_name",
    ]))

    seller_type = clean_text(get_first(ad, [
        "seller_type",
        "account_type",
        "seller_type_name",
    ]))

    list_time = safe_int(get_first(ad, [
        "list_time",
        "date",
        "created_at",
        "timestamp",
    ]))

    latitude = safe_float(get_first(ad, ["latitude", "lat"]))
    longitude = safe_float(get_first(ad, ["longitude", "lng", "lon"]))

    price_per_m2 = None
    if area_m2 is not None and area_m2 > 0:
        price_per_m2 = price_vnd / area_m2

    price_billion = price_vnd / 1_000_000_000

    return Listing(
        id=list_id,
        source="chotot_gateway",
        source_url=build_source_url(list_id),

        price_vnd=price_vnd,
        price_billion=price_billion,

        category=clean_text(get_first(ad, ["category_name", "category"])),
        is_rent=False,
        title=title,
        description=description,
        area_m2=area_m2,
        price_per_m2=price_per_m2,

        region_name=clean_text(ad.get("region_name")),
        district=clean_text(get_first(ad, ["area_name", "district", "district_name"])),
        ward=clean_text(get_first(ad, ["ward_name", "ward"])),

        latitude=latitude,
        longitude=longitude,

        legal_document=legal_document,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        floors=floors,
        furnishing=furnishing,
        property_status=property_status,
        seller_type=seller_type,
        list_time=list_time,

        has_full_furniture=keyword_flag(
            combined_text,
            [
                "full nội thất",
                "đầy đủ nội thất",
                "nội thất cao cấp",
                "full nt",
                "full đồ",
            ],
        ),
        has_balcony=keyword_flag(
            combined_text,
            [
                "ban công",
                "balcony",
                "logia",
            ],
        ),
        has_lake_view=keyword_flag(
            combined_text,
            [
                "view hồ",
                "nhìn hồ",
                "hồ tây",
                "hồ gươm",
                "view sông",
            ],
        ),
        has_city_view=keyword_flag(
            combined_text,
            [
                "view thành phố",
                "city view",
                "view đẹp",
                "view thoáng",
            ],
        ),
        has_red_book=keyword_flag(
            combined_text,
            [
                "sổ đỏ",
                "sổ hồng",
                "pháp lý đầy đủ",
                "pháp lý rõ ràng",
                "có sổ",
            ],
        ),
        has_car_access=keyword_flag(
            combined_text,
            [
                "ô tô",
                "oto",
                "gara",
                "đỗ xe",
                "hầm để xe",
                "chỗ để xe",
            ],
        ),
        has_luxury_keyword=keyword_flag(
            combined_text,
            [
                "cao cấp",
                "luxury",
                "sang trọng",
                "đẳng cấp",
                "penthouse",
                "duplex",
                "residence",
            ],
        ),
        has_new_keyword=keyword_flag(
            combined_text,
            [
                "mới",
                "nhà mới",
                "vừa bàn giao",
                "mới nhận nhà",
                "chưa ở",
            ],
        ),
        has_corner_keyword=keyword_flag(
            combined_text,
            [
                "căn góc",
                "góc",
                "2 mặt thoáng",
                "hai mặt thoáng",
            ],
        ),
        has_near_center=keyword_flag(
            combined_text,
            [
                "trung tâm",
                "gần phố",
                "gần trường",
                "gần bệnh viện",
                "gần metro",
                "gần hồ",
                "gần công viên",
            ],
        ),
    )


# Crawl state
class CrawlState:
    def __init__(self, target: int):
        self.target = target
        self.seen_ids: set[int] = set()
        self.total = 0
        self.lock = asyncio.Lock()
        self.done = False

    async def register(self, listings: list[Listing]) -> list[Listing]:
        async with self.lock:
            if self.done:
                return []

            new_listings = []

            for listing in listings:
                if listing.id not in self.seen_ids:
                    self.seen_ids.add(listing.id)
                    new_listings.append(listing)

            self.total += len(new_listings)

            if self.total >= self.target:
                self.done = True

            return new_listings


class OptionalSupabaseWriter:
    def __init__(self):
        self.enabled = False
        self.client = None
        self.written = 0
        self.lock = asyncio.Lock()

        if USE_SUPABASE:
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise RuntimeError(
                    "USE_SUPABASE=True but SUPABASE_URL or SUPABASE_KEY is missing."
                )

            try:
                from supabase import create_client
            except ImportError as error:
                raise RuntimeError(
                    "Supabase package is not installed. Run: pip install supabase"
                ) from error

            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self.enabled = True

    async def upsert(self, rows: list[dict]):
        if not self.enabled or not rows:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._upsert_sync, rows)

        async with self.lock:
            self.written += len(rows)

        log.info(f"Supabase rows written so far: {self.written}")

    def _upsert_sync(self, rows: list[dict]):
        self.client.table(SUPABASE_TABLE).upsert(rows, on_conflict="id").execute()


# Network functions

async def polite_sleep():
    await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
    retries: int = RETRIES,
) -> Optional[dict]:
    for attempt in range(retries):
        try:
            async with session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as response:

                if response.status in [401, 403, 404, 429]:
                    text = await response.text()
                    log.warning(
                        f"Stop-like status {response.status}. "
                        f"URL={response.url}. Response preview={text[:120]}"
                    )
                    return None

                # Retry only server-side errors
                if response.status >= 500:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="Server error",
                        headers=response.headers,
                    )

                response.raise_for_status()
                return await response.json(content_type=None)

        except Exception as error:
            wait = (2 ** attempt) + random.uniform(0, 1)
            log.warning(f"Retry {attempt + 1}/{retries} in {wait:.1f}s | {error}")
            await asyncio.sleep(wait)

    return None


async def fetch_listing_page(
    session: aiohttp.ClientSession,
    offset: int,
) -> list[dict]:
    params = {
        "cg": APARTMENT_CATEGORY_ID,
        "o": offset,
        "limit": PAGE_SIZE,
        "st": SALE_STATUS,
        "region": HANOI_REGION,
        "key_param_included": "true",
    }

    data = await fetch_json(session, API_LISTING_URL, params=params)

    if not data:
        return []

    ads = data.get("ads", [])

    if not isinstance(ads, list):
        return []

    return ads

def save_csv(rows: list[dict], output_file: str):
    if not rows:
        log.warning("No rows to save.")
        return

    fieldnames = list(rows[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Saved {len(rows)} rows to {output_file}")


def save_raw_samples(raw_samples: list[dict], output_file: str):
    if not raw_samples:
        return

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(raw_samples, file, ensure_ascii=False, indent=2)

    log.info(f"Saved raw debug samples to {output_file}")


# Data quality report
def print_quality_report(rows: list[dict]):
    if not rows:
        return

    try:
        import pandas as pd
    except ImportError:
        log.info("pandas not installed, skipping quality report.")
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nMissing values (%):")
    missing = df.isna().mean().sort_values(ascending=False) * 100
    print(missing.round(2))

    print("\nPrice summary, billion VND:")
    print(df["price_billion"].describe())

    print("\nArea summary, m2:")
    print(df["area_m2"].describe())

    if "price_per_m2" in df.columns:
        df["price_per_m2_million"] = df["price_per_m2"] / 1_000_000
        print("\nPrice per m2 summary, million VND/m2:")
        print(df["price_per_m2_million"].describe())

    print("\nDistrict counts:")
    print(df["district"].value_counts(dropna=False).head(20))

    useful_cols = [
        "bedrooms",
        "bathrooms",
        "floors",
        "furnishing",
        "legal_document",
        "has_full_furniture",
        "has_balcony",
        "has_lake_view",
        "has_luxury_keyword",
        "has_red_book",
        "has_car_access",
    ]

    print("\nUseful feature availability:")
    for col in useful_cols:
        if col in df.columns:
            non_missing = df[col].notna().mean() * 100
            print(f"{col}: {non_missing:.2f}% non-missing")

    print("\nPreview:")
    print(df.head())


# Main crawler

async def crawl_apartment_sale():
    start_time = time.time()

    state = CrawlState(target=TARGET_TOTAL)
    writer = OptionalSupabaseWriter()

    all_rows: list[dict] = []
    raw_samples: list[dict] = []

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        offset = 0
        page = 0

        while page < MAX_PAGES and not state.done:
            log.info(f"Fetching listing page {page + 1}, offset={offset}")

            ads = await fetch_listing_page(session, offset=offset)
            await polite_sleep()

            if not ads:
                log.info("No more ads returned. Stopping.")
                break

            if len(raw_samples) < 5:
                remaining = 5 - len(raw_samples)
                raw_samples.extend(ads[:remaining])

            parsed_listings: list[Listing] = []

            for ad in ads:
                listing = parse_listing(ad)

                if listing:
                    parsed_listings.append(listing)

            new_listings = await state.register(parsed_listings)

            if new_listings:
                rows = [listing.to_row() for listing in new_listings]
                all_rows.extend(rows)
                await writer.upsert(rows)

            log.info(
                f"Page={page + 1:<3} raw={len(ads):<3} "
                f"parsed={len(parsed_listings):<3} "
                f"new={len(new_listings):<3} total={state.total}"
            )

            offset += PAGE_SIZE
            page += 1

            if len(ads) < PAGE_SIZE:
                log.info("Last page reached because returned ads < PAGE_SIZE.")
                break

    save_csv(all_rows, OUTPUT_CSV)
    save_raw_samples(raw_samples, RAW_SAMPLE_JSON)
    print_quality_report(all_rows)

    elapsed = time.time() - start_time

    log.info("=" * 70)
    log.info(f"Finished. Total rows collected: {len(all_rows)}")
    log.info(f"Time elapsed: {elapsed:.0f} seconds")
    log.info(f"Raw debug sample: {RAW_SAMPLE_JSON}")

    if USE_SUPABASE:
        log.info(f"Supabase rows written: {writer.written}")

if __name__ == "__main__":
    asyncio.run(crawl_apartment_sale())
