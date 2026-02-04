"""Simple data loading script for Yelp restaurant reviews in Philadelphia."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


if __name__ == "__main__":
    # Step 0: Set file paths and fixed settings
    business_path = Path("data/raw/yelp_academic_dataset_business.json")
    review_path = Path("data/raw/yelp_academic_dataset_review.json")
    output_path = Path("data/processed/merged_reviews.csv")
    max_reviews = 100_000
    random_seed = 42

    # Step 1: Read business data (line-delimited JSON)
    # Each line is one JSON object representing a business.
    businesses = []
    with business_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            businesses.append(json.loads(line))

    # Step 2: Filter to Philadelphia restaurants
    # Conditions:
    # - city == "Philadelphia" (case-insensitive)
    # - categories contains "Restaurants"
    filtered_businesses = []
    for business in businesses:
        city = str(business.get("city", "")).strip().lower()
        categories = str(business.get("categories", "") or "")
        if city == "philadelphia" and "Restaurants" in categories:
            filtered_businesses.append(business)

    # Step 3: Build dictionary: business_id -> business_name
    business_id_to_name = {}
    for business in filtered_businesses:
        business_id = business.get("business_id")
        business_name = business.get("name", "")
        if business_id:
            business_id_to_name[business_id] = business_name

    # Step 4 + Step 5 + Step 6:
    # Stream through reviews, keep only matching businesses, and downsample
    # to at most 100,000 rows using reservoir sampling with a fixed seed.
    random_generator = random.Random(random_seed)
    sampled_rows = []
    seen_matching_reviews = 0

    with review_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            review = json.loads(line)
            business_id = review.get("business_id")

            # Keep only reviews for filtered Philadelphia restaurant businesses.
            if business_id not in business_id_to_name:
                continue

            seen_matching_reviews += 1
            row = {
                "review_id": review.get("review_id", ""),
                "business_id": business_id,
                "business_name": business_id_to_name[business_id],
                "stars": review.get("stars", ""),
                "date": review.get("date", ""),
                "text": review.get("text", ""),
            }

            # Reservoir sampling logic:
            # - Fill the reservoir first.
            # - Then randomly replace existing rows with decreasing probability.
            if len(sampled_rows) < max_reviews:
                sampled_rows.append(row)
            else:
                replacement_index = random_generator.randint(1, seen_matching_reviews)
                if replacement_index <= max_reviews:
                    sampled_rows[replacement_index - 1] = row

    # Step 7: Save merged result to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "review_id",
                "business_id",
                "business_name",
                "stars",
                "date",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Filtered businesses: {len(business_id_to_name):,}")
    print(f"Matching reviews seen: {seen_matching_reviews:,}")
    print(f"Rows saved: {len(sampled_rows):,}")
    print(f"Output written to: {output_path}")
