# Places-Attribute-Conflation

## Project Overview

**Author:** James Conrad Manlangit

When creating a universal database of locations and businesses, Overture Maps gathers their information from multiple different sources. This introduces the issue of conflicting information from different sources for key location attributes for the same place, such as discrepancies between business names, addresses and categories. This project aims to address this issue by implementing a rule-based conflation algorithm that determines which attributes are more reliable, consistent, and "more correct."

## Description

### Labeling Approach:

The algorithm assigns a score between 0.0-1.0 for each major attribute (name, address, categories, websites, phones, email, social, brand) across both sources. The evaluator scores the attributes through 8 main functions:

- **evaluate_name_quality()**
  - Rewards: Simple names (1-5 words), proper Title Case capitalization, reasonable length (5-100 chars)
  - Penalizes: Trailing numbers (e.g., "Store 6285"), business suffixes (LLC, Inc, Corp)

- **evaluate_address_quality()**
  - Rewards: Complete street addresses (number + street name), abbreviated street types (St, Ave, Dr), abbreviated directions (N, S, E, W), 5-digit ZIP codes
  - Penalizes: Incomplete addresses (building names only), unabbreviated forms, ZIP+4 format

- **evaluate_categories_quality()**
  - Rewards: Having multiple categories (up to 5), longer/more specific category names (>10 chars)

- **evaluate_website_quality()**
  - Rewards: Valid URLs, HTTPS protocol, multiple website entries

- **evaluate_phone_quality()**
  - Rewards: Valid phone numbers, country codes (starting with +), multiple phone entries

- **evaluate_email_quality()**
  - Rewards: Valid email format, multiple email addresses

- **evaluate_social_quality()**
  - Rewards: Having social media links, multiple social profiles

- **evaluate_brand_quality()**
  - Rewards: Valid brand strings (>2 chars)
  - Returns 0 for: Empty/null brands or empty brand dictionaries like {'names': {}}

For each attribute, the algorithm compares the calculated scores of the two sources' information, and labels the attribute as follows:

- **1** = Place A (base_) attribute is better
- **0** = Place B (non_base) attribute is better
- **2** = Tie
  - Defaults to base_ attribute when creating the golden dataset.

### Validation Approach:

Because the Yelp Dataset's attributes do not include all attributes from the Sample B Project Dataset, this project only validates address and name labels. Address works as follows:

- For each record and their name/address attributes in the Project B dataset, find a match to a Yelp Dataset record. Calculate the name and address string similarity (fuzz.token_set_ratio) for both sources.
  - **Consideration:** This is quite expensive, since the time complexity of fuzzy matching in this validation process is **O(N × M × K)** where:
    - N = number of records in labels_df (~2000)
    - M = number of Yelp businesses
    - K = complexity of each fuzzy string comparison

- Filter out records with a combined fuzz.token_set_ratio (Between highest place and address fuzz.token_set_ratios) of 0.71
  - Ensures that records in the validation set match closely with a Yelp Dataset record
  - Reduces validation set to 36 records

- Compare labels to fuzzy similarity scores

## Input Files:

- **project_b_samples_2k.parquet** - Project B sample dataset of 2000 records of businesses with key attribute information from two different sources.
- **yelp_academic_dataset_business.json** - Open Yelp dataset of 150,000 businesses, used for validating labels


## Output Files:

- **output_data/algorithm_labels.csv** - Full Project B Sample dataset with label results from rule-based conflation algorithm, along with scores for each source's attributes calculated by algorithm.
- **output_data/golden_dataset.csv** - Full Project B Sample Dataset with only the attributes selected by the algorithm.

### Validation (v3 is most accurate and up-to-date version → 77.8% name label accuracy, 80.6% address label accuracy):

- **validation/v3/v3_validation_results.csv** - 36 Project B Sample Dataset records with at least a combined 0.71 fuzz.token_set_ratio (Similarity score) to a record in the Yelp Dataset
  - Only records that closely match with a record in the Yelp Dataset are chosen for validation

- **validation/v3/v3_address_disagreements.csv** - Dataset indicating where the algorithm's address labels disagree with labels determined by matching with the Yelp Dataset.

- **validation/v3/v3_name_disagreements.csv** - Dataset indicating where the algorithm's name labels disagree with labels determined by matching with the Yelp Dataset.

- **validation/v3/v3_validation_confusion_matrices.png** - Visualization depicting how many algorithm labels agree/disagree.


## Getting Started

Follow these steps in your terminal:

**1. Create and start a Python virtual environment**
```sh
python3 -m venv env
source env/bin/activate
```

**2. Install requirements**
```sh
pip install -r requirements.txt
```

**3. Run the dataset processing script**
```sh
python scripts/process_dataset.py
```

**4. Run the rule-based algorithm notebook**
Open and run all cells in:
```
scripts/rule_based_algorithm.ipynb
```

**5. Run the evaluation notebook**
Open and run all cells in:
```
scripts/evaluate.ipynb
```

**6. View results**
Results will be available in the `validation/v3` folder.


## Progress and OKRs

Full OKRs: [Up-To-Date OKRs](https://docs.google.com/document/d/1bEj2Y5F-YPrAL0L3Q_gu1b6vKbR_uUJiXCk8CZ3b_RI/edit?tab=t.0)

