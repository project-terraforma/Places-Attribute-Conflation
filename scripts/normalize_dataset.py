import pandas as pd 
import json
import ast
import re
import os

# Import sample data
sample_df = pd.read_parquet('../input_data/project_b_samples_2k.parquet')
#sample_df

# Import Yelp dataset (JSON Lines format)
yelp_df = pd.read_json('../Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json', lines=True)
#yelp_df.head()

# Drop unneeded Yelp Columns
unimportant = ["latitude", "longitude", "stars", "review_count", "is_open", "attributes", "hours"]
yelp_df.drop(columns=unimportant, inplace=True)


# Assigns data types to attributes
def parse_category(x):
    if isinstance(x, (list, dict)) or pd.isna(x): return x
    try: return ast.literal_eval(x)
    except: return None

for c in ["sources", "names", "addresses", "categories", "websites", 
          "brand", "emails", "socials", "base_names","base_addresses",
          "base_categories", "base_websites", "base_brand", "base_emails", "base_socials"]:
    if c in sample_df: sample_df[c] = sample_df[c].apply(parse_category)


# Extract country information from address dictionaries
def extract_country(addr_list):
    """Extract country from address list (non-base attributes)"""
    if isinstance(addr_list, list) and addr_list:
        country = addr_list[0].get("country")
        if country:
            return country.strip().upper()
    return None

def extract_base_country(base_addr_list):
    """Extract country from base address list (base_ attributes)"""
    if isinstance(base_addr_list, list) and base_addr_list:
        country = base_addr_list[0].get("country")
        if country:
            return country.strip().upper()
    return None

# Create country columns
sample_df["country"] = sample_df["addresses"].apply(extract_country)
sample_df["base_country"] = sample_df["base_addresses"].apply(extract_base_country)

# Track conflicting countries between base and non-base
conflicting_countries = ((sample_df["country"] != sample_df["base_country"]) & 
                        (sample_df["country"].notna()) & 
                        (sample_df["base_country"].notna())).sum()

# Show rows with conflicting countries
conflicting_mask = ((sample_df["country"] != sample_df["base_country"]) & 
                   (sample_df["country"].notna()) & 
                   (sample_df["base_country"].notna()))

conflicting_rows = sample_df[conflicting_mask]

# Filter to keep only rows where BOTH country and base_country are "US" or "CA" (Canada)
print(f"Original dataset size: {len(sample_df)} rows")

# Create filter condition for US and Canada
allowed_countries = ["US", "CA"]
us_ca_filter = (
    (sample_df["country"].isin(allowed_countries)) & 
    (sample_df["base_country"].isin(allowed_countries))
)

# Apply filter - keep the original index to preserve sample_idx
sample_df_eval = sample_df[us_ca_filter].copy()
# Preserve the original DataFrame index for sample_idx tracking
sample_df_eval['sample_idx'] = sample_df_eval.index

print(f"Filtered dataset size (US/CA only): {len(sample_df_eval)} rows")
print(f"Removed {len(sample_df) - len(sample_df_eval)} rows")
print(f"Kept {len(sample_df_eval) / len(sample_df) * 100:.2f}% of original data")

# Verify the filter worked
print(f"\nCountry distribution in filtered data:")
print(f"Country values:\n{sample_df_eval['country'].value_counts()}")
print(f"\nBase country values:\n{sample_df_eval['base_country'].value_counts()}")


# Extract and normalize address components from sample_df

def extract_address_component(addr_list, component):
    """Extract a specific component from address list"""
    if isinstance(addr_list, list) and addr_list:
        addr_dict = addr_list[0]
        if isinstance(addr_dict, dict):
            return addr_dict.get(component, '')
    return ''

def normalize_street_address(address):
    """Normalize street address for comparison"""
    if not address:
        return ''
    # Convert to lowercase and strip whitespace
    normalized = str(address).lower().strip()
    return normalized

def normalize_city(city):
    """Normalize city name for comparison"""
    if not city:
        return ''
    # Convert to lowercase and strip whitespace
    normalized = str(city).lower().strip()
    return normalized

def normalize_state(state):
    """Normalize state for comparison"""
    if not state:
        return ''
    # Convert to uppercase and strip whitespace
    normalized = str(state).upper().strip()
    return normalized

def normalize_postcode(postcode):
    """Normalize postcode - handle both US ZIP codes and Canadian postal codes"""
    if not postcode:
        return ''
    
    # Convert to string and strip whitespace
    normalized = str(postcode).strip().upper()
    
    # Check if it looks like a Canadian postal code (A1A 1A1 or A1A1A1)
    # Canadian format: Letter-Digit-Letter space Digit-Letter-Digit
    if len(normalized) >= 6 and normalized[0].isalpha():
        # Remove any spaces first
        no_space = normalized.replace(' ', '')
        
        # Check if it matches Canadian pattern (6 characters, alternating letter-digit-letter-digit-letter-digit)
        if (len(no_space) == 6 and 
            no_space[0].isalpha() and no_space[1].isdigit() and no_space[2].isalpha() and
            no_space[3].isdigit() and no_space[4].isalpha() and no_space[5].isdigit()):
            # Return in standard Canadian format with space: A1A 1A1
            return f"{no_space[:3]} {no_space[3:6]}"
    
    # Otherwise, treat as US ZIP code
    # Remove ZIP+4 extension (keep only first 5 digits)
    if '-' in normalized:
        normalized = normalized.split('-')[0]
    
    # Ensure it's exactly 5 digits if it looks like a ZIP code
    if normalized.isdigit() and len(normalized) == 5:
        return normalized
    elif normalized.isdigit() and len(normalized) < 5:
        # Pad with leading zeros if needed
        return normalized.zfill(5)
    else:
        return normalized

def extract_primary_name(name_data):
    """Extract primary name from name structure"""
    if isinstance(name_data, dict):
        primary = name_data.get('primary', '')
        if primary:
            return str(primary)
    elif isinstance(name_data, str):
        return str(name_data)
    return ''

def normalize_name(name):
    """Normalize business name for comparison"""
    
    if not name:
        return ''
    
    # Convert to string and lowercase
    normalized = str(name).lower().strip()
    
    # Remove common business suffixes and prefixes
    normalized = re.sub(r'\b(llc|inc|corp|corporation|co|ltd|limited|the)\b', '', normalized)
    
    # Remove special characters except spaces, hyphens, apostrophes, and ampersands
    normalized = re.sub(r"[^a-z0-9\s&\-']", ' ', normalized)
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

# Define helper functions for normalization
def create_address_dict(street, city, state, postcode):
    """Create a consolidated address dictionary from normalized components"""
    return {
        'address': street if street else '',
        'city': city if city else '',
        'state': state if state else '',
        'postal_code': postcode if postcode else ''
    }

def extract_and_normalize_categories(category_data):
    """Extract and normalize categories from sample_df format"""
    if pd.isna(category_data) or category_data is None:
        return []
    
    normalized = []
    
    # Handle dictionary format with 'primary' and 'alternate' keys
    if isinstance(category_data, dict):
        # Add primary category
        if 'primary' in category_data and category_data['primary']:
            primary = str(category_data['primary']).strip().lower().replace(' ', '_')
            if primary:
                normalized.append(primary)
        
        # Add alternate categories
        if 'alternate' in category_data and isinstance(category_data['alternate'], list):
            for cat in category_data['alternate']:
                if cat and isinstance(cat, str):
                    alt = cat.strip().lower().replace(' ', '_')
                    if alt and alt not in normalized:
                        normalized.append(alt)
        
        return normalized
    
    # Handle list format
    if isinstance(category_data, list):
        for cat in category_data:
            if isinstance(cat, str) and cat.strip():
                normalized.append(cat.strip().lower().replace(' ', '_'))
        return normalized
    
    # Handle string format
    if isinstance(category_data, str):
        return [cat.strip().lower().replace(' ', '_') for cat in category_data.split(',') if cat.strip()]
    
    return []

# Function to normalize a dataframe
def normalize_sample_dataframe(df, preserve_index=False):
    """Apply all normalization steps to a sample dataframe"""
    df = df.copy()
    
    # Store original index if needed for subset tracking
    if preserve_index and 'sample_idx' not in df.columns:
        df['sample_idx'] = df.index
    
    # Extract and normalize names
    print("\nExtracting and normalizing names...")
    df['name_primary'] = df['names'].apply(extract_primary_name)
    df['base_name_primary'] = df['base_names'].apply(extract_primary_name)
    
    df['name_norm'] = df['name_primary'].apply(normalize_name)
    df['base_name_norm'] = df['base_name_primary'].apply(normalize_name)
    
    # Extract address components for non-base addresses
    print("Extracting address components from non-base addresses...")
    df['address_street'] = df['addresses'].apply(lambda x: extract_address_component(x, 'freeform'))
    df['address_city'] = df['addresses'].apply(lambda x: extract_address_component(x, 'locality'))
    df['address_state'] = df['addresses'].apply(lambda x: extract_address_component(x, 'region'))
    df['address_postcode'] = df['addresses'].apply(lambda x: extract_address_component(x, 'postcode'))
    
    # Extract address components for base addresses
    print("Extracting address components from base addresses...")
    df['base_address_street'] = df['base_addresses'].apply(lambda x: extract_address_component(x, 'freeform'))
    df['base_address_city'] = df['base_addresses'].apply(lambda x: extract_address_component(x, 'locality'))
    df['base_address_state'] = df['base_addresses'].apply(lambda x: extract_address_component(x, 'region'))
    df['base_address_postcode'] = df['base_addresses'].apply(lambda x: extract_address_component(x, 'postcode'))
    
    # Normalize all address components
    print("Normalizing address components...")
    
    # Normalize non-base address components
    df['address_street_norm'] = df['address_street'].apply(normalize_street_address)
    df['address_city_norm'] = df['address_city'].apply(normalize_city)
    df['address_state_norm'] = df['address_state'].apply(normalize_state)
    df['address_postcode_norm'] = df['address_postcode'].apply(normalize_postcode)
    
    # Normalize base address components
    df['base_address_street_norm'] = df['base_address_street'].apply(normalize_street_address)
    df['base_address_city_norm'] = df['base_address_city'].apply(normalize_city)
    df['base_address_state_norm'] = df['base_address_state'].apply(normalize_state)
    df['base_address_postcode_norm'] = df['base_address_postcode'].apply(normalize_postcode)
    
    # Create consolidated address dictionaries
    df['address_norm'] = df.apply(lambda row: create_address_dict(
        row['address_street_norm'],
        row['address_city_norm'], 
        row['address_state_norm'],
        row['address_postcode_norm']
    ), axis=1)
    
    df['base_address_norm'] = df.apply(lambda row: create_address_dict(
        row['base_address_street_norm'],
        row['base_address_city_norm'],
        row['base_address_state_norm'], 
        row['base_address_postcode_norm']
    ), axis=1)
    
    # Normalize categories
    df['categories_norm'] = df['categories'].apply(extract_and_normalize_categories)
    df['base_categories_norm'] = df['base_categories'].apply(extract_and_normalize_categories)
    
    print("Normalization complete!")
    return df

# Normalize both datasets
print("\n=== NORMALIZING FULL DATASET ===")
sample_df = normalize_sample_dataframe(sample_df, preserve_index=False)
# Add sample_idx to full dataset based on position
sample_df.insert(0, 'sample_idx', range(len(sample_df)))

print("\n=== NORMALIZING US/CA FILTERED DATASET ===")
# Note: sample_idx was already preserved from the filter operation
sample_df_eval = normalize_sample_dataframe(sample_df_eval, preserve_index=True)
# Move sample_idx to the first column
if 'sample_idx' in sample_df_eval.columns:
    cols = ['sample_idx'] + [col for col in sample_df_eval.columns if col != 'sample_idx']
    sample_df_eval = sample_df_eval[cols]

# Show examples
print("\n=== NAME NORMALIZATION EXAMPLES (from full dataset) ===")
for i in range(min(3, len(sample_df))):
    row = sample_df.iloc[i]
    print(f"\nExample {i+1}:")
    print(f"  Original names: {row['names']}")
    print(f"  Primary name: '{row['name_primary']}'")
    print(f"  Normalized name: '{row['name_norm']}'")
    print(f"  Original base_names: {row['base_names']}")
    print(f"  Primary base name: '{row['base_name_primary']}'")
    print(f"  Normalized base name: '{row['base_name_norm']}')")


# Normalize Yelp dataset address components for comparison
print("=== NORMALIZING YELP DATASET FOR COMPARISON ===")

# Normalize Yelp address components using the same functions
yelp_df['address_norm'] = yelp_df['address'].apply(normalize_street_address)
yelp_df['city_norm'] = yelp_df['city'].apply(normalize_city)
yelp_df['state_norm'] = yelp_df['state'].apply(normalize_state)
yelp_df['postal_code_norm'] = yelp_df['postal_code'].apply(normalize_postcode)

# Also normalize Yelp business names
yelp_df['name_norm'] = yelp_df['name'].apply(normalize_name)

# Normalize categories column - convert to list format
def normalize_categories(categories):
    """Normalize categories - convert to list format with lowercase and underscores"""
    if pd.isna(categories) or categories is None:
        return []
    if isinstance(categories, list):
        return [cat.strip().lower().replace(' ', '_') for cat in categories if cat and cat.strip()]
    if isinstance(categories, str):
        # Yelp categories are comma-separated strings
        return [cat.strip().lower().replace(' ', '_') for cat in categories.split(',') if cat.strip()]
    return []

yelp_df['categories_norm'] = yelp_df['categories'].apply(normalize_categories)

# Create consolidated address dictionary for Yelp data
yelp_df['address_dict_norm'] = yelp_df.apply(lambda row: create_address_dict(
    row['address_norm'],
    row['city_norm'],
    row['state_norm'],
    row['postal_code_norm']
), axis=1)

print("Yelp dataset normalization complete!")
print(f"Yelp dataset size: {len(yelp_df)} businesses")

print(f"\nDatasets ready!")
print(f"  Full sample size: {len(sample_df)} rows")
print(f"  US/CA eval sample size: {len(sample_df_eval)} rows")
print(f"  Yelp size: {len(yelp_df)} businesses")

# Export normalized dataframes to CSV

# Create output directory if it doesn't exist
output_dir = '../input_data'
os.makedirs(output_dir, exist_ok=True)

# Export full sample_df (all countries)
sample_output_path = os.path.join(output_dir, 'normalized_samples.csv')
sample_df.to_csv(sample_output_path, index=False)
print(f"\nExported normalized sample data (all countries) to: {sample_output_path}")
print(f"  Rows: {len(sample_df)}, Columns: {len(sample_df.columns)}")

# Export US/CA filtered sample_df for evaluation
eval_output_path = os.path.join(output_dir, 'normalized_eval_samples.csv')
sample_df_eval.to_csv(eval_output_path, index=False)
print(f"Exported normalized evaluation sample data (US/CA only) to: {eval_output_path}")
print(f"  Rows: {len(sample_df_eval)}, Columns: {len(sample_df_eval.columns)}")

# Export yelp_df
yelp_output_path = os.path.join(output_dir, 'normalized_yelp.csv')
yelp_df.to_csv(yelp_output_path, index=False)
print(f"Exported normalized Yelp data to: {yelp_output_path}")
print(f"  Rows: {len(yelp_df)}, Columns: {len(yelp_df.columns)}")

print("\nExport complete!")
