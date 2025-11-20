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

# Filter to keep only rows where BOTH country and base_country are "US"
print(f"Original dataset size: {len(sample_df)} rows")

# Create filter condition for US
allowed_countries = ["US"]
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

def extract_primary_name(name_data):
    """Extract primary name from name structure"""
    if isinstance(name_data, dict):
        primary = name_data.get('primary', '')
        if primary:
            return str(primary)
    elif isinstance(name_data, str):
        return str(name_data)
    return ''

# Define helper functions for extraction
def create_address_string(street, city, state, postcode):
    """Create a single concatenated address string from components"""
    # Build address parts, filtering out empty strings
    parts = []
    
    if street and str(street).strip():
        parts.append(str(street).strip())
    if city and str(city).strip():
        parts.append(str(city).strip())
    if state and str(state).strip():
        parts.append(str(state).strip())
    if postcode and str(postcode).strip():
        parts.append(str(postcode).strip())
    
    # Join with comma and space
    return ', '.join(parts) if parts else ''

def extract_categories(category_data):
    """Extract categories from sample_df format without normalization"""
    if pd.isna(category_data) or category_data is None:
        return []
    
    categories = []
    
    # Handle dictionary format with 'primary' and 'alternate' keys
    if isinstance(category_data, dict):
        # Add primary category
        if 'primary' in category_data and category_data['primary']:
            primary = str(category_data['primary']).strip()
            if primary:
                categories.append(primary)
        
        # Add alternate categories
        if 'alternate' in category_data and isinstance(category_data['alternate'], list):
            for cat in category_data['alternate']:
                if cat and isinstance(cat, str):
                    alt = cat.strip()
                    if alt and alt not in categories:
                        categories.append(alt)
        
        return categories
    
    # Handle list format
    if isinstance(category_data, list):
        for cat in category_data:
            if isinstance(cat, str) and cat.strip():
                categories.append(cat.strip())
        return categories
    
    # Handle string format
    if isinstance(category_data, str):
        return [cat.strip() for cat in category_data.split(',') if cat.strip()]
    
    return []

# Function to process a dataframe (extraction only, no normalization)
def normalize_sample_dataframe(df, preserve_index=False):
    """Apply all extraction steps to a sample dataframe (no normalization)"""
    df = df.copy()
    
    # Store original index if needed for subset tracking
    if preserve_index and 'sample_idx' not in df.columns:
        df['sample_idx'] = df.index
    
    # Extract primary names
    print("\nExtracting primary names...")
    df['name_primary'] = df['names'].apply(extract_primary_name)
    df['base_name_primary'] = df['base_names'].apply(extract_primary_name)
    
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
    
    # Create consolidated address strings
    df['address_string'] = df.apply(lambda row: create_address_string(
        row['address_street'],
        row['address_city'], 
        row['address_state'],
        row['address_postcode']
    ), axis=1)
    
    df['base_address_string'] = df.apply(lambda row: create_address_string(
        row['base_address_street'],
        row['base_address_city'],
        row['base_address_state'], 
        row['base_address_postcode']
    ), axis=1)
    
    # Extract categories
    df['categories_list'] = df['categories'].apply(extract_categories)
    df['base_categories_list'] = df['base_categories'].apply(extract_categories)
    
    print("Extraction complete!")
    return df

# Normalize both datasets
print("\n=== PROCESSING FULL DATASET ===")
sample_df = normalize_sample_dataframe(sample_df, preserve_index=False)
# Add sample_idx to full dataset based on position
sample_df.insert(0, 'sample_idx', range(len(sample_df)))

print("\n=== PROCESSING US/CA FILTERED DATASET ===")
# Note: sample_idx was already preserved from the filter operation
sample_df_eval = normalize_sample_dataframe(sample_df_eval, preserve_index=True)
# Move sample_idx to the first column
if 'sample_idx' in sample_df_eval.columns:
    cols = ['sample_idx'] + [col for col in sample_df_eval.columns if col != 'sample_idx']
    sample_df_eval = sample_df_eval[cols]

# Show examples
print("\n=== NAME EXTRACTION EXAMPLES (from full dataset) ===")
for i in range(min(3, len(sample_df))):
    row = sample_df.iloc[i]
    print(f"\nExample {i+1}:")
    print(f"  Original names: {row['names']}")
    print(f"  Primary name: '{row['name_primary']}'")
    print(f"  Original base_names: {row['base_names']}")
    print(f"  Primary base name: '{row['base_name_primary']}')")


# Process Yelp dataset for comparison
print("=== PROCESSING YELP DATASET FOR COMPARISON ===")

# Extract categories - convert to list format
def extract_yelp_categories(categories):
    """Extract categories - convert to list format"""
    if pd.isna(categories) or categories is None:
        return []
    if isinstance(categories, list):
        return [cat.strip() for cat in categories if cat and cat.strip()]
    if isinstance(categories, str):
        # Yelp categories are comma-separated strings
        return [cat.strip() for cat in categories.split(',') if cat.strip()]
    return []

yelp_df['categories_list'] = yelp_df['categories'].apply(extract_yelp_categories)

# Create consolidated address string for Yelp data
yelp_df['address_string'] = yelp_df.apply(lambda row: create_address_string(
    row['address'],
    row['city'],
    row['state'],
    row['postal_code']
), axis=1)

print("Yelp dataset processing complete!")
print(f"Yelp dataset size: {len(yelp_df)} businesses")

print(f"\nDatasets ready!")
print(f"  Full sample size: {len(sample_df)} rows")
print(f"  US/CA eval sample size: {len(sample_df_eval)} rows")
print(f"  Yelp size: {len(yelp_df)} businesses")

# Export processed dataframes to CSV

# Create output directory if it doesn't exist
output_dir = '../input_data'
os.makedirs(output_dir, exist_ok=True)

# Export full sample_df (all countries)
sample_output_path = os.path.join(output_dir, 'processed_samples.csv')
sample_df.to_csv(sample_output_path, index=False)
print(f"\nExported processed sample data (all countries) to: {sample_output_path}")
print(f"  Rows: {len(sample_df)}, Columns: {len(sample_df.columns)}")

# Export US/CA filtered sample_df for evaluation
eval_output_path = os.path.join(output_dir, 'processed_eval_samples.csv')
sample_df_eval.to_csv(eval_output_path, index=False)
print(f"Exported processed evaluation sample data (US/CA only) to: {eval_output_path}")
print(f"  Rows: {len(sample_df_eval)}, Columns: {len(sample_df_eval.columns)}")

# Export yelp_df
yelp_output_path = os.path.join(output_dir, 'processed_yelp.csv')
yelp_df.to_csv(yelp_output_path, index=False)
print(f"Exported processed Yelp data to: {yelp_output_path}")
print(f"  Rows: {len(yelp_df)}, Columns: {len(yelp_df.columns)}")

print("\nExport complete!")
