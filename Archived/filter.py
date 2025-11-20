import pandas as pd
import ast

# Load the weighted algorithm labels dataset
print("Loading weighted_algorithm_labels.csv...")
df = pd.read_csv('../output_data/weighted_algorithm_labels.csv')
print(f"Total records: {len(df)}")

# Parse the addresses column to extract country information
def get_country_from_address(address_str):
    """Extract country code from address string representation"""
    if pd.isna(address_str) or not address_str:
        return None
    
    try:
        # Parse the string representation of the list of dictionaries
        address_list = ast.literal_eval(address_str)
        if isinstance(address_list, list) and len(address_list) > 0:
            first_address = address_list[0]
            if isinstance(first_address, dict):
                return first_address.get('country')
    except (ValueError, SyntaxError):
        pass
    
    return None

# Extract country from both addresses (primary and base)
print("\nExtracting country information from addresses...")
df['country'] = df['addresses'].apply(get_country_from_address)
df['base_country'] = df['base_addresses'].apply(get_country_from_address)

# Filter for US and Canada
# Include records where either address (Place A or Place B) is from US or CA
print("\nFiltering for US and Canada only...")
us_ca_mask = (
    (df['country'].isin(['US', 'CA'])) | 
    (df['base_country'].isin(['US', 'CA']))
)

filtered_df = df[us_ca_mask].copy()

# Remove the temporary country columns
filtered_df = filtered_df.drop(columns=['country', 'base_country'])

# Export to filtered_labels.csv
output_path = '../output_data/filtered_labels.csv'
filtered_df.to_csv(output_path, index=False)

print(f"\n=== FILTERING SUMMARY ===")
print(f"Original records: {len(df)}")
print(f"Filtered records (US/CA): {len(filtered_df)}")
print(f"Records removed: {len(df) - len(filtered_df)}")
print(f"Percentage retained: {len(filtered_df)/len(df)*100:.1f}%")
print(f"\n✓ Exported to {output_path}")
