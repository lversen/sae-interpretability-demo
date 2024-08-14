import pandas as pd

# Read the main dataset
df = pd.read_csv("data/final_data.csv")

# Filter for Harry Potter entries
harry_potter_df = df[df['Name'].str.contains('Harry Potter', case=False, na=False)]

# Filter for Star Wars entries
star_wars_df = df[df['Name'].str.contains('Star Wars:', case=False, na=False)]

# Combine the two filtered DataFrames
test_df = pd.concat([harry_potter_df, star_wars_df], ignore_index=True)

# Print the shape of the resulting DataFrame to verify
print(f"Combined DataFrame shape: {test_df.shape}")