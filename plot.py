import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

file_paths = glob.glob('data_episodelen_100/*.csv')

# 1. Create the figure BEFORE starting to read the files
plt.figure(figsize=(14, 8))

print(f"Found {len(file_paths)} files to process.\n")

for file_path in file_paths:
    print(f"Reading: {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        limit = 1_000_000
        subset = df.iloc[:limit]
        
        # Optional: If the data is very noisy, you can use a moving average
        # subset['Value'] = subset['Value'].rolling(window=50).mean()

        # Define a clean label (filename without the .csv extension)
        label_name = os.path.splitext(file_path)[0]
        
        # 2. Plot the line for this file in the figure created earlier
        # We removed 'color="blue"' so each line has a different color
        plt.plot(subset['Step'], subset['Value'], label=label_name, linewidth=2, alpha=1)
        
        # NOTE: I removed the standard deviation ranges (fill_between) and average lines
        # as in a comparative chart they cause too much visual clutter.
        
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")

# 3. Final chart settings (outside the loop)
plt.title(f'Comparative Evolution of Samples (First {limit} iterations)', fontsize=24)
plt.xlabel('Step', fontsize=18)
plt.ylabel('Episode Len', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

# Position the legend outside the chart if there are many files, or in the best place
plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
plt.tight_layout()

output_name = 'comparative_chart_overall.png'
plt.savefig(output_name) # dpi=300 improves resolution
print(f"\n{'-'*50}")
print(f"Chart saved as: {output_name}")
plt.show()

