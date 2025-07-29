import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Load a small sample of data to test
df = pd.read_csv("ES_timeframes/ES_15min.csv", parse_dates=[0])
df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
df.set_index('timestamp', inplace=True)

# Get just February 6 data around pattern 17
test_data = df['2025-02-06 08:00:00':'2025-02-06 20:00:00']

print(f"Testing with {len(test_data)} bars")

# Find peaks
highs, _ = find_peaks(test_data['high'].values, distance=2, prominence=0.5)
peaks = test_data.index[highs].tolist()

print(f"Found {len(peaks)} peaks")

# Test the peak adjustment logic
iteration_count = 0
max_iterations = 100

for i, peak_a in enumerate(peaks[:-1]):
    for j, original_peak_c in enumerate(peaks[i+1:]):
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"ERROR: Too many iterations - possible infinite loop!")
            break
            
        peak_c = original_peak_c
        
        # Find actual highest point between peak_a and peak_c
        cup_validation_period = test_data.loc[peak_a:peak_c]
        cup_max_high = cup_validation_period['high'].max()
        cup_max_high_idx = cup_validation_period['high'].idxmax()
        
        # If there's a higher point than peak_c, use it as the actual peak_c
        if cup_max_high > test_data.loc[peak_c, 'high']:
            print(f"Adjusting peak_c from {peak_c} ({test_data.loc[peak_c, 'high']:.2f}) to {cup_max_high_idx} ({cup_max_high:.2f})")
            peak_c = cup_max_high_idx
            
        # This is where the pattern would continue...
        print(f"  Iteration {iteration_count}: peak_a={peak_a}, original_peak_c={original_peak_c}, adjusted_peak_c={peak_c}")
        
print(f"Total iterations: {iteration_count}")