import pandas as pd
from detector_micro import CupHandleDetector
import sys

# Load data
df = pd.read_csv("ES_timeframes/ES_15min.csv", parse_dates=[0])
df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
df.set_index('timestamp', inplace=True)

# Filter recent data
cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
df = df[df.index >= cutoff_date]

print(f"Testing with {len(df)} bars")

# Add logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG)

# Run detector
micro_detector = CupHandleDetector()

# Add timeout check
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Detection took too long - possible infinite loop")

# Set 30 second timeout
if sys.platform != "win32":
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

try:
    patterns = micro_detector.detect(df)
    print(f"Found {len(patterns)} patterns")
except TimeoutError:
    print("TIMEOUT: Infinite loop detected!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()