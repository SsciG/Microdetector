#!/usr/bin/env python3
"""
Standalone test for micro detector
"""
import pandas as pd
import time
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the micro detector
try:
    from detector_micro import CupHandleDetector as DetectorMicro
    print("✅ Successfully imported DetectorMicro")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_micro_detector():
    print("🔬 TESTING MICRO DETECTOR")
    print("=" * 50)
    
    # Load test data
    data_path = "ES_timeframes/ES_15min.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        print(f"📂 Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=[0])
        df.set_index(df.columns[0], inplace=True)
        
        # Use smaller dataset for testing
        original_len = len(df)
        df = df.tail(1000)  # Last 1000 bars only
        print(f"📊 Using {len(df)} bars (from {original_len} total)")
        
        # Create detector
        print("🔧 Creating micro detector...")
        detector = DetectorMicro()
        print(f"   Config loaded: {len(detector.config)} parameters")
        
        # Time the detection
        print("🔍 Starting detection...")
        start_time = time.time()
        
        try:
            patterns = detector.detect(df)
            end_time = time.time()
            
            # Results
            print(f"\n✅ DETECTION COMPLETED:")
            print(f"   Patterns found: {len(patterns) if patterns else 0}")
            print(f"   Execution time: {end_time - start_time:.2f} seconds")
            
            if patterns and len(patterns) > 0:
                print(f"\n📋 FIRST PATTERN DETAILS:")
                p = patterns[0]
                for key, value in p.items():
                    print(f"     {key}: {value}")
            else:
                print(f"\n⚠️  No patterns found")
                
            return True
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting micro detector test...")
    success = test_micro_detector()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)