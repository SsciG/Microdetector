#!/usr/bin/env python3
"""
Test the STRICT micro detector with ground rules
"""

import pandas as pd
import sys
import os

def test_strict_micro():
    """Test the strict micro detector"""
    
    print("🔒 TESTING STRICT MICRO DETECTOR - GROUND RULES ENFORCED")
    print("=" * 70)
    
    # Load data
    input_path = "ES_timeframes/ES_15min.csv"
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Check data range
    print(f"📅 Data range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Use last 6 months for testing
    cutoff_date = df.index[-1] - pd.DateOffset(months=6)
    df = df[df.index >= cutoff_date]
    
    print(f"📅 Using {len(df)} bars from {cutoff_date.date()}")
    
    if len(df) == 0:
        print("❌ No data after filtering")
        return
    
    # Test with strict micro detector
    from detector_micro import CupHandleDetector
    
    print(f"\n🔍 Running STRICT micro detector...")
    print(f"📋 Ground Rules:")
    print(f"   1. Rim symmetry: ≤2% difference")
    print(f"   2. Handle: Below rim, above cup bottom")
    print(f"   3. Breakout: At or above rim level")
    print(f"   4. Duration: 1-8 hours ONLY")
    print(f"   5. Cup depth: 0.5%-5% for micro patterns")
    
    detector = CupHandleDetector()
    patterns = detector.detect(df)
    
    print(f"\n📊 STRICT RESULTS:")
    print(f"   🔒 Patterns found: {len(patterns)}")
    print(f"   ⏱️  Should complete quickly (strict filtering)")
    
    if len(patterns) > 0:
        print(f"\n📈 STRICT MICRO PATTERNS:")
        for i, pattern in enumerate(patterns):
            peak_a = pattern['peak_a']
            cup_depth = pattern.get('cup_depth_pct', 0)
            duration_min = pattern.get('cup_duration_min', 0)
            duration_hours = duration_min / 60
            
            print(f"   Pattern {i+1}:")
            print(f"     Time: {peak_a}")
            print(f"     Depth: {cup_depth:.2f}%")
            print(f"     Duration: {duration_hours:.1f} hours")
            print(f"     Type: {pattern.get('pattern_type', 'unknown')}")
    else:
        print(f"\n💡 No patterns found - this is normal with strict rules!")
        print(f"   Strict filtering ensures only perfect micro patterns pass")
        print(f"   Try relaxing parameters if needed")
    
    return len(patterns)

if __name__ == "__main__":
    test_strict_micro()