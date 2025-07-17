import pandas as pd
import json
import os
import sys
import time
from datetime import datetime

def run_detector_v1_sequential(df):
    """Run V1 in isolation"""
    from detector_v1 import CupHandleDetector as V1Detector
    
    custom_config = {
        "rim_height_tolerance_pct": 1.0,
        "min_cup_roundness": 0.3,
        "min_cup_symmetry": 0.15,
        "min_quality_score": 55,
        "min_cup_depth": 0.003,
        "breakout_max_above_rim_pct": 0.5,
        "disable_deduplication": True,
        "min_handle_gap_minutes": 150, 
        "max_handle_depth_pct": 25.0,
        "min_handle_depth_pct": 0.05,
        "breakout_minimum_pct": 100.2,
        "handle_must_be_below_rim_pct": 2.0,  
        "breakout_tolerance_pct": 0.2,
        "max_price_above_rim_during_cup_pct": 0.1, 
        "min_handle_zone_duration_minutes": 150,
        "breakout_minimum_above_rim_pct": 0.2,
        "breakout_minimum_above_rim_points": 5.0,    
        "process_resistance_limit": 500,    
        "max_resistance_bars": 250000,   
        "min_cup_atr_multiple": 4.0,        
        "min_handle_atr_multiple": 1.5,     
        "atr_period": 20,                   
        "min_cup_duration": 60,
        "max_cup_duration": 2880,
        "min_handle_duration": 30,
        "max_handle_duration": 360,
        "breakout_search_duration": 480,
        "rim_search_duration": 1440,
        "handle_search_duration": 960,
        "max_handle_drop": 25.0,
        "skip_rim_adjustment": True,
    }

    detector = V1Detector(config=custom_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df.copy(deep=True))
    
    return patterns if patterns else []

def run_detector_v2_sequential(df):
    """Run V2 in isolation"""
    from detector_v2 import CupHandleDetector as V2Detector
    
    v2_config = {
        "rim_height_tolerance_pct": 1.5,
        "min_cup_roundness": 0.05,
        "min_cup_symmetry": 0.08,
        "min_quality_score": 55,
        "min_cup_depth": 0.003,
        "breakout_max_above_rim_pct": 0.5,
        "disable_deduplication": True,
        "enable_formation_first": True,
        "max_handle_depth_pct": 75.0,
        "min_handle_depth_pct": 0.03,
        "breakout_minimum_pct": 100.2,
        "handle_must_be_below_rim_pct": 5.0,
        "breakout_tolerance_pct": 0.2,
        "max_price_above_rim_during_cup_pct": 0.1,
        "min_handle_zone_duration_minutes": 150,
        "breakout_minimum_above_rim_pct": 0.2,
        "breakout_minimum_above_rim_points": 5.0,
        "process_resistance_limit": 500,
        "max_resistance_bars": 250000,
        "min_cup_atr_multiple": 4.0,
        "min_handle_atr_multiple": 1.5,
        "atr_period": 20,
        "min_cup_duration": 30,
        "max_cup_duration": 2880,
        "min_handle_duration": 30,
        "max_handle_duration": 360,
        "breakout_search_duration": 480,
        "rim_search_duration": 1440,
        "handle_search_duration": 1440,
        "max_handle_drop": 75.0,
        "use_hybrid_detection": True,
        "enable_atr_filtering": True,
        "atr_multiplier": 1.5,
        "skip_rim_adjustment": True,
    }
    
    detector = V2Detector(config=v2_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df.copy(deep=True))
    
    return patterns if patterns else []

def run_detector_v3_sequential(df):
    """Run V3 in isolation"""
    from detector_v3 import CupHandleDetector as V3Detector
    
    v3_config = {
        "rim_height_tolerance_pct": 1.0,
        "min_cup_roundness": 0.1,
        "min_cup_symmetry": 0.15,
        "min_quality_score": 55,
        "min_cup_depth": 0.003,
        "disable_deduplication": True,
        "min_handle_gap_minutes": 150,
        "max_handle_depth_pct": 25.0,
        "min_handle_depth_pct": 0.05,
        "breakout_minimum_pct": 100.2,
        "handle_must_be_below_rim_pct": 2.0,
        "breakout_tolerance_pct": 0.2,
        "max_price_above_rim_during_cup_pct": 0.1,
        "min_handle_zone_duration_minutes": 150,
        "breakout_minimum_above_rim_pct": 0.2,
        "breakout_minimum_above_rim_points": 5.0,
        "process_resistance_limit": 500,
        "max_resistance_bars": 250000,
        "min_cup_atr_multiple": 4.0,
        "min_handle_atr_multiple": 1.5,
        "atr_period": 20,
        "min_cup_duration": 60,
        "max_cup_duration": 2880,
        "min_handle_duration": 30,
        "max_handle_duration": 360,
        "breakout_search_duration": 480,
        "rim_search_duration": 1440,
        "handle_search_duration": 960,
        "max_handle_drop": 25.0,
        "skip_rim_adjustment": True,
    }
    
    detector = V3Detector(config=v3_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df.copy(deep=True))
    
    return patterns if patterns else []

def validate_pattern_timeline(patterns, version_name):
    """Check for D=E issues"""
    invalid_patterns = []
    
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, dict):
            handle_d = pattern.get('handle_d')
            breakout_e = pattern.get('breakout_e')
            
            if handle_d and breakout_e:
                if pd.to_datetime(handle_d) == pd.to_datetime(breakout_e):
                    invalid_patterns.append({
                        'pattern_index': i,
                        'handle_d': handle_d,
                        'breakout_e': breakout_e,
                        'issue': 'same_timestamp'
                    })
                    print(f"❌ {version_name} Pattern {i+1}: D={handle_d} == E={breakout_e}")
                elif pd.to_datetime(handle_d) > pd.to_datetime(breakout_e):
                    invalid_patterns.append({
                        'pattern_index': i,
                        'handle_d': handle_d,
                        'breakout_e': breakout_e,
                        'issue': 'reversed_timeline'
                    })
                    print(f"❌ {version_name} Pattern {i+1}: D={handle_d} > E={breakout_e}")
    
    if not invalid_patterns:
        print(f"✅ {version_name}: All patterns have valid D < E timeline")
    
    return invalid_patterns

def main():
    # Load data
    input_path = "ES_timeframes/ES_15min.csv"
    
    print(f"📂 Loading data from {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Filter data
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    original_length = len(df)
    df = df[df.index >= cutoff_date]
    print(f"📅 Filtered: {original_length} → {len(df)} bars (from {cutoff_date.date()})")
    
    print(f"\n🔍 SEQUENTIAL EXECUTION TEST (no threading)")
    print(f"=" * 60)
    
    # Run detectors one by one (no parallel)
    results = {}
    
    # V1
    print(f"\n🔍 Running V1 sequentially...")
    start_time = time.time()
    v1_patterns = run_detector_v1_sequential(df)
    v1_time = time.time() - start_time
    v1_invalid = validate_pattern_timeline(v1_patterns, "V1")
    results['V1'] = {'patterns': v1_patterns, 'count': len(v1_patterns), 'time': v1_time, 'invalid': v1_invalid}
    print(f"✅ V1: {len(v1_patterns)} patterns in {v1_time:.2f}s")
    
    # V2  
    print(f"\n🔍 Running V2 sequentially...")
    start_time = time.time()
    v2_patterns = run_detector_v2_sequential(df)
    v2_time = time.time() - start_time
    v2_invalid = validate_pattern_timeline(v2_patterns, "V2")
    results['V2'] = {'patterns': v2_patterns, 'count': len(v2_patterns), 'time': v2_time, 'invalid': v2_invalid}
    print(f"✅ V2: {len(v2_patterns)} patterns in {v2_time:.2f}s")
    
    # V3
    print(f"\n🔍 Running V3 sequentially...")
    start_time = time.time()
    v3_patterns = run_detector_v3_sequential(df)
    v3_time = time.time() - start_time
    v3_invalid = validate_pattern_timeline(v3_patterns, "V3")
    results['V3'] = {'patterns': v3_patterns, 'count': len(v3_patterns), 'time': v3_time, 'invalid': v3_invalid}
    print(f"✅ V3: {len(v3_patterns)} patterns in {v3_time:.2f}s")
    
    # Summary
    print(f"\n📊 SEQUENTIAL EXECUTION SUMMARY:")
    print(f"=" * 60)
    total_invalid = 0
    for version, result in results.items():
        invalid_count = len(result['invalid'])
        total_invalid += invalid_count
        status = "✅" if invalid_count == 0 else "❌"
        print(f"{status} {version}: {result['count']} patterns, {invalid_count} invalid timeline issues")
    
    if total_invalid == 0:
        print(f"\n🎉 NO TIMELINE ISSUES in sequential execution!")
        print(f"   This confirms the problem is threading-related.")
    else:
        print(f"\n⚠️  {total_invalid} timeline issues found even in sequential execution.")
        print(f"   This suggests a fundamental logic problem in the detector code.")
    
    # Save results
    output_data = {
        'execution_type': 'sequential_test',
        'total_invalid_patterns': total_invalid,
        'results': results,
        'conclusion': 'threading_issue' if total_invalid == 0 else 'logic_issue'
    }
    
    with open('results/sequential_test_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"💾 Test results saved to results/sequential_test_results.json")

if __name__ == "__main__":
    main()