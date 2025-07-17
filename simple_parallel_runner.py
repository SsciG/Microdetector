import pandas as pd
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

def run_detector_v1_exact(df_copy):
    """Run detector_v1.py EXACTLY like your individual script"""
    # Copy your exact V1 individual execution code here
    from detector_v1 import CupHandleDetector
    
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

    detector = CupHandleDetector(config=custom_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df_copy)
    
    return patterns if patterns else []

def run_detector_v2_exact(df_copy):
    """Run detector_v2.py exactly like it would run individually"""
    # This should mimic exactly how you run detector_v2.py individually
    from detector_v2 import CupHandleDetector
    
    # Use the SAME config as V1 since that's what works
    # (assuming V2 and V3 should use similar configs when run individually)
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

    detector = CupHandleDetector(config=custom_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df_copy)
    
    return patterns if patterns else []

def run_detector_v3_exact(df_copy):
    """Run detector_v3.py exactly like it would run individually"""
    from detector_v3 import CupHandleDetector
    
    # Use the SAME config as V1 but with V3 specific modification (relaxed roundness)
    custom_config = {
        "rim_height_tolerance_pct": 1.0,
        "min_cup_roundness": 0.1,  # V3 specific: more relaxed roundness
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

    detector = CupHandleDetector(config=custom_config)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    patterns = detector.detect(df_copy)
    
    return patterns if patterns else []

def run_single_detector_wrapper(detector_func, df, version_name):
    """Wrapper to run one detector with error handling"""
    start_time = time.time()
    try:
        print(f"🔍 Starting {version_name}...")
        # Create a completely independent copy of the data
        df_copy = df.copy(deep=True)
        patterns = detector_func(df_copy)
        execution_time = time.time() - start_time
        print(f"✅ {version_name}: {len(patterns)} patterns in {execution_time:.2f}s")
        
        return {
            'version': version_name,
            'patterns': patterns,
            'count': len(patterns),
            'execution_time': execution_time,
            'status': 'success'
        }
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {version_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'version': version_name,
            'patterns': [],
            'count': 0,
            'execution_time': execution_time,
            'status': 'failed',
            'error': str(e)
        }

def main():
    # Load data exactly like individual scripts
    input_path = "ES_timeframes/ES_15min.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Loading data from {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Filter data exactly like individual scripts
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    original_length = len(df)
    df = df[df.index >= cutoff_date]
    print(f"📅 Filtered: {original_length} → {len(df)} bars (from {cutoff_date.date()})")
    
    # Define the 3 detectors with their exact individual execution functions
    detectors = [
        (run_detector_v1_exact, 'V1'),
        (run_detector_v2_exact, 'V2'), 
        (run_detector_v3_exact, 'V3')
    ]
    
    print(f"\n🚀 Running 3 detectors in parallel (exact individual execution)...")
    start_time = time.time()
    
    results = {}
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_version = {
            executor.submit(run_single_detector_wrapper, detector_func, df, version_name): version_name
            for detector_func, version_name in detectors
        }
        
        # Collect results
        for future in as_completed(future_to_version):
            version_name = future_to_version[future]
            try:
                result = future.result()
                results[result['version']] = result
            except Exception as e:
                print(f"❌ {version_name} failed with exception: {e}")
                results[version_name] = {
                    'version': version_name,
                    'patterns': [],
                    'count': 0,
                    'execution_time': 0,
                    'status': 'failed',
                    'error': str(e)
                }
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n📊 EXACT INDIVIDUAL EXECUTION RESULTS:")
    print(f"=" * 50)
    for version, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} {version}: {result['count']} patterns ({result['execution_time']:.2f}s)")
    
    print(f"\n⏱️  Total time: {total_time:.2f}s")
    
    # Combine all patterns
    all_patterns = []
    for version, result in results.items():
        for pattern in result['patterns']:
            pattern_copy = pattern.copy() if isinstance(pattern, dict) else pattern
            if isinstance(pattern_copy, dict):
                pattern_copy['source_version'] = version
            all_patterns.append(pattern_copy)
    
    # Save results
    output_data = {
        'metadata': {
            'detection_timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'data_bars': len(df),
            'execution_type': 'exact_individual_parallel'
        },
        'individual_results': results,
        'all_patterns': all_patterns,
        'total_patterns': len(all_patterns)
    }
    
    # Save to JSON
    output_file = os.path.join(output_dir, "exact_individual_results.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"💾 Results saved to {output_file}")
    
    # Convert to labeler format
    labeler_patterns = []
    for pattern in all_patterns:
        if isinstance(pattern, dict) and 'peak_a' in pattern:
            labeler_pattern = {
                'id': f"pattern_{pattern['peak_a']}_{pattern.get('source_version', 'unknown')}",
                'type': 'positive',
                'timestamps': [
                    pattern.get('peak_a'),
                    pattern.get('trough_b'), 
                    pattern.get('peak_c'),
                    pattern.get('handle_d'),
                    pattern.get('breakout_e')
                ],
                'algorithm_confidence': pattern.get('confidence_score', 0.8),
                'algorithm_quality': pattern.get('quality_score', 0),
                'cup_depth_pct': pattern.get('cup_depth_pct', 0),
                'handle_depth_pct': pattern.get('handle_depth_pct', 0),
                'source_version': pattern.get('source_version', 'unknown'),
                'source': 'exact_individual_parallel_detection'
            }
            labeler_patterns.append(labeler_pattern)
    
    labeler_file = os.path.join(output_dir, "patterns_for_labeler_exact.json")
    with open(labeler_file, 'w') as f:
        json.dump(labeler_patterns, f, indent=2, default=str)
    
    print(f"💾 Labeler patterns saved to {labeler_file}")
    print(f"\n🎉 Done! Found {len(all_patterns)} total patterns.")

if __name__ == "__main__":
    main()