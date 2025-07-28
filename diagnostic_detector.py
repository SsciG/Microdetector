import pandas as pd
import json
import hashlib
from detector_v1 import CupHandleDetector
import os

def create_data_fingerprint(df):
    """Create a fingerprint of the DataFrame to ensure it's identical"""
    # Create a hash of the key data properties
    fingerprint = {
        'shape': df.shape,
        'columns': list(df.columns),
        'index_start': str(df.index[0]),
        'index_end': str(df.index[-1]),
        'first_close': df['close'].iloc[0],
        'last_close': df['close'].iloc[-1],
        'close_sum': df['close'].sum(),
        'data_hash': hashlib.md5(df.to_string().encode()).hexdigest()[:16]
    }
    return fingerprint

def create_config_fingerprint(config):
    """Create a fingerprint of the configuration"""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]

def run_individual_detection_with_diagnostics(df, config, version_name="v1"):
    """Run detection exactly like your individual scripts, but with diagnostics"""
    
    print(f"\n🔍 DIAGNOSTIC RUN: {version_name}")
    print(f"=" * 50)
    
    # Step 1: Data diagnostics
    data_fp = create_data_fingerprint(df)
    print(f"📊 DATA FINGERPRINT:")
    print(f"   Shape: {data_fp['shape']}")
    print(f"   Time range: {data_fp['index_start']} to {data_fp['index_end']}")
    print(f"   First/Last close: {data_fp['first_close']:.2f} / {data_fp['last_close']:.2f}")
    print(f"   Data hash: {data_fp['data_hash']}")
    
    # Step 2: Config diagnostics
    config_fp = create_config_fingerprint(config)
    print(f"⚙️  CONFIG FINGERPRINT:")
    print(f"   Hash: {config_fp}")
    print(f"   Key settings: rim_tol={config.get('rim_height_tolerance_pct')}, min_depth={config.get('min_cup_depth')}")
    print(f"   Skip rim adj: {config.get('skip_rim_adjustment')}")
    
    # Step 3: Detector setup diagnostics
    print(f"🏗️  DETECTOR SETUP:")
    detector = CupHandleDetector(config=config)
    print(f"   Created detector instance")
    print(f"   Original method: {hasattr(detector, 'detect_cup_and_handle')}")
    print(f"   Deduped method exists: {hasattr(detector, 'detect_cup_and_handle_deduped')}")
    
    # Step 4: Method replacement (EXACTLY like your individual scripts)
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    print(f"   Method replaced successfully")
    print(f"   Current method: {detector.detect_cup_and_handle.__name__}")
    
    # Step 5: Detection execution with timing
    start_time = time.time()
    print(f"🚀 STARTING DETECTION...")
    
    patterns = detector.detect(df)
    
    execution_time = time.time() - start_time
    print(f"✅ DETECTION COMPLETE in {execution_time:.2f}s")
    
    # Step 6: Results diagnostics
    if patterns is None:
        patterns = []
    
    print(f"📈 RESULTS:")
    print(f"   Patterns found: {len(patterns)}")
    
    if len(patterns) > 0:
        for i, pattern in enumerate(patterns[:3]):  # Show first 3
            print(f"   Pattern {i+1}:")
            print(f"     Peak A: {pattern.get('peak_a')}")
            print(f"     Quality: {pattern.get('quality_score', 0):.1f}")
            print(f"     Cup depth: {pattern.get('cup_depth_pct', 0):.2f}%")
    
    # Step 7: Detection stats
    if hasattr(detector, 'detection_stats'):
        stats = detector.detection_stats
        print(f"🏛️  DETECTION STATS:")
        print(f"   Resistance levels: {stats['resistance_levels']['total_found']}")
        print(f"   Valid accumulations: {stats['accumulation_analysis']['valid_accumulations']}")
        print(f"   Handles found: {stats['handle_analysis']['handles_found']}")
        print(f"   Patterns created: {stats['pattern_creation']['patterns_created']}")
    
    return {
        'patterns': patterns,
        'count': len(patterns),
        'execution_time': execution_time,
        'data_fingerprint': data_fp,
        'config_fingerprint': config_fp,
        'diagnostics': {
            'method_replaced': True,
            'detector_class': str(type(detector)),
            'detection_stats': getattr(detector, 'detection_stats', {})
        }
    }

def compare_results(result1, result2, name1="Run 1", name2="Run 2"):
    """Compare two detection results in detail"""
    print(f"\n🔄 COMPARING {name1} vs {name2}")
    print(f"=" * 50)
    
    # Basic comparison
    print(f"📊 BASIC COMPARISON:")
    print(f"   {name1}: {result1['count']} patterns in {result1['execution_time']:.2f}s")
    print(f"   {name2}: {result2['count']} patterns in {result2['execution_time']:.2f}s")
    print(f"   Count match: {result1['count'] == result2['count']}")
    
    # Data fingerprint comparison
    print(f"📋 DATA FINGERPRINT COMPARISON:")
    fp1, fp2 = result1['data_fingerprint'], result2['data_fingerprint']
    print(f"   Shape match: {fp1['shape'] == fp2['shape']}")
    print(f"   Hash match: {fp1['data_hash'] == fp2['data_hash']}")
    print(f"   Close sum match: {abs(fp1['close_sum'] - fp2['close_sum']) < 0.01}")
    
    # Config fingerprint comparison
    print(f"⚙️  CONFIG FINGERPRINT COMPARISON:")
    print(f"   Config hash match: {result1['config_fingerprint'] == result2['config_fingerprint']}")
    
    # Pattern-level comparison if both have patterns
    if result1['count'] > 0 and result2['count'] > 0:
        print(f"🎯 PATTERN-LEVEL COMPARISON:")
        
        # Compare first pattern details
        p1 = result1['patterns'][0]
        p2 = result2['patterns'][0]
        
        print(f"   First pattern peak_a: {p1.get('peak_a')} vs {p2.get('peak_a')}")
        print(f"   First pattern quality: {p1.get('quality_score', 0):.2f} vs {p2.get('quality_score', 0):.2f}")
        print(f"   First pattern cup depth: {p1.get('cup_depth_pct', 0):.2f}% vs {p2.get('cup_depth_pct', 0):.2f}%")
    
    # Detection stats comparison
    if 'detection_stats' in result1['diagnostics'] and 'detection_stats' in result2['diagnostics']:
        stats1 = result1['diagnostics']['detection_stats']
        stats2 = result2['diagnostics']['detection_stats']
        
        print(f"🏛️  DETECTION STATS COMPARISON:")
        if 'resistance_levels' in stats1 and 'resistance_levels' in stats2:
            r1 = stats1['resistance_levels']['total_found']
            r2 = stats2['resistance_levels']['total_found']
            print(f"   Resistance levels: {r1} vs {r2} (match: {r1 == r2})")

if __name__ == "__main__":
    import time
    
    # Load your data exactly like in main.py
    input_path = "ES_timeframes/ES_15min.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        exit(1)
    
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Filter recent data (same as main.py)
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=12)
    original_length = len(df)
    df = df[df.index >= cutoff_date]
    print(f"📅 Filtered: {original_length} → {len(df)} bars (from {cutoff_date.date()})")
    
    # Your exact config from detector_v1.py
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
    
    # Run two consecutive individual detections to establish baseline
    print("🔬 RUNNING BASELINE TESTS...")
    
    # Test 1: Run detection exactly like your individual script
    result1 = run_individual_detection_with_diagnostics(df.copy(), custom_config.copy(), "Individual-1")
    
    # Test 2: Run again to check for consistency
    result2 = run_individual_detection_with_diagnostics(df.copy(), custom_config.copy(), "Individual-2")
    
    # Compare the two individual runs
    compare_results(result1, result2, "Individual-1", "Individual-2")
    
    print(f"\n🎯 BASELINE ESTABLISHED:")
    print(f"   Expected pattern count: {result1['count']}")
    print(f"   Expected execution time: ~{result1['execution_time']:.2f}s")
    print(f"   Consistency: {'✅ GOOD' if result1['count'] == result2['count'] else '❌ INCONSISTENT'}")