import pandas as pd
import json
import time
import os
from detector_v3 import CupHandleDetector

class IncrementalTester:
    def __init__(self, data_path):
        """Load data once"""
        df = pd.read_csv(data_path, parse_dates=[0])
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df.set_index('timestamp', inplace=True)
        
        # Filter like your main.py
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
        df = df[df.index >= cutoff_date]
        self.df = df
        print(f"📊 Data loaded: {len(df)} bars")
        
        # Your current working config (baseline)
        self.baseline_config = {
            "rim_height_tolerance_pct": 1.0,
            "min_cup_roundness": 0.3,
            "min_cup_symmetry": 0.05,
            "min_quality_score": 55,
            "min_cup_depth": 0.001,
            "disable_deduplication": True,
            "min_handle_gap_minutes": 150, 
            "max_handle_depth_pct": 150.0,
            "min_handle_depth_pct": 0.05,
            "process_resistance_limit": 500,
            "min_cup_duration": 30,           # ← Current: 30 minutes minimum
            "max_cup_duration": 2880,
            "min_handle_duration": 30,        # ← Current: 30 minutes minimum
            "max_handle_duration": 360,
            "breakout_search_duration": 240,
            "skip_rim_adjustment": True,
        }
        
        self.baseline_result = None
    
    def test_baseline(self):
        """Test current configuration"""
        print("\n🏁 TESTING BASELINE (your current config)")
        print("-" * 50)
        
        result = self._run_single_test("BASELINE", self.baseline_config)
        self.baseline_result = result
        
        print(f"✅ BASELINE: {result['count']} patterns in {result['time']:.1f}s")
        return result
    
    def test_single_change(self, param_name, new_value, description):
        """Test changing just ONE parameter"""
        print(f"\n🧪 TESTING: {description}")
        print(f"   Changing {param_name}: {self.baseline_config.get(param_name, 'NOT_SET')} → {new_value}")
        print("-" * 50)
        
        # Create new config with just this one change
        test_config = self.baseline_config.copy()
        test_config[param_name] = new_value
        
        result = self._run_single_test(f"{param_name}={new_value}", test_config)
        
        # Compare to baseline
        if self.baseline_result:
            change = result['count'] - self.baseline_result['count']
            if change > 0:
                print(f"📈 IMPROVEMENT: +{change} patterns ({self.baseline_result['count']} → {result['count']})")
            elif change < 0:
                print(f"📉 DECLINE: {change} patterns ({self.baseline_result['count']} → {result['count']})")
            else:
                print(f"➡️  NO CHANGE: Still {result['count']} patterns")
        
        return result
    
    def _run_single_test(self, name, config):
        """Run detection with one config"""
        detector = CupHandleDetector(config=config)
        
        # Apply your working method replacement
        detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
        detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
        
        start_time = time.time()
        patterns = detector.detect(self.df.copy())
        execution_time = time.time() - start_time
        
        if patterns is None:
            patterns = []
        
        avg_quality = sum(p.get('quality_score', 0) for p in patterns) / len(patterns) if patterns else 0
        
        result = {
            'name': name,
            'count': len(patterns),
            'time': execution_time,
            'avg_quality': avg_quality,
            'patterns': patterns,
            'config': config
        }
        
        print(f"   Result: {len(patterns)} patterns, quality={avg_quality:.1f}, time={execution_time:.1f}s")
        return result

# Functions outside the class
def run_test_and_save(param_name, new_value):
    """Test one change and save results for labeler"""
    print(f"🚀 Testing {param_name} = {new_value}")
    
    tester = IncrementalTester("ES_timeframes/ES_5min.csv")
    baseline = tester.test_baseline()
    result = tester.test_single_change(param_name, new_value, f"Testing {param_name}")
    
    # Save results if we got patterns
    if result['patterns']:
        os.makedirs("test_results", exist_ok=True)
        
        # Create labeler-compatible format
        labeler_patterns = []
        for i, pattern in enumerate(result['patterns']):
            labeler_pattern = {
                'id': f"test_pattern_{i+1}",
                'type': 'positive',
                'timestamps': [
                    pattern['peak_a'],
                    pattern['trough_b'], 
                    pattern['peak_c'],
                    pattern['handle_d'],
                    pattern['breakout_e']
                ],
                'algorithm_confidence': pattern.get('confidence_score', 0.8),
                'algorithm_quality': pattern.get('quality_score', 0),
                'cup_depth_pct': pattern.get('cup_depth_pct', 0),
                'handle_depth_pct': pattern.get('handle_depth_pct', 0),
                'source_version': f"test_{param_name}_{new_value}",
                'labeled_at': pd.Timestamp.now().isoformat(),
                'source': 'incremental_test'
            }
            labeler_patterns.append(labeler_pattern)
        
        # Save for labeler
        filename = f"test_results/patterns_{param_name}_{new_value}.json"
        with open(filename, 'w') as f:
            json.dump(labeler_patterns, f, indent=2, default=str)
        
        print(f"💾 Saved {len(labeler_patterns)} patterns to {filename}")
        print(f"📋 Load this file in the labeler to review patterns!")
    else:
        print(f"❌ No patterns found to save")
    
    return baseline['count'], result['count']

def quick_test_one_change(param_name, new_value):
    """Test just one parameter change quickly"""
    print(f"🚀 Quick test: {param_name} = {new_value}")
    
    tester = IncrementalTester("ES_timeframes/ES_15min.csv")
    baseline = tester.test_baseline()
    result = tester.test_single_change(param_name, new_value, f"Testing {param_name}")
    
    return baseline['count'], result['count']

# Main execution
if __name__ == "__main__":
    print("🔬 INCREMENTAL TESTING")
    print("Testing ONE change at a time and saving results...")

    print("\n🚀 Testing breakout_tolerance_pct = 0.5...")
    baseline_count, new_count = run_test_and_save("breakout_tolerance_pct", 0.5)
    print(f"Result: {baseline_count} → {new_count} patterns")
    
    # Test 1: Handle duration (since symmetry is already at 0.05 in baseline)
    print("\n1️⃣ Testing min_handle_duration = 15...")
    baseline_count, new_count = run_test_and_save("min_handle_duration", 15)
    print(f"Result: {baseline_count} → {new_count} patterns")
    
    # Test 2: Quality threshold
    print("\n2️⃣ Testing min_quality_score = 45...")
    baseline_count, new_count = run_test_and_save("min_quality_score", 45)
    print(f"Result: {baseline_count} → {new_count} patterns")
    
    # Test 3: Processing limits
    print("\n3️⃣ Testing process_resistance_limit = 100...")
    baseline_count, new_count = run_test_and_save("process_resistance_limit", 100)
    print(f"Result: {baseline_count} → {new_count} patterns")

    print("\n🚀 Testing disable_deduplication = False...")
    baseline_count, new_count = run_test_and_save("disable_deduplication", False)
    print(f"Result: {baseline_count} → {new_count} patterns")


    print("\n🚀 Testing max_handle_depth_pct = 50...")
    baseline_count, new_count = run_test_and_save("max_handle_depth_pct", 50.0)
    print(f"Result: {baseline_count} → {new_count} patterns")
    
    print(f"\n✅ RESULTS SAVED!")
    print(f"📁 Check test_results/ folder for JSON files")
    print(f"📋 Load any .json file in the Cup & Handle Labeler to review patterns")
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. python cup_handle_labeler_dash_copy.py")
    print(f"2. Load ES_15min.csv data")
    print(f"3. Load test_results/patterns_*.json to review patterns")