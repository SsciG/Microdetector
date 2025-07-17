import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import traceback
import sys
import os

# Import your 4 detector classes directly
sys.path.append('.')
from detector_v1 import CupHandleDetector as DetectorV1
from detector_v2 import CupHandleDetector as DetectorV2  
from detector_v3 import CupHandleDetector as DetectorV3
from detector_micro import CupHandleDetector as DetectorMicro

class MultiVersionDetector:
    def __init__(self):
        """Initialize configs for all 4 detector versions"""
        print("🔧 Initializing 4 detector version configs...")
        
        # Define different configurations for each version
        self.detector_configs = {
            'v1': {
                'name': 'V1 (As-Is)', 
                'class': DetectorV1,
                'config': {
                                # Ultra-relaxed detection
                    "rim_height_tolerance_pct": 1.0,
                    "min_cup_roundness": 0.3,
                    "min_cup_symmetry": 0.15,
                    "min_quality_score": 55,
                    "min_cup_depth": 0.003,
                    "breakout_max_above_rim_pct": 0.5,
                    "disable_deduplication": True,
                    "min_handle_gap_minutes": 150, 
                    "max_handle_depth_pct": 150.0,  # Keep existing max
                    "min_handle_depth_pct": 0.05,   # ✅ FIXED: Reduced from 2.0 to 0.3 for ES futures
                    "breakout_minimum_pct": 100.2,
                    "handle_must_be_below_rim_pct": 2.0,  
                    "breakout_tolerance_pct": 0.2,
                    "breakout_tolerance_points": 10.0,  # Allow breakouts within 10 points of threshold
                    "breakout_near_miss_allowed": True,  # Accept "near miss" breakouts
                    "max_price_above_rim_during_cup_pct": 0.1, 
                    "min_handle_zone_duration_minutes": 150,
                    "breakout_minimum_above_rim_pct": 0.2,
                    "breakout_minimum_above_rim_points": 5.0,    
                    # Processing limits - INCREASED
                    "process_resistance_limit": 500,    
                    "max_resistance_bars": 250000,   
                    "min_cup_atr_multiple": 4.0,        
                    "min_handle_atr_multiple": 1.5,     
                    "atr_period": 20,                   
                    # Durations
                    "min_cup_duration": 15,
                    "max_cup_duration": 480,
                    "min_handle_duration": 15,
                    "max_handle_duration": 360,
                    "breakout_search_duration": 240,
                    "rim_search_duration": 1440,
                    "handle_search_duration": 960,
                    "max_handle_drop": 25.0,
                    
                    # Keep working settings
                    "skip_rim_adjustment": True,
}
            },
            'v2': {
                'name': 'V2 (Hybrid)', 
                'class': DetectorV2,
                'config': {
                            # Ultra-relaxed detection
                "rim_height_tolerance_pct": 1.0,
                "min_cup_roundness": 0.3,
                "min_cup_symmetry": 0.15,
                "min_quality_score": 55,
                "min_cup_depth": 0.003,
                "breakout_max_above_rim_pct": 0.5,
                "disable_deduplication": True,
                "enable_formation_first": True,
                "min_handle_gap_minutes": 150, 
                "max_handle_depth_pct": 150.0,  # Keep existing max
                "min_handle_depth_pct": 0.05,   # ✅ FIXED: Reduced from 2.0 to 0.3 for ES futures
                "breakout_minimum_pct": 100.2,
                "handle_must_be_below_rim_pct": 2.0,  
                "breakout_tolerance_pct": 0.2,
                "max_price_above_rim_during_cup_pct": 0.1, 
                "min_handle_zone_duration_minutes": 150,
                "breakout_minimum_above_rim_pct": 0.2,
                "breakout_minimum_above_rim_points": 5.0,    
                # Processing limits - INCREASED
                "process_resistance_limit": 500,    
                "max_resistance_bars": 250000,   
                "min_cup_atr_multiple": 4.0,        
                "min_handle_atr_multiple": 1.5,     
                "atr_period": 20,                   
                # Durations
                "breakout_tolerance_points": 10.0,  # Allow breakouts within 10 points of threshold
                "breakout_near_miss_allowed": True,  # Accept "near miss" breakouts
                "min_cup_duration": 15,
                "max_cup_duration": 480,
                "min_handle_duration": 15,
                "max_handle_duration": 360,
                "breakout_search_duration": 240,
                "rim_search_duration": 1440,
                "handle_search_duration": 960,
                "max_handle_drop": 25.0,
                "use_hybrid_detection": True,      
                "enable_atr_filtering": True,    
                "atr_multiplier": 1.5,           
                # Keep working settings
                "skip_rim_adjustment": True,
            }
            },
            'v3': {
                'name': 'V3 (Relaxed)', 
                'class': DetectorV3,
                'config': {
                    # MATCH YOUR STANDALONE DETECTOR EXACTLY
                    "rim_height_tolerance_pct": 1.0,
                    "min_cup_roundness": 0.1,                    # ← CHANGED from 0.3 to 0.1
                    "min_cup_symmetry": 0.15,
                    "min_quality_score": 55,
                    "min_cup_depth": 0.003,
                    "disable_deduplication": True,
                    "min_handle_gap_minutes": 150, 
                    "max_handle_depth_pct": 150.0,
                    "min_handle_depth_pct": 0.05,               # ← ADDED (was missing)
                    "breakout_minimum_pct": 100.2,              # ← ADDED (was missing)
                    "handle_must_be_below_rim_pct": 2.0,        # ← ADDED (was missing)
                    "breakout_tolerance_pct": 0.2,              # ← ADDED (was missing)
                    "max_price_above_rim_during_cup_pct": 0.1,  # ← ADDED (was missing)
                    "min_handle_zone_duration_minutes": 150,    # ← ADDED (was missing)
                    "breakout_minimum_above_rim_pct": 0.2,      # ← ADDED (was missing)
                    "breakout_minimum_above_rim_points": 5.0,   # ← ADDED (was missing)
                    "breakout_tolerance_points": 10.0,  # Allow breakouts within 10 points of threshold
                    "breakout_near_miss_allowed": True,  # Accept "near miss" breakouts
                    # Processing limits
                    "process_resistance_limit": 500,            # ← ADDED (was missing)
                    "max_resistance_bars": 250000,              # ← ADDED (was missing)
                    "min_cup_atr_multiple": 4.0,                # ← ADDED (was missing)
                    "min_handle_atr_multiple": 1.5,             # ← ADDED (was missing)
                    "atr_period": 20,                           # ← ADDED (was missing)
                    
                    # Durations - MATCH YOUR STANDALONE
                    "min_cup_duration": 15,                     # ← ADDED (was missing)
                    "max_cup_duration": 480,                   # ← ADDED (was missing)
                    "min_handle_duration": 15,                  # ← ADDED (was missing)
                    "max_handle_duration": 360,                 # ← ADDED (was missing)
                    "breakout_search_duration": 240,            # ← ADDED (was missing)
                    "rim_search_duration": 1440,                # ← ADDED (was missing)
                    "handle_search_duration": 960,              # ← ADDED (was missing)
                    "max_handle_drop": 25.0,                    # ← ADDED (was missing)
                    
                    # Critical settings
                    "skip_rim_adjustment": True,                 # ← ADDED (was missing)
                }
            },
            'micro': {
                'name': 'Micro Patterns',
                'class': DetectorMicro,
                'config': {
                    # Optimized for 15-minute timeframe micro patterns
                    "min_cup_bars": 3,              # Just 3 bars minimum (45 min)
                    "max_cup_bars": 20,             # Max 20 bars (5 hours)
                    "min_handle_bars": 1,           # Single bar handles OK
                    "max_handle_bars": 6,           # Max 6 bars (1.5 hours)
                    "min_cup_depth_pct": 0.05,     # Very shallow cups
                    "max_cup_depth_pct": 1.2,      # Keep it micro
                    "min_handle_depth_pct": 0.5,  # Tiny handle drops OK
                    "max_handle_depth_pct": 33.0,  # Not too deep
                    "rim_tolerance_points": 2.5,   # ES point tolerance
                    "min_quality_score": 70,       # Lower quality bar
                    "max_patterns_per_scan": 30,   # Limit for performance
                    "min_cup_roundness": 0.45,
                    "min_cup_duration": 180,
                }
            }
        }
        
        print(f"✅ All {len(self.detector_configs)} detector configs ready!")
    
    def detect_all_parallel(self, df, timeout_seconds=120):
        """Run all 4 versions in parallel with better error handling"""
        print(f"\n🚀 Running parallel detection on {len(df)} bars...")
        print(f"⏰ Timeout set to {timeout_seconds} seconds per detector")
        start_time = time.time()
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all 4 detection tasks
            future_to_version = {}
            for version_id, config in self.detector_configs.items():
                future = executor.submit(
                    self._safe_detect, 
                    version_id, 
                    config['class'], 
                    config['config'],
                    df.copy(),
                    config['name']
                )
                future_to_version[future] = version_id
            
            # Collect results as they complete
            for future in as_completed(future_to_version, timeout=timeout_seconds):
                version_id = future_to_version[future]
                try:
                    patterns, execution_time = future.result()
                    results[version_id] = {
                        'patterns': patterns,
                        'count': len(patterns),
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                    print(f"✅ {version_id} ({self.detector_configs[version_id]['name']}): {len(patterns)} patterns in {execution_time:.2f}s")
                    
                except Exception as e:
                    print(f"❌ {version_id} failed: {str(e)[:100]}...")
                    results[version_id] = {
                        'patterns': [],
                        'count': 0,
                        'execution_time': 0,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        total_time = time.time() - start_time
        print(f"⏱️  Total parallel execution time: {total_time:.2f} seconds")
        
        return self._combine_results(results, total_time)
    
    def _safe_detect(self, version_id, detector_class, config, df, name):
        """Safely run detection with error handling and timing - EXACTLY like your working code"""
        start_time = time.time()
        try:
            print(f"🔍 {version_id} ({name}) starting detection...")
            
            # Create detector instance in this thread
            detector = detector_class(config=config)
            
            # CRITICAL: Apply the EXACT same method replacement as your individual files
            # But only for non-micro detectors (micro detector doesn't have deduped method)
            if hasattr(detector, 'detect_cup_and_handle_deduped'):
                detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
                detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
                print(f"  📍 {version_id}: Applied method replacement (deduped)")
            else:
                print(f"  📍 {version_id}: Using original detection method (micro)")
            
            # Run detection - this should now work exactly like your individual runs
            patterns = detector.detect(df)
            
            # Ensure patterns is a list
            if patterns is None:
                patterns = []
            
            execution_time = time.time() - start_time
            print(f"✅ {version_id} completed successfully in {execution_time:.2f}s")
            
            return patterns, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {version_id} error after {execution_time:.2f}s: {str(e)[:100]}...")
            print(f"🔍 Full traceback for {version_id}:")
            traceback.print_exc()
            return [], execution_time
    
    def _combine_results(self, results, total_time):
        """Combine all results with comprehensive analysis"""
        combined = {
            'metadata': {
                'detection_timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'versions_run': len(results),
                'successful_versions': len([r for r in results.values() if r['status'] == 'success'])
            },
            'individual_results': {},
            'combined_patterns': [],
            'analysis': {}
        }
        
        # Process individual results
        all_patterns = []
        for version_id, result in results.items():
            combined['individual_results'][version_id] = {
                'count': result['count'],
                'execution_time': result['execution_time'],
                'status': result['status'],
                'patterns': result['patterns']
            }
            
            # Add error info if failed
            if 'error' in result:
                combined['individual_results'][version_id]['error'] = result['error']
            
            # Add source tracking to patterns
            for pattern in result['patterns']:
                pattern_copy = pattern.copy()
                pattern_copy['source_version'] = version_id
                pattern_copy['source_name'] = self.detector_configs[version_id]['name']
                pattern_copy['detection_timestamp'] = datetime.now().isoformat()
                all_patterns.append(pattern_copy)
        
        combined['combined_patterns'] = all_patterns
        
        # Comprehensive analysis
        combined['analysis'] = self._analyze_results(results, all_patterns)
        
        # Deduplicate patterns
        combined['unique_patterns'] = self._deduplicate_patterns(all_patterns)
        
        self._print_summary(combined)
        
        return combined
    
    def _analyze_results(self, results, all_patterns):
        """Analyze patterns across versions"""
        analysis = {
            'summary': {
                'total_patterns': len(all_patterns),
                'by_version': {v: r['count'] for v, r in results.items()},
                'execution_times': {v: r['execution_time'] for v, r in results.items()},
                'success_rate': len([r for r in results.values() if r['status'] == 'success']) / len(results)
            },
            'pattern_overlap': {},
            'quality_comparison': {}
        }
        
        # Pattern quality analysis
        if all_patterns:
            qualities = [p.get('quality_score', 0) for p in all_patterns if p.get('quality_score')]
            if qualities:
                analysis['quality_comparison'] = {
                    'avg_quality': sum(qualities) / len(qualities),
                    'max_quality': max(qualities),
                    'min_quality': min(qualities),
                    'by_version': {}
                }
                
                # Quality by version
                for version_id in results.keys():
                    version_qualities = [p.get('quality_score', 0) for p in all_patterns 
                                       if p.get('source_version') == version_id and p.get('quality_score')]
                    if version_qualities:
                        analysis['quality_comparison']['by_version'][version_id] = {
                            'avg': sum(version_qualities) / len(version_qualities),
                            'count': len(version_qualities)
                        }
        
        return analysis
    
    def _deduplicate_patterns(self, patterns, time_window_minutes=60):
        """Remove patterns that are too close in time"""
        if not patterns:
            return []
        
        # Sort by start time
        try:
            patterns.sort(key=lambda p: pd.to_datetime(p['peak_a']))
        except:
            print("⚠️ Warning: Could not sort patterns by time")
            return patterns
        
        unique = []
        
        for pattern in patterns:
            # Check if too close to existing patterns
            is_duplicate = False
            for existing in unique:
                try:
                    time_diff = abs((pd.to_datetime(pattern['peak_a']) - 
                                   pd.to_datetime(existing['peak_a'])).total_seconds() / 60)
                    if time_diff < time_window_minutes:
                        # Keep the higher quality pattern
                        if pattern.get('quality_score', 0) > existing.get('quality_score', 0):
                            unique.remove(existing)
                            unique.append(pattern)
                        is_duplicate = True
                        break
                except:
                    continue  # Skip if timestamp parsing fails
            
            if not is_duplicate:
                unique.append(pattern)
        
        return unique
    
    def _print_summary(self, combined):
        """Print comprehensive summary"""
        print(f"\n📊 COMPREHENSIVE RESULTS SUMMARY:")
        print(f"=" * 60)
        
        # Execution summary
        print(f"🚀 EXECUTION:")
        print(f"   Total time: {combined['metadata']['total_execution_time']:.2f}s")
        print(f"   Versions: {combined['metadata']['successful_versions']}/{combined['metadata']['versions_run']} successful")
        
        # Pattern counts
        print(f"\n📈 PATTERN COUNTS:")
        for version_id, result in combined['individual_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            name = self.detector_configs[version_id]['name']
            print(f"   {status_icon} {version_id} ({name}): {result['count']} patterns ({result['execution_time']:.2f}s)")
        
        print(f"\n🔗 COMBINED:")
        print(f"   Total patterns: {combined['analysis']['summary']['total_patterns']}")
        print(f"   After deduplication: {len(combined['unique_patterns'])}")
        
        # Quality analysis
        if 'quality_comparison' in combined['analysis'] and combined['analysis']['quality_comparison']:
            qc = combined['analysis']['quality_comparison']
            print(f"\n⭐ QUALITY ANALYSIS:")
            print(f"   Overall avg quality: {qc.get('avg_quality', 0):.1f}")
            print(f"   Range: {qc.get('min_quality', 0):.1f} - {qc.get('max_quality', 0):.1f}")
            
            if 'by_version' in qc:
                print(f"   By version:")
                for v, stats in qc['by_version'].items():
                    print(f"     {v}: {stats['avg']:.1f} avg ({stats['count']} patterns)")
        
        print(f"=" * 60)