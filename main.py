import pandas as pd
import json
import os
from multi_detector import MultiVersionDetector

def main():
    # Configuration
    input_path = "ES_timeframes/ES_15min.csv"
    output_dir = "results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Filter recent data (optional - adjust as needed)
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    original_length = len(df)
    df = df[df.index >= cutoff_date]
    print(f"📅 Filtered: {original_length} → {len(df)} bars (from {cutoff_date.date()})")
    
    # Create multi-detector and run
    print(f"\n🎯 Starting parallel detection with 3 versions...")
    multi_detector = MultiVersionDetector()
    results = multi_detector.detect_all_parallel(df, timeout_seconds=1200)  # 3 minutes per detector
    
    # Save comprehensive results
    output_file = os.path.join(output_dir, "multi_version_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to {output_file}")
    
    # Save individual version results for easier access
    combined_output = {
    "metadata": results['metadata'],
    "summary": results['analysis']['summary'],
    "all_patterns_by_version": {
        version_id: result['patterns'] 
        for version_id, result in results['individual_results'].items()
    },
    "unique_patterns": results['unique_patterns'],
    "quality_analysis": results['analysis'].get('quality_comparison', {}),
    "execution_stats": {
        version_id: {
            'count': result['count'],
            'execution_time': result['execution_time'],
            'status': result['status']
        } for version_id, result in results['individual_results'].items()
    }
}

    # Single comprehensive file
    single_file = os.path.join(output_dir, "all_cup_handle_results.json")
    with open(single_file, 'w') as f:
        json.dump(combined_output, f, indent=2, default=str)

    print(f"💾 All results saved to {single_file}")
    print(f"\n🔄 Converting patterns to labeler format...")
    labeler_compatible_patterns = []

    for pattern in results['unique_patterns']:
        labeler_pattern = {
            'id': f"pattern_{pattern['peak_a'].strftime('%Y%m%d_%H%M%S')}",  # ← FIXED
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
            'source_version': pattern.get('source_version', 'unknown'),
            'labeled_at': pd.Timestamp.now().isoformat(),
            'source': 'multi_algorithm_detection'
        }
        labeler_compatible_patterns.append(labeler_pattern)

    labeler_file = os.path.join(output_dir, "patterns_for_labeler.json")
    with open(labeler_file, 'w') as f:
        json.dump(labeler_compatible_patterns, f, indent=2, default=str)

    print(f"💾 Labeler-compatible patterns saved to {labeler_file}")


    
    # Create summary report
    create_summary_report(results, output_dir)
    
    print(f"\n🎉 All done! Check the '{output_dir}' folder for results.")

    

def create_summary_report(results, output_dir):
    """Create a human-readable summary report with proper Unicode handling"""
    report_file = os.path.join(output_dir, "summary_report.txt")
    
    # Use UTF-8 encoding and ASCII-safe characters for Windows compatibility
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("CUP & HANDLE DETECTION - MULTI-VERSION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Detection Time: {results['metadata']['detection_timestamp']}\n")
        f.write(f"Total Execution: {results['metadata']['total_execution_time']:.2f} seconds\n")
        f.write(f"Success Rate: {results['metadata']['successful_versions']}/{results['metadata']['versions_run']}\n\n")
        
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("-" * 30 + "\n")
        for version_id, result in results['individual_results'].items():
            # Use ASCII-safe status indicators
            status = "[SUCCESS]" if result['status'] == 'success' else "[FAILED]"
            f.write(f"{version_id.upper()}: {status}\n")
            f.write(f"  Patterns Found: {result['count']}\n")
            f.write(f"  Execution Time: {result['execution_time']:.2f}s\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
        
        f.write("COMBINED ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Patterns: {results['analysis']['summary']['total_patterns']}\n")
        f.write(f"After Deduplication: {len(results['unique_patterns'])}\n")
        
        if 'quality_comparison' in results['analysis']:
            qc = results['analysis']['quality_comparison']
            f.write(f"Average Quality: {qc.get('avg_quality', 0):.1f}\n")
            f.write(f"Quality Range: {qc.get('min_quality', 0):.1f} - {qc.get('max_quality', 0):.1f}\n")
        
        f.write("\n")
        
        # Pattern details
        if results['unique_patterns']:
            f.write("TOP 10 PATTERNS BY QUALITY:\n")
            f.write("-" * 30 + "\n")
            sorted_patterns = sorted(results['unique_patterns'], 
                                   key=lambda p: p.get('quality_score', 0), reverse=True)
            
            for i, pattern in enumerate(sorted_patterns[:10], 1):
                f.write(f"{i:2d}. {pattern.get('peak_a', 'N/A')} ")
                f.write(f"(Quality: {pattern.get('quality_score', 0):.1f}, ")
                f.write(f"Source: {pattern.get('source_version', 'Unknown')})\n")
        
        f.write("\n")
        f.write("PATTERN ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        # Add pattern metrics analysis
        if results['unique_patterns']:
            cup_depths = [p.get('cup_depth_pct', 0) for p in results['unique_patterns'] if p.get('cup_depth_pct')]
            handle_depths = [p.get('handle_depth_pct', 0) for p in results['unique_patterns'] if p.get('handle_depth_pct')]
            durations = []
            
            for p in results['unique_patterns']:
                try:
                    if p.get('peak_a') and p.get('breakout_e'):
                        start = pd.to_datetime(p['peak_a'])
                        end = pd.to_datetime(p['breakout_e'])
                        duration_hours = (end - start).total_seconds() / 3600
                        durations.append(duration_hours)
                except:
                    continue
            
            if cup_depths:
                f.write(f"Cup Depth Range: {min(cup_depths):.2f}% - {max(cup_depths):.2f}%\n")
                f.write(f"Average Cup Depth: {sum(cup_depths)/len(cup_depths):.2f}%\n")
            
            if handle_depths:
                f.write(f"Handle Depth Range: {min(handle_depths):.2f}% - {max(handle_depths):.2f}%\n")
                f.write(f"Average Handle Depth: {sum(handle_depths)/len(handle_depths):.2f}%\n")
            
            if durations:
                f.write(f"Pattern Duration Range: {min(durations):.1f}h - {max(durations):.1f}h\n")
                f.write(f"Average Duration: {sum(durations)/len(durations):.1f}h\n")
    
    print(f"📄 Summary report saved to {report_file}")

if __name__ == "__main__":
    main()