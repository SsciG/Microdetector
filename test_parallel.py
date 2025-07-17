import pandas as pd
import sys
import os

# Test the parallel detection setup
def quick_test():
    print("🧪 Quick parallel detection test...")
    
    # Create sample data if main file doesn't exist
    if not os.path.exists("ES_timeframes/ES_15min.csv"):
        print("📊 Creating sample data for testing...")
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 4500 + pd.Series(range(1000)).apply(lambda x: x * 0.1 + (x % 10) * 0.5),
            'high': 4500 + pd.Series(range(1000)).apply(lambda x: x * 0.1 + (x % 10) * 0.5) + 2,
            'low': 4500 + pd.Series(range(1000)).apply(lambda x: x * 0.1 + (x % 10) * 0.5) - 2,
            'close': 4500 + pd.Series(range(1000)).apply(lambda x: x * 0.1 + (x % 10) * 0.5) + 1,
            'volume': 1000
        })
        os.makedirs("ES_timeframes", exist_ok=True)
        sample_data.to_csv("ES_timeframes/ES_15min.csv", index=False)
        print("✅ Sample data created!")
    
    # Test imports
    try:
        from multi_detector import MultiVersionDetector
        print("✅ MultiVersionDetector imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test with small dataset
    try:
        df = pd.read_csv("ES_timeframes/ES_15min.csv", parse_dates=[0])
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df.set_index('timestamp', inplace=True)
        df = df.tail(200)  # Small test dataset
        
        print(f"📊 Testing with {len(df)} bars...")
        
        multi_detector = MultiVersionDetector()
        results = multi_detector.detect_all_parallel(df, timeout_seconds=60)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"   Total patterns found: {results['analysis']['summary']['total_patterns']}")
        print(f"   Unique patterns: {len(results['unique_patterns'])}")
        print(f"   Versions successful: {results['metadata']['successful_versions']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Ready to run the full detection!")
        print("Run: python main.py")
    else:
        print("\n❌ Fix the issues above before running main.py")