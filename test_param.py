"""
Cup & Handle Parameter Sweep Tool
Automatically tests different parameter combinations to find optimal settings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your detector
from detector_v2 import CupHandleDetector

class ParameterSweep:
    """
    Systematic parameter testing for Cup & Handle detection
    """
    
    def __init__(self, data_path, output_dir="parameter_sweep_results"):
        """Initialize with data and output directory"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = []
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔬 Parameter Sweep Tool Initialized")
        print(f"📂 Data: {data_path}")
        print(f"📁 Output: {output_dir}")
    
    def load_data(self):
        """Load and prepare data"""
        print(f"\n📊 Loading data...")
        
        self.df = pd.read_csv(self.data_path, parse_dates=[0])
        self.df.rename(columns={self.df.columns[0]: "timestamp"}, inplace=True)
        self.df.set_index('timestamp', inplace=True)
        
        # Filter to recent data (last 6 months) for faster testing
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
        original_length = len(self.df)
        self.df = self.df[self.df.index >= cutoff_date]
        
        print(f"   Original: {original_length:,} bars")
        print(f"   Filtered: {len(self.df):,} bars (last 6 months)")
        print(f"   Date range: {self.df.index[0]} to {self.df.index[-1]}")
    
    def define_parameter_ranges(self):
        """Define parameter combinations to test"""
        
        # 🎯 FOCUSED PARAMETER RANGES for Cup & Handle
        self.parameter_grid = {
            # RIM SYMMETRY - Your main concern
            'rim_height_tolerance_pct': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
            
            # CUP QUALITY FILTERS
            'min_cup_depth': [0.002, 0.005, 0.008, 0.01, 0.015, 0.02],
            'min_cup_roundness': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'min_quality_score': [20, 30, 40, 50, 60, 70],
            
            # DURATION LIMITS (in minutes)
            'min_cup_duration': [180, 360, 720, 1440],  # 3h, 6h, 12h, 24h
            'max_cup_duration': [1440, 2880, 4320, 7200],  # 24h, 48h, 72h, 120h
            
            # PROCESSING LIMITS
            'process_resistance_limit': [50, 100, 200],
        }
        
        # Calculate total combinations
        total_combinations = 1
        for param, values in self.parameter_grid.items():
            total_combinations *= len(values)
        
        print(f"\n🧪 Parameter Grid Defined:")
        for param, values in self.parameter_grid.items():
            print(f"   {param}: {len(values)} values {values}")
        print(f"\n⚠️  Total combinations: {total_combinations:,}")
        
        if total_combinations > 1000:
            print(f"🚨 WARNING: {total_combinations:,} combinations will take a long time!")
            print(f"   Consider using run_smart_sweep() for faster results.")
        
        return total_combinations
    
    def run_full_sweep(self, max_combinations=500):
        """Run systematic parameter sweep"""
        
        if not hasattr(self, 'parameter_grid'):
            total_combos = self.define_parameter_ranges()
        else:
            total_combos = 1
            for values in self.parameter_grid.values():
                total_combos *= len(values)
        
        print(f"\n🚀 Starting Full Parameter Sweep...")
        print(f"   Testing up to {min(max_combinations, total_combos)} combinations")
        
        # Generate all combinations
        param_names = list(self.parameter_grid.keys())
        param_values = list(self.parameter_grid.values())
        
        combinations_tested = 0
        successful_runs = 0
        
        for combo in product(*param_values):
            if combinations_tested >= max_combinations:
                print(f"🛑 Stopped at {max_combinations} combinations limit")
                break
            
            # Create config from combination
            config = dict(zip(param_names, combo))
            
            # Add base config
            config.update({
                'disable_deduplication': True,
                'skip_rim_adjustment': True,
                'breakout_search_duration': 480,
            })
            
            combinations_tested += 1
            
            try:
                # Run detection
                result = self._test_single_config(config, combinations_tested)
                if result:
                    self.results.append(result)
                    successful_runs += 1
                
                # Progress update
                if combinations_tested % 50 == 0:
                    print(f"   Progress: {combinations_tested}/{min(max_combinations, total_combos)} "
                          f"({combinations_tested/min(max_combinations, total_combos)*100:.1f}%) "
                          f"- {successful_runs} successful")
                    
            except Exception as e:
                print(f"   ❌ Config {combinations_tested} failed: {str(e)[:50]}...")
                continue
        
        print(f"\n✅ Sweep Complete!")
        print(f"   Combinations tested: {combinations_tested}")
        print(f"   Successful runs: {successful_runs}")
        print(f"   Results collected: {len(self.results)}")
    
    def run_smart_sweep(self):
        """Run intelligent parameter sweep focusing on promising areas"""
        
        print(f"\n🧠 Starting Smart Parameter Sweep...")
        print(f"   Testing key parameter combinations strategically")
        
        # STAGE 1: Test rim tolerance ranges (most important for you)
        print(f"\n🎯 STAGE 1: Rim Tolerance Sweep")
        rim_tolerances = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        
        for tolerance in rim_tolerances:
            config = {
                'rim_height_tolerance_pct': tolerance,
                'min_cup_depth': 0.01,
                'min_cup_roundness': 0.3,
                'min_quality_score': 40,
                'min_cup_duration': 360,
                'max_cup_duration': 2880,
                'process_resistance_limit': 100,
                'disable_deduplication': True,
                'skip_rim_adjustment': True,
            }
            
            result = self._test_single_config(config, f"rim_{tolerance}")
            if result:
                self.results.append(result)
        
        # STAGE 2: Test quality combinations for best rim tolerance
        best_rim_tolerance = self._find_best_rim_tolerance()
        print(f"\n🏆 STAGE 2: Quality Sweep (best rim tolerance: {best_rim_tolerance}%)")
        
        quality_combinations = [
            {'min_cup_depth': 0.005, 'min_quality_score': 30, 'min_cup_roundness': 0.2},
            {'min_cup_depth': 0.01, 'min_quality_score': 40, 'min_cup_roundness': 0.3},
            {'min_cup_depth': 0.015, 'min_quality_score': 50, 'min_cup_roundness': 0.4},
            {'min_cup_depth': 0.02, 'min_quality_score': 60, 'min_cup_roundness': 0.5},
        ]
        
        for i, quality_config in enumerate(quality_combinations):
            config = {
                'rim_height_tolerance_pct': best_rim_tolerance,
                'min_cup_duration': 360,
                'max_cup_duration': 2880,
                'process_resistance_limit': 100,
                'disable_deduplication': True,
                'skip_rim_adjustment': True,
            }
            config.update(quality_config)
            
            result = self._test_single_config(config, f"quality_{i+1}")
            if result:
                self.results.append(result)
        
        # STAGE 3: Test duration combinations for best overall
        print(f"\n⏰ STAGE 3: Duration Sweep")
        duration_combinations = [
            {'min_cup_duration': 180, 'max_cup_duration': 1440},   # Short
            {'min_cup_duration': 360, 'max_cup_duration': 2880},   # Medium  
            {'min_cup_duration': 720, 'max_cup_duration': 4320},   # Long
        ]
        
        best_config = self._find_best_overall_config()
        
        for i, duration_config in enumerate(duration_combinations):
            config = best_config.copy()
            config.update(duration_config)
            
            result = self._test_single_config(config, f"duration_{i+1}")
            if result:
                self.results.append(result)
        
        print(f"\n✅ Smart Sweep Complete! Found {len(self.results)} configurations")
    
    def _test_single_config(self, config, config_id):
        """Test a single configuration"""
        
        try:
            # Create detector with config
            detector = CupHandleDetector(config=config)
            
            # Run detection
            patterns = detector.detect(self.df.copy())
            
            # Calculate metrics
            metrics = self._calculate_metrics(patterns, config)
            
            # Add metadata
            result = {
                'config_id': config_id,
                'config': config,
                'patterns_found': len(patterns),
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Config {config_id} failed: {e}")
            return None
    
    def _calculate_metrics(self, patterns, config):
        """Calculate quality metrics for pattern results"""
        
        if len(patterns) == 0:
            return {
                'avg_quality_score': 0,
                'avg_rim_symmetry': 0,
                'avg_cup_depth': 0,
                'avg_cup_roundness': 0,
                'patterns_with_good_rims': 0,
                'patterns_with_good_quality': 0,
                'rim_tolerance_used': config.get('rim_height_tolerance_pct', 0),
            }
        
        # Extract pattern metrics
        quality_scores = [p.get('quality_score', 0) for p in patterns]
        rim_diffs = [p.get('rim_diff_pct', 0) for p in patterns]
        cup_depths = [p.get('cup_depth_pct', 0) for p in patterns]
        cup_roundness = [p.get('cup_roundness', 0) for p in patterns]
        
        # Calculate aggregate metrics
        tolerance = config.get('rim_height_tolerance_pct', 2.0)
        
        return {
            'avg_quality_score': np.mean(quality_scores),
            'avg_rim_symmetry': np.mean(rim_diffs),
            'avg_cup_depth': np.mean(cup_depths),
            'avg_cup_roundness': np.mean(cup_roundness),
            'patterns_with_good_rims': sum(1 for r in rim_diffs if r <= tolerance),
            'patterns_with_good_quality': sum(1 for q in quality_scores if q >= 60),
            'rim_tolerance_used': tolerance,
            'max_rim_diff': max(rim_diffs) if rim_diffs else 0,
            'min_rim_diff': min(rim_diffs) if rim_diffs else 0,
        }
    
    def _find_best_rim_tolerance(self):
        """Find best rim tolerance from initial results"""
        rim_results = [r for r in self.results if str(r['config_id']).startswith('rim_')]
        
        if not rim_results:
            return 0.5  # Default
        
        # Find tolerance with best balance of patterns found and quality
        best_score = 0
        best_tolerance = 0.5
        
        for result in rim_results:
            # Score = patterns found * avg quality * rim symmetry bonus
            patterns = result['patterns_found']
            quality = result['avg_quality_score']
            rim_bonus = 100 / (result['avg_rim_symmetry'] + 1)  # Reward low rim differences
            
            score = patterns * quality * rim_bonus / 1000  # Normalize
            
            if score > best_score:
                best_score = score
                best_tolerance = result['config']['rim_height_tolerance_pct']
        
        print(f"   🏆 Best rim tolerance: {best_tolerance}% (score: {best_score:.2f})")
        return best_tolerance
    
    def _find_best_overall_config(self):
        """Find best overall configuration so far"""
        if not self.results:
            return {'rim_height_tolerance_pct': 0.5, 'min_cup_depth': 0.01}
        
        # Score based on multiple factors
        best_score = 0
        best_config = None
        
        for result in self.results:
            patterns = result['patterns_found']
            quality = result['avg_quality_score']
            rim_symmetry = result['avg_rim_symmetry']
            
            # Composite score
            score = patterns * quality / (rim_symmetry + 1)
            
            if score > best_score:
                best_score = score
                best_config = result['config']
        
        return best_config or {'rim_height_tolerance_pct': 0.5}
    
    def generate_analysis(self):
        """Generate comprehensive analysis of results"""
        
        if not self.results:
            print("❌ No results to analyze!")
            return
        
        print(f"\n📊 Generating Analysis...")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Save raw results
        results_file = os.path.join(self.output_dir, "sweep_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"   💾 Raw results: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(df_results)
        
        # Generate visualizations
        self._generate_visualizations(df_results)
        
        # Find and save best configurations
        self._find_best_configurations(df_results)
    
    def _generate_summary_report(self, df_results):
        """Generate text summary report"""
        
        report_file = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("CUP & HANDLE PARAMETER SWEEP SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data: {self.data_path}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configurations tested: {len(df_results)}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total patterns found: {df_results['patterns_found'].sum()}\n")
            f.write(f"  Average patterns per config: {df_results['patterns_found'].mean():.1f}\n")
            f.write(f"  Best single config found: {df_results['patterns_found'].max()} patterns\n")
            f.write(f"  Configs with 0 patterns: {(df_results['patterns_found'] == 0).sum()}\n\n")
            
            # Rim symmetry analysis
            f.write("RIM SYMMETRY ANALYSIS:\n")
            rim_data = df_results[df_results['patterns_found'] > 0]
            if len(rim_data) > 0:
                f.write(f"  Best rim symmetry achieved: {rim_data['avg_rim_symmetry'].min():.3f}%\n")
                f.write(f"  Average rim asymmetry: {rim_data['avg_rim_symmetry'].mean():.3f}%\n")
                f.write(f"  Configs with <0.5% rim diff: {(rim_data['avg_rim_symmetry'] < 0.5).sum()}\n")
            
            # Quality analysis
            f.write("\nQUALITY ANALYSIS:\n")
            if len(rim_data) > 0:
                f.write(f"  Highest quality score: {rim_data['avg_quality_score'].max():.1f}\n")
                f.write(f"  Average quality score: {rim_data['avg_quality_score'].mean():.1f}\n")
            
            # Parameter impact analysis
            f.write("\nPARAMETER IMPACT:\n")
            
            # Rim tolerance impact
            if 'rim_height_tolerance_pct' in df_results.columns:
                rim_impact = df_results.groupby('rim_tolerance_used')['patterns_found'].mean()
                f.write("  Rim Tolerance vs Pattern Count:\n")
                for tolerance, avg_patterns in rim_impact.items():
                    f.write(f"    {tolerance:>5.2f}%: {avg_patterns:>6.1f} patterns average\n")
        
        print(f"   📄 Summary report: {report_file}")
    
    def _generate_visualizations(self, df_results):
        """Generate visualization plots"""
        
        print(f"   🎨 Generating visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. RIM TOLERANCE IMPACT
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cup & Handle Parameter Sweep Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Rim Tolerance vs Pattern Count
        if len(df_results) > 1:
            # Group by rim tolerance if multiple values tested
            tolerance_groups = df_results.groupby('rim_tolerance_used').agg({
                'patterns_found': ['mean', 'sum', 'count'],
                'avg_rim_symmetry': 'mean'
            }).round(2)
            
            tolerance_groups.columns = ['avg_patterns', 'total_patterns', 'configs_tested', 'avg_rim_diff']
            
            axes[0,0].bar(tolerance_groups.index, tolerance_groups['avg_patterns'])
            axes[0,0].set_title('Rim Tolerance vs Average Patterns Found')
            axes[0,0].set_xlabel('Rim Height Tolerance (%)')
            axes[0,0].set_ylabel('Average Patterns Found')
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Quality Score Distribution
        quality_data = df_results[df_results['patterns_found'] > 0]['avg_quality_score']
        if len(quality_data) > 0:
            axes[0,1].hist(quality_data, bins=20, alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Quality Score Distribution')
            axes[0,1].set_xlabel('Average Quality Score')
            axes[0,1].set_ylabel('Number of Configurations')
            axes[0,1].axvline(quality_data.mean(), color='red', linestyle='--', 
                            label=f'Mean: {quality_data.mean():.1f}')
            axes[0,1].legend()
        
        # Plot 3: Rim Symmetry vs Quality
        pattern_data = df_results[df_results['patterns_found'] > 0]
        if len(pattern_data) > 0:
            scatter = axes[1,0].scatter(pattern_data['avg_rim_symmetry'], 
                                      pattern_data['avg_quality_score'],
                                      c=pattern_data['patterns_found'], 
                                      cmap='viridis', alpha=0.7)
            axes[1,0].set_title('Rim Symmetry vs Quality Score')
            axes[1,0].set_xlabel('Average Rim Difference (%)')
            axes[1,0].set_ylabel('Average Quality Score')
            plt.colorbar(scatter, ax=axes[1,0], label='Patterns Found')
        
        # Plot 4: Parameter Correlation Heatmap
        numeric_cols = df_results.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:
            corr_matrix = df_results[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1,1], fmt='.2f')
            axes[1,1].set_title('Parameter Correlation Matrix')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "parameter_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Main analysis plot: {plot_file}")
        
        # 2. DETAILED RIM TOLERANCE ANALYSIS
        if len(df_results) > 5:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Detailed rim tolerance plot
            rim_detailed = df_results.groupby('rim_tolerance_used').agg({
                'patterns_found': 'sum',
                'avg_rim_symmetry': 'mean',
                'avg_quality_score': 'mean'
            })
            
            ax1.plot(rim_detailed.index, rim_detailed['patterns_found'], 'o-', label='Total Patterns')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(rim_detailed.index, rim_detailed['avg_quality_score'], 's-', 
                         color='orange', label='Avg Quality')
            
            ax1.set_xlabel('Rim Height Tolerance (%)')
            ax1.set_ylabel('Total Patterns Found', color='blue')
            ax1_twin.set_ylabel('Average Quality Score', color='orange')
            ax1.set_title('Rim Tolerance: Quantity vs Quality Trade-off')
            ax1.grid(True, alpha=0.3)
            
            # Best configurations scatter
            ax2.scatter(df_results['patterns_found'], df_results['avg_quality_score'],
                       alpha=0.6, s=50)
            ax2.set_xlabel('Patterns Found')
            ax2.set_ylabel('Average Quality Score')
            ax2.set_title('Pattern Count vs Quality Trade-off')
            ax2.grid(True, alpha=0.3)
            
            # Highlight best configs
            top_configs = df_results.nlargest(5, 'patterns_found')
            ax2.scatter(top_configs['patterns_found'], top_configs['avg_quality_score'],
                       color='red', s=100, alpha=0.8, label='Top 5 Configs')
            ax2.legend()
            
            plt.tight_layout()
            detailed_plot = os.path.join(self.output_dir, "detailed_analysis.png")
            plt.savefig(detailed_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Detailed analysis plot: {detailed_plot}")
    
    def _find_best_configurations(self, df_results):
        """Identify and save best configurations"""
        
        best_configs_file = os.path.join(self.output_dir, "best_configurations.json")
        
        # Find top configurations by different criteria
        best_configs = {}
        
        # 1. Most patterns found
        most_patterns = df_results.loc[df_results['patterns_found'].idxmax()]
        best_configs['most_patterns'] = {
            'description': 'Configuration that found the most patterns',
            'patterns_found': int(most_patterns['patterns_found']),
            'avg_quality': float(most_patterns['avg_quality_score']),
            'avg_rim_diff': float(most_patterns['avg_rim_symmetry']),
            'config': most_patterns['config']
        }
        
        # 2. Best rim symmetry (among configs with patterns)
        pattern_configs = df_results[df_results['patterns_found'] > 0]
        if len(pattern_configs) > 0:
            best_symmetry = pattern_configs.loc[pattern_configs['avg_rim_symmetry'].idxmin()]
            best_configs['best_rim_symmetry'] = {
                'description': 'Configuration with best rim symmetry',
                'patterns_found': int(best_symmetry['patterns_found']),
                'avg_quality': float(best_symmetry['avg_quality_score']),
                'avg_rim_diff': float(best_symmetry['avg_rim_symmetry']),
                'config': best_symmetry['config']
            }
        
        # 3. Best quality score
        if len(pattern_configs) > 0:
            best_quality = pattern_configs.loc[pattern_configs['avg_quality_score'].idxmax()]
            best_configs['best_quality'] = {
                'description': 'Configuration with highest quality scores',
                'patterns_found': int(best_quality['patterns_found']),
                'avg_quality': float(best_quality['avg_quality_score']),
                'avg_rim_diff': float(best_quality['avg_rim_symmetry']),
                'config': best_quality['config']
            }
        
        # 4. Best balanced (composite score)
        if len(pattern_configs) > 0:
            # Calculate composite score: patterns * quality / (rim_diff + 1)
            pattern_configs = pattern_configs.copy()
            pattern_configs['composite_score'] = (
                pattern_configs['patterns_found'] * 
                pattern_configs['avg_quality_score'] / 
                (pattern_configs['avg_rim_symmetry'] + 1)
            )
            
            best_balanced = pattern_configs.loc[pattern_configs['composite_score'].idxmax()]
            best_configs['best_balanced'] = {
                'description': 'Best balanced configuration (patterns × quality ÷ rim_diff)',
                'patterns_found': int(best_balanced['patterns_found']),
                'avg_quality': float(best_balanced['avg_quality_score']),
                'avg_rim_diff': float(best_balanced['avg_rim_symmetry']),
                'composite_score': float(best_balanced['composite_score']),
                'config': best_balanced['config']
            }
        
        # Save best configurations
        with open(best_configs_file, 'w') as f:
            json.dump(best_configs, f, indent=2, default=str)
        
        print(f"   🏆 Best configurations: {best_configs_file}")
        
        # Print summary
        print(f"\n🏆 BEST CONFIGURATIONS FOUND:")
        for name, config_info in best_configs.items():
            print(f"\n   {name.upper().replace('_', ' ')}:")
            print(f"     Patterns: {config_info['patterns_found']}")
            print(f"     Quality: {config_info['avg_quality']:.1f}")
            print(f"     Rim Diff: {config_info['avg_rim_diff']:.3f}%")
            print(f"     Key params: rim_tol={config_info['config']['rim_height_tolerance_pct']}%, "
                  f"min_depth={config_info['config']['min_cup_depth']}")


def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "ES_timeframes/ES_15min.csv"
    OUTPUT_DIR = "parameter_sweep_results"
    
    print("🔬 CUP & HANDLE PARAMETER SWEEP TOOL")
    print("=" * 50)
    
    # Initialize sweep
    sweep = ParameterSweep(DATA_PATH, OUTPUT_DIR)
    
    # Load data
    sweep.load_data()
    
    # Ask user for sweep type
    print(f"\n🤔 Choose sweep type:")
    print(f"   1. Smart Sweep (Recommended) - ~30 configurations, 5-10 minutes")
    print(f"   2. Full Sweep - All combinations, 30+ minutes")
    print(f"   3. Custom Sweep - Define your own ranges")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Smart sweep
        sweep.run_smart_sweep()
    elif choice == "2":
        # Full sweep
        total_combos = sweep.define_parameter_ranges()
        max_combos = int(input(f"Max combinations to test (recommended: 500): ") or 500)
        sweep.run_full_sweep(max_combos)
    elif choice == "3":
        # Custom sweep - let user define ranges
        print("Custom sweep not implemented yet. Using smart sweep.")
        sweep.run_smart_sweep()
    else:
        print("Invalid choice. Using smart sweep.")
        sweep.run_smart_sweep()
    
    # Generate analysis
    sweep.generate_analysis()
    
    print(f"\n✅ Parameter Sweep Complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"🔍 Check best_configurations.json for optimal settings")


if __name__ == "__main__":
    main()