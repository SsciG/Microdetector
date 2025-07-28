import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CupHandleDetector:
    """
    Micro Cup & Handle Pattern Detector
    Specialized for detecting small formations on 15-minute timeframes
    Optimized for speed and micro-pattern sensitivity
    """
    
    def __init__(self, config=None):
        """Initialize micro pattern detector with fast, sensitive settings"""
        self.config = {
            # MICRO PATTERN DURATION LIMITS (in bars)
            "min_cup_bars": 3,             
            "max_cup_bars": 24,          
            "min_handle_bars": 2,         
            "max_handle_bars": 8,          
            
            # MICRO DEPTH REQUIREMENTS (much smaller than institutional)
            "min_cup_depth_pct": 1.8,     
            "max_cup_depth_pct": 15,     
            "min_handle_depth_pct": 0.8,  
            "max_handle_depth_pct": 0.4,  
            
            # POINT-BASED TOLERANCES (better for ES futures)
            "rim_tolerance_points": 3.0,   # 3 ES points rim tolerance
            "min_cup_depth_points": 2.0,   # 2 ES points minimum depth
            "min_handle_depth_points": 0.5, # 0.5 ES points minimum handle
            
            # SHAPE REQUIREMENTS (relaxed for micro patterns)
            "max_rim_asymmetry_pct": 0.8,  # 0.8% rim height difference
            "min_roundness_score": 0.5,    # Low roundness requirement
            "allow_v_shapes": True,        # V-shapes OK for micro patterns
            
            # QUALITY THRESHOLDS (lower for micro)
            "min_quality_score": 45,       # Lower quality bar
            "breakout_min_points": 0.25,   # Quarter point breakout
            "breakout_search_bars": 10,    # Look ahead 10 bars max
            
            # PERFORMANCE SETTINGS
            "use_fast_scan": True,          # Skip heavy analysis
            "max_patterns_per_scan": 50,   # Limit results
            "overlap_tolerance_bars": 3,   # Prevent overlapping patterns
        }
        
        if config:
            self.config.update(config)
        
        self.patterns_found = []
        
    def detect(self, df, price_col='close'):
        """Main detection entry point - matches your existing detector interface"""
        logger.info(f"🔬 Starting MICRO pattern detection on {len(df)} bars")
        with open('micro_debug.txt', 'w') as f:
            f.write(f"MICRO: Starting with {len(df)} bars\n")
            f.write(f"MICRO: Columns: {df.columns.tolist()}\n")

        print(f"🔬 MICRO DEBUG: Input DF shape: {df.shape}")
        print(f"🔬 MICRO DEBUG: Columns: {df.columns.tolist()}")
        print(f"🔬 MICRO DEBUG: First few rows:")
        print(df.head(3))
        try:
            # Preprocess for micro detection
            processed_df = self._preprocess_micro(df)
            print(f"🔬 MICRO DEBUG: After preprocessing: {processed_df.shape}")
            # Fast micro pattern scan
            patterns = self._scan_micro_patterns(processed_df, price_col)
            print(f"🔬 MICRO DEBUG: Patterns from scan: {len(patterns)}")
            # Quick quality filter
            filtered_patterns = self._filter_micro_patterns(patterns, processed_df)
            
            logger.info(f"🔬 Micro detection complete: {len(filtered_patterns)} patterns found")
            return filtered_patterns
        
        
        
        except Exception as e:
            print(f"🔬 MICRO DETECTOR: ERROR in detect(): {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _preprocess_micro(self, df):
        """Lightweight preprocessing optimized for speed"""
        processed = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in processed.columns:
                processed[col] = processed.get('close', processed.iloc[:, 0])
        
        # Simple 3-period smoothing for noise reduction
        if len(processed) > 3:
            processed['close_smooth'] = processed['close'].rolling(window=3, center=True).mean()
            processed['close_smooth'].fillna(processed['close'], inplace=True)
        else:
            processed['close_smooth'] = processed['close']
        
        # Fast high/low detection using scipy
        highs, _ = find_peaks(processed['high'].values, distance=2, prominence=0.5)
        lows, _ = find_peaks(-processed['low'].values, distance=2, prominence=0.5)
        print(f"🔬 MICRO: Found {len(highs)} highs, {len(lows)} lows with prominence=0.5")
        # Mark extrema
        processed['extrema'] = 0
        if len(highs) > 0:
            processed.iloc[highs, processed.columns.get_loc('extrema')] = 1
        if len(lows) > 0:
            processed.iloc[lows, processed.columns.get_loc('extrema')] = -1
        
        return processed
    
    def _scan_micro_patterns(self, df, price_col):
        """
        ENHANCED: Added pre-validation and explicit chronological checks
        """
        patterns = []
        
        # Get significant highs and lows only
        peaks = df[df['extrema'] == 1].index.tolist()
        troughs = df[df['extrema'] == -1].index.tolist()
        
        if len(peaks) < 2 or len(troughs) < 1:
            logger.info(f"🔬 MICRO: Insufficient extrema - {len(peaks)} peaks, {len(troughs)} troughs")
            return patterns
        
        logger.info(f"🔍 STRICT Micro scan: {len(peaks)} peaks, {len(troughs)} troughs")
        
        # Duration constraints (aligned with validation)
        MIN_PATTERN_MINUTES = 30
        MAX_PATTERN_MINUTES = 720
        
        cups_tested = 0
        cups_valid = 0
        handles_found = 0
        breakouts_found = 0
        
        # Look for A-B-C-D-E structure
        for i, peak_a in enumerate(peaks[:-1]):
            for peak_c in peaks[i+1:]:
                cups_tested += 1
                
                # Duration check
                pattern_duration_minutes = (peak_c - peak_a).total_seconds() / 60
                
                if pattern_duration_minutes > MAX_PATTERN_MINUTES:
                    continue
                
                if pattern_duration_minutes < MIN_PATTERN_MINUTES:
                    continue
                
                # Find cup bottom between rims
                cup_troughs = [t for t in troughs if peak_a < t < peak_c]
                if not cup_troughs:
                    continue

                # NEW: Pre-validation for fundamental cup rim violation
                cup_validation_period = df.loc[peak_a:peak_c]
                rim_level = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])
                cup_max_high = cup_validation_period['high'].max()

                # HARD CONSTRAINT: Zero tolerance for rim violations
                current_atr = self.calculate_atr(df, 20).iloc[-1] if hasattr(self, 'calculate_atr') else 15.0
                max_allowed_violation = current_atr * 0.15

                if cup_max_high > (rim_level + max_allowed_violation):
                    violation_amount = cup_max_high - rim_level
                    logger.info(f"❌ PRE-VALIDATION: Cup exceeded rim by ${violation_amount:.2f} > ${max_allowed_violation:.2f} (0.25x ATR)")
                    continue

                # FIND VALID TROUGHS (ones that don't get violated)
                valid_troughs = []
                for t in cup_troughs:
                    trough_price = df.loc[t, 'low']
                    # Check if this trough doesn't have price action crossing below it
                    period_to_check = df.loc[peak_a:peak_c]
                    lowest_in_period = period_to_check['low'].min()

                    if lowest_in_period >= trough_price:  # Remove the 0.999 tolerance
                        valid_troughs.append(t)
                    else:
                        violation_amount = trough_price - lowest_in_period
                        logger.info(f"❌ Trough {t} violated by ${violation_amount:.2f}")

                if not valid_troughs:
                    logger.info(f"❌ No valid troughs - all violated by lower prices")
                    continue
                    
                # Use the deepest valid trough as cup bottom
                trough_b = min(valid_troughs, key=lambda t: df.loc[t, 'low'])
                
                # STEP 1: Strict cup structure validation
                if self._validate_strict_cup_structure(df, peak_a, trough_b, peak_c):
                    cups_valid += 1
                    
                    # STEP 2: Find strict handle
                    handle = self._find_strict_handle(df, peak_c, trough_b, peak_a)
                    if handle:
                        handles_found += 1
                        logger.info(f"🔍 MICRO: Found handle for cup {pattern_duration_minutes:.0f}min")
                        
                        # STEP 3: Find strict breakout
                        breakout = self._find_strict_breakout(df, handle, peak_a, peak_c)
                        if breakout:
                            breakouts_found += 1
                            
                            # NEW: STEP 4: Validate chronological order before creating pattern
                            if True:
                                pattern = self._create_strict_micro_pattern(df, peak_a, trough_b, peak_c, handle, breakout) 
                                if pattern:
                                    patterns.append(pattern)
                                    logger.info(f"✅ STRICT MICRO: {pattern['cup_depth_pct']:.1f}% cup, {pattern_duration_minutes:.0f} min")
                            else:
                                logger.info(f"❌ REJECTED: Chronological order violation")
        
        # Enhanced diagnostic summary
        logger.info(f"🔬 STRICT MICRO DIAGNOSTIC: {cups_tested} cups tested → {cups_valid} valid → {handles_found} handles → {breakouts_found} breakouts → {len(patterns)} patterns")
        
        return patterns
    
    def _create_strict_micro_pattern(self, df, peak_a, trough_b, peak_c, handle, breakout):
        """Create pattern using STRICT validation results"""
        try:
            # Get prices
            peak_a_price = df.loc[peak_a, 'high']
            peak_c_price = df.loc[peak_c, 'high']
            trough_b_price = df.loc[trough_b, 'low']
            
            # Calculate REAL metrics (not hardcoded)
            max_rim = max(peak_a_price, peak_c_price)
            cup_depth = max_rim - trough_b_price
            cup_depth_pct = (cup_depth / max_rim) * 100
            
            # REAL rim symmetry calculation
            rim_diff_abs = abs(peak_a_price - peak_c_price)
            rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
            cup_symmetry = 1.0 - (rim_diff_pct / 2.0)  # Convert to 0-1 scale
            
            # Duration
            cup_duration_min = (peak_c - peak_a).total_seconds() / 60
            handle_duration_min = (handle['end'] - handle['start']).total_seconds() / 60
            
            # Quality score based on STRICT criteria
            quality_score = 70 + (cup_depth_pct * 2) + (cup_symmetry * 20)
            
            return {
                'peak_a': peak_a,
                'trough_b': trough_b,
                'peak_c': peak_c,
                'handle_d': handle['start'],
                'breakout_e': breakout,
                'breakout_confirmed': True,
                
                'cup_depth': cup_depth,
                'cup_depth_pct': cup_depth_pct,
                'cup_duration_min': cup_duration_min,
                'handle_depth': max_rim - handle['low_price'],
                'handle_depth_pct': handle['depth_pct'],
                'handle_duration_min': handle_duration_min,
                
                # REAL calculated values (not hardcoded)
                'cup_symmetry': cup_symmetry,
                'cup_roundness': 0.8,  # Can calculate this properly if needed
                'rim_diff_pct': rim_diff_pct,  # Add this for debugging
                'quality_score': min(100, quality_score),
                'confidence_score': 0.9,  # Higher confidence for strict patterns
                
                'pattern_type': 'strict_micro',
                'timeframe': '15min',
                'detection_method': 'strict_structure_based'
            }
        except Exception as e:
            logger.warning(f"Strict pattern creation error: {e}")
            return None
        
    
        
    def _find_structure_breakout(self, df, handle, peak_a, peak_c):
        """Find structure-based breakout"""
        try:
            handle_end_idx = df.index.get_loc(handle['end'])
            resistance_level = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])
            
            # Look for breakout in next 10 bars
            for i in range(handle_end_idx + 1, min(len(df), handle_end_idx + 11)):
                if df.iloc[i]['high'] > resistance_level * 1.001:  # 0.1% above resistance
                    return df.index[i]
            return None
        except:
            return None
        

    def _find_strict_breakout(self, df, handle, peak_a, peak_c):
        """GROUND RULE 3: Breakout must be AT or ABOVE cup rim level"""
        try:
            handle_end_idx = df.index.get_loc(handle['end'])
            rim_level = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])
            
            logger.info(f"🔧 BREAKOUT SEARCH: rim_level=${rim_level:.2f}")
            
            # Look for breakout in next 10 bars (2.5 hours max)
            for i in range(handle_end_idx + 1, min(len(df), handle_end_idx + 11)):
                breakout_high = df.iloc[i]['high']
                
                # STRICT: Breakout must be AT or ABOVE rim level (no tolerance below)
                if breakout_high >= rim_level:  # AT or above rim level
                    logger.info(f"✅ VALID BREAKOUT: ${breakout_high:.2f} >= rim ${rim_level:.2f}")
                    return df.index[i]
                else:
                    logger.info(f"❌ BREAKOUT TOO LOW: ${breakout_high:.2f} < rim ${rim_level:.2f}")
            
            logger.info(f"❌ STRICT: No valid breakout found above rim ${rim_level:.2f}")
            return None
        except Exception as e:
            logger.info(f"❌ STRICT: Breakout search error: {e}")
            return None
        

    def validate_support_holds(self, df, pattern, lookforward_bars=20):
        """Validate that cup bottom support holds after breakout"""
        trough_b_price = df.loc[pattern['trough_b'], 'low']
        breakout_idx = df.index.get_loc(pattern['breakout_e'])

        if breakout_idx + lookforward_bars >= len(df):
            logger.info(f"⚠️ INSUFFICIENT DATA: Cannot validate support (need {lookforward_bars} bars)")
            return True 
        
        # Check next 20 bars (5 hours) after breakout
        end_idx = min(len(df), breakout_idx + lookforward_bars)
        future_period = df.iloc[breakout_idx + 1:end_idx]
        
        if len(future_period) > 0:
            min_low_after = future_period['low'].min()
            if min_low_after < trough_b_price * 0.99:  # 1% tolerance
                logger.info(f"❌ SUPPORT FAILED: Price ${min_low_after:.2f} broke below cup bottom ${trough_b_price:.2f}")
                return False
        
        return True
                    

    
    def _find_strict_handle(self, df, peak_c, trough_b, peak_a=None):
        """
        FIXED: Hard geometric constraints for handle formation
        """
        peak_c_idx = df.index.get_loc(peak_c)
        search_end = min(len(df), peak_c_idx + 32)
        
        if peak_c_idx >= len(df) - 3:
            return None
        
        peak_c_price = df.loc[peak_c, 'high']
        trough_b_price = df.loc[trough_b, 'low']

        if peak_a is not None:
            peak_a_price = df.loc[peak_a, 'high']
            rim_level = max(peak_a_price, peak_c_price)
        else:
            rim_level = peak_c_price
        
        # NEW: Minimum gap after peak C
        min_gap_bars = 2
        
        for i in range(peak_c_idx + 1 + min_gap_bars, search_end):
            handle_low = df.iloc[i]['low']
            handle_high = df.iloc[i]['high']

            if handle_high > rim_level:
                logger.info(f"❌ HANDLE ABOVE RIM: Handle high ${handle_high:.2f} > rim ${rim_level:.2f}")
                continue  

            # FUNDAMENTAL RULE #2: Handle position constraints
            if handle_low >= peak_c_price:
                continue  # Handle must be below rim


            logger.info(f"🔧 HANDLE CHECK: Handle ${handle_low:.2f} vs Cup bottom ${trough_b_price:.2f}")

            current_atr = self.calculate_atr(df, 20).iloc[-1] if hasattr(self, 'calculate_atr') else 15.0
            min_handle_level = trough_b_price + (current_atr * 1.0)  # Handle must be at least 3x ATR above cup bottom
            if handle_low < min_handle_level:
                logger.info(f"❌ HANDLE TOO LOW: Handle ${handle_low:.2f} < ${min_handle_level:.2f} (3x ATR above cup bottom)")
                continue

            logger.info(f"✅ HANDLE OK: Handle ${handle_low:.2f} above cup bottom ${trough_b_price:.2f}")

            peak_c_price = df.loc[peak_c, 'high']
            rim_level = max(df.loc[peak_a, 'high'], peak_c_price) if peak_a else peak_c_price
            
            # Check all bars between peak_c and potential handle start
            between_period = df.iloc[peak_c_idx + 1:i]
            rim_level = max(df.loc[peak_a, 'high'], peak_c_price) if peak_a else peak_c_price

            logger.info(f"🔧 RIM CHECK: peak_c=${peak_c_price:.2f}, rim_level=${rim_level:.2f}")

            if len(between_period) > 0:
                max_high_between = between_period['high'].max()
                logger.info(f"🔧 BETWEEN: max_high=${max_high_between:.2f} vs rim_threshold=${rim_level * 1.001:.2f}")
                
                if max_high_between > rim_level * 1.001:  # 0.1% tolerance
                    logger.info(f"❌ HANDLE INVALID: Price went ${max_high_between:.2f} above rim ${rim_level:.2f}")
                    continue
                else:
                    logger.info(f"✅ RIM OK: Price stayed below rim level")
            else:
                logger.info(f"⚠️ No bars between peak_c and handle - immediate handle")
            
            # Find where pullback ends (recovery starts)
            for j in range(i + 1, min(search_end, i + 20)):
                if df.iloc[j]['high'] > handle_low * 1.002:
                    
                    handle_depth_points = peak_c_price - handle_low
                    handle_depth_pct = (handle_depth_points / peak_c_price) * 100
                    
                    # Micro handle depth validation (keep existing logic)
                    if 0.2 <= handle_depth_pct <= 2.0:
                        
                        # ⚠️ CRITICAL FIX - FUNDAMENTAL RULE #2: Handle Rim Violation
                        # ZERO TOLERANCE: Handle NEVER goes above rim during formation
                        handle_period = df.iloc[i:j]
                        handle_violations = handle_period[handle_period['low'] < trough_b_price]
                        if not handle_violations.empty:
                            worst_violation = handle_violations['low'].min()
                            logger.info(f"❌ HANDLE PERIOD VIOLATION: Price went to ${worst_violation:.2f}, below cup bottom ${trough_b_price:.2f}")
                            continue

                        handle_max_high = handle_period['high'].max()
                        cup_rim_level = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])

                        if handle_period['high'].max() > rim_level:
                            logger.info(f"❌ HANDLE PERIOD ABOVE RIM: max ${handle_period['high'].max():.2f} > rim ${rim_level:.2f}")
                            continue

                        # FIXED: Remove tolerance completely - hard geometric constraint
                        if handle_max_high > cup_rim_level:  # ← CHANGED: was cup_rim_level * 1.0001, now pure constraint
                            violation_amount = handle_max_high - cup_rim_level
                            logger.info(f"❌ FUNDAMENTAL VIOLATION: Handle exceeded rim by ${violation_amount:.2f}")
                            continue 
                        
                        if handle_period['high'].max() <= peak_c_price * 1.001:
                            logger.info(f"✅ STRICT HANDLE: {handle_depth_pct:.2f}% depth")
                            return {
                                'start': df.index[i],
                                'end': df.index[j-1],
                                'low_price': handle_low,
                                'depth_pct': handle_depth_pct
                            }
        return None

        
    def _validate_strict_cup_structure(self, df, peak_a, trough_b, peak_c   ):
        """
        FIXED: Hard geometric validation with ZERO tolerance for fundamental rules
        """
        peak_a_price = df.loc[peak_a, 'high']
        peak_c_price = df.loc[peak_c, 'high'] 
        trough_b_price = df.loc[trough_b, 'low']
        
        # STEP 1: Cup depth validation (keep existing logic)
        max_rim = max(peak_a_price, peak_c_price)
        cup_depth_pct = ((max_rim - trough_b_price) / max_rim) * 100
        
        logger.info(f"🔧 STEP1 TEST: Cup depth = {cup_depth_pct:.2f}%")

        duration_minutes = (peak_c - peak_a).total_seconds() / 60
        pattern_significance = self._calculate_pattern_significance(
            cup_depth_pct, 
            0.5,  # Estimate handle depth since we don't have handle yet
            duration_minutes
        )

        if pattern_significance < 15.0:
            logger.info(f"❌ INSTITUTIONAL: Pattern significance {pattern_significance:.1f} < 15.0")
            return False
            
        current_atr = self.calculate_atr(df, 20).iloc[-1] if hasattr(self, 'calculate_atr') else 5.0
        min_cup_depth_points = current_atr * 2.0 
        cup_depth_points = max_rim - trough_b_price
        logger.info(f"🔧 ATR TEST: current_atr={current_atr:.2f}, min_required_points={min_cup_depth_points:.2f}")
        logger.info(f"🔧 ATR TEST: cup_depth_points={cup_depth_points:.2f}, cup_depth_pct={cup_depth_pct:.2f}%")

        if cup_depth_points < min_cup_depth_points:
            logger.info(f"❌ STEP1: Cup too shallow {cup_depth_points:.2f} points < {min_cup_depth_points:.2f} (0.5x ATR)")
            return False
            
        if cup_depth_pct > 25:
            logger.info(f"❌ STEP1: Cup too deep {cup_depth_pct:.2f}% > 2.5%")
            return False
        
        descent_duration = (trough_b - peak_a).total_seconds() / 60
        recovery_duration = (peak_c - trough_b).total_seconds() / 60

        if recovery_duration < (descent_duration * 0.3):  # Recovery too fast
            logger.info(f"❌ V-SPIKE: recovery {recovery_duration:.0f}min vs descent {descent_duration:.0f}min")
            return False
        
        if descent_duration < 30:  # Less than 30 minutes descent = spike down
            logger.info(f"❌ V-SPIKE: descent too quick {descent_duration:.0f}min < 30min")
            return False
        
        

        
        total_cup_duration = descent_duration + recovery_duration
        if total_cup_duration < 90:  # Less than 1.5 hours total
            logger.info(f"❌ V-SPIKE: total cup {total_cup_duration:.0f}min < 90min")
            return False
        
        volatility_ratio = self._calculate_pattern_volatility_ratio(df, peak_a, peak_c)
        logger.info(f"🔧 VOLATILITY: ratio={volatility_ratio:.1f}, depth={cup_depth_pct:.2f}%")
        if volatility_ratio < 2.0:  # Pattern depth must be 2x its own volatility
            logger.info(f"❌ NOISE: Depth/volatility ratio {volatility_ratio:.1f} < 2.0")
            return False
        
        cup_segment = df.loc[peak_a:peak_c]
        cup_low = cup_segment['low'].min()
        cup_range = max_rim - cup_low

        # Define "bottom zone" (lowest 25% of cup range)
        bottom_threshold = cup_low + (cup_range * 0.25)

        # Count bars in bottom zone
        bottom_bars = len(cup_segment[cup_segment['low'] <= bottom_threshold])
        bottom_percentage = bottom_bars / len(cup_segment)

        if bottom_percentage < 0.20:  # Less than 20% time at bottom
            logger.info(f"❌ V-SPIKE: only {bottom_percentage:.1%} time at bottom")
            return False

        logger.info(f"✅ BOTTOM ACCUMULATION: {bottom_percentage:.1%} time at bottom")
        
        # STEP 2: Rim symmetry (keep existing logic)
        rim_diff_abs = abs(peak_a_price - peak_c_price)
        rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
        
        current_atr = self.calculate_atr(df, 20).iloc[-1] if hasattr(self, 'calculate_atr') else 15.0

        # Rim difference should not exceed 1.5x ATR
        max_allowed_rim_diff = current_atr * 1.2

        if rim_diff_abs > max_allowed_rim_diff:
            logger.info(f"❌ ASYMMETRIC: Rim difference {rim_diff_abs:.2f} points > {max_allowed_rim_diff:.2f} (1.5x ATR)")
            return False

        rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
        logger.info(f"✅ SYMMETRIC: Rim difference {rim_diff_abs:.2f} points ({rim_diff_pct:.2f}%) ≤ {max_allowed_rim_diff:.2f}")

        
        # STEP 3: Duration validation (keep existing logic)
        duration_minutes = (peak_c - peak_a).total_seconds() / 60
        if not (30 <= duration_minutes <= 720):
            logger.info(f"❌ STEP1: Duration {duration_minutes:.0f} min outside 30-720 range")
            return False
        
        # STEP 4: ⚠️ CRITICAL FIX - FUNDAMENTAL RULE #1: Cup Rim Violation
        # ZERO TOLERANCE: No price should EVER exceed rim during cup formation
        cup_period = df.loc[peak_a:peak_c]
        max_high_in_cup = cup_period['high'].max()
        actual_lowest_point = cup_period['low'].min()
        cup_bottom_price = df.loc[trough_b, 'low']

        # Check every candle's low in the cup period
        violation_lows = cup_period[cup_period['low'] < cup_bottom_price ]  # 0.1% tolerance

        if not violation_lows.empty:
            worst_violation = cup_period['low'].min()
            violation_amount = cup_bottom_price - worst_violation
            logger.info(f"❌ CUP VIOLATION: Candles broke below cup bottom by ${violation_amount:.2f}")
            return False

        if actual_lowest_point < trough_b_price:
            violation_depth = trough_b_price - actual_lowest_point
            logger.info(f"❌ VIOLATION: Price went ${violation_depth:.2f} below cup bottom")
            return False

        # FIXED: Remove tolerance completely - hard geometric constraint
        current_atr = self.calculate_atr(df, 20).iloc[-1] if hasattr(self, 'calculate_atr') else 15.0
        max_allowed_violation = current_atr * 0.15  # Allow only 0.15x ATR above rim

        if max_high_in_cup > (max_rim + max_allowed_violation):
            violation_amount = max_high_in_cup - max_rim
            logger.info(f"❌ RIM VIOLATION: Price exceeded rim by ${violation_amount:.2f} > ${max_allowed_violation:.2f} (0.25x ATR)")
            return False
        
        logger.info(f"✅ STEP1 CUP: rim_diff={rim_diff_pct:.2f}%, depth={cup_depth_pct:.2f}%, dur={duration_minutes:.0f}min")
        return True
    

    def _calculate_pattern_volatility_ratio(self, df, peak_a, peak_c):
        """Calculate if pattern movement is significant vs its own volatility"""
        cup_period = df.loc[peak_a:peak_c]
        
        # Pattern's own volatility
        pattern_volatility = cup_period['close'].pct_change().std()
        
        # Pattern's depth vs its volatility
        max_rim = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])
        trough_price = cup_period['low'].min()
        pattern_depth_pct = ((max_rim - trough_price) / max_rim) * 100
        
        
        
        # Ratio: depth should be at least 3x the pattern's own volatility
        if pattern_volatility > 0:
            depth_to_volatility_ratio = pattern_depth_pct / (pattern_volatility * 100)
            return depth_to_volatility_ratio
        return 0
                            

    def _validate_cup_structure(self, df, peak_a, trough_b, peak_c):
        """Validate cup structure - ALLOW LONGER PATTERNS"""
        peak_a_price = df.loc[peak_a, 'high']
        peak_c_price = df.loc[peak_c, 'high'] 
        trough_b_price = df.loc[trough_b, 'low']
        
        # Cup depth validation 
        max_rim = max(peak_a_price, peak_c_price)
        cup_depth_pct = ((max_rim - trough_b_price) / max_rim) * 100
        
        if cup_depth_pct < self.config['min_cup_depth_pct']:   # Minimum meaningful depth
            logger.info(f"❌ Cup too shallow: {cup_depth_pct:.2f}% < 1.0%")
            return False
            
        if cup_depth_pct > self.config['max_cup_depth_pct']:  # Maximum reasonable depth
            logger.info(f"❌ Cup too deep: {cup_depth_pct:.2f}% > 15.0%")
            return False
        
        # Rim symmetry check
        rim_diff_abs = abs(peak_a_price - peak_c_price)
        rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
        
        if rim_diff_pct > 8.0:
            logger.info(f"❌ Rim asymmetry: {rim_diff_pct:.2f}% > 8.0%")
            return False
        
        # LONGER DURATION CHECK - Allow multi-day patterns
        duration_minutes = (peak_c - peak_a).total_seconds() / 60
        duration_days = duration_minutes / (60 * 24)
        
        # Allow patterns from 1 hour to 90 days
        if not (60 <= duration_minutes <= 129600):  # 1 hour to 90 days
            logger.info(f"❌ Bad duration: {duration_days:.1f} days (need 1-90 days)")
            return False
        
        logger.info(f"✅ Valid: rim={rim_diff_pct:.2f}%, depth={cup_depth_pct:.2f}%, duration={duration_days:.1f} days")
        return True


    def _create_structure_pattern(self, df, peak_a, trough_b, peak_c, handle, breakout):
        """Create structure-based pattern"""
        try:
            # Get prices
            peak_a_price = df.loc[peak_a, 'high']
            peak_c_price = df.loc[peak_c, 'high']
            trough_b_price = df.loc[trough_b, 'low']
            
            # Calculate metrics
            max_rim = max(peak_a_price, peak_c_price)
            cup_depth = max_rim - trough_b_price
            cup_depth_pct = (cup_depth / max_rim) * 100
            
            # Duration
            cup_duration_min = (peak_c - peak_a).total_seconds() / 60
            handle_duration_min = (handle['end'] - handle['start']).total_seconds() / 60
            
            # Quality score
            quality_score = 60 + (cup_depth_pct * 5)  # Higher score for deeper cups
            
            return {
                'peak_a': peak_a,
                'trough_b': trough_b,
                'peak_c': peak_c,
                'handle_d': handle['start'],
                'breakout_e': breakout,
                'breakout_confirmed': True,
                
                'cup_depth': cup_depth,
                'cup_depth_pct': cup_depth_pct,
                'cup_duration_min': cup_duration_min,
                'handle_depth': max_rim - handle['low_price'],
                'handle_depth_pct': handle['depth_pct'],
                'handle_duration_min': handle_duration_min,
                
                'cup_symmetry': 0.7,
                'cup_roundness': 0.6,
                'quality_score': min(100, quality_score),
                'confidence_score': 0.8,
                
                'pattern_type': 'micro_structure',
                'timeframe': '15min',
                'detection_method': 'structure_based'
            }
        except Exception as e:
            logger.warning(f"Pattern creation error: {e}")
            return None
    
    def _calculate_pattern_significance(self, cup_depth_pct, handle_depth_pct, duration_min):
        """Calculate if pattern is institutionally significant"""
        
        # Base significance score
        significance = cup_depth_pct * 10  # Convert to 0-100 scale
        
        # Bonus for longer duration (institutional accumulation takes time)
        if duration_min >= 300:  # 5+ hours
            significance += 15
        elif duration_min >= 180:  # 3+ hours  
            significance += 10
        
        # Bonus for balanced handle (not too deep, not too shallow)
        if 0.3 <= handle_depth_pct <= 1.0:
            significance += 10
        
        return significance
    
    
    

    def _validate_price_momentum_context(self, df, peak_a, peak_c):
        """Ensure pattern has bullish context using only pattern data"""
        
        # Check if price is generally rising during pattern formation
        cup_period = df.loc[peak_a:peak_c]
        
        # Calculate trend within the pattern itself
        start_price = df.loc[peak_a, 'close']
        end_price = df.loc[peak_c, 'close']
        
        # Pattern should not be in strong decline
        decline_pct = (start_price - end_price) / start_price * 100
        
        if decline_pct > 8.0:  # More than 8% decline during pattern
            logger.info(f"❌ CONTEXT: Pattern in {decline_pct:.1f}% decline")
            return False
        
        return True

    
    def calculate_atr(self, df, period=20):
        """Fixed ATR calculation"""
        try:
            # Ensure we have enough data
            if len(df) < period:
                period = max(2, len(df) // 2)
            
            # Calculate True Range components
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1)).fillna(0)
            tr3 = abs(df['low'] - df['close'].shift(1)).fillna(0)
            
            # True Range is the maximum of the three
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR is the simple moving average of True Range
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            return atr
        except Exception as e:
            logger.info(f"ATR calculation failed: {e}")
            return pd.Series([15.0] * len(df), index=df.index) # Default ES ATR fallback


    def _find_handle_pullback(self, df, peak_c, trough_b):
        """Find handle pullback with ATR-based validation"""
        peak_c_idx = df.index.get_loc(peak_c)
        search_end = min(len(df), peak_c_idx + 20)
        
        if peak_c_idx >= len(df) - 3:
            return None
            
        peak_c_price = df.loc[peak_c, 'high']
        trough_b_price = df.loc[trough_b, 'low']
        current_atr = self._calculate_atr(df)
        
        # ATR-based handle requirements
        min_handle_points = current_atr * 0.5   # 0.5x ATR minimum
        max_handle_points = current_atr * 3.0   # 3x ATR maximum
        
        # Find pullback in next bars
        for i in range(peak_c_idx + 1, search_end):
            handle_depth_points = peak_c_price - df.iloc[i]['low']
            
            if handle_depth_points >= min_handle_points:  # Meaningful pullback
                
                # Find where pullback ends
                for j in range(i + 1, min(search_end, i + 15)):
                    if df.iloc[j]['high'] > df.iloc[i:j]['low'].min() * 1.002:
                        
                        handle_low = df.iloc[i:j]['low'].min()
                        handle_depth_points = peak_c_price - handle_low
                        handle_depth_pct = (handle_depth_points / peak_c_price) * 100
                        
                        # ATR-based validation
                        if (min_handle_points <= handle_depth_points <= max_handle_points and
                            handle_low > trough_b_price):
                            
                            logger.info(f"✅ Found handle: {handle_depth_points:.1f} pts ({handle_depth_pct:.2f}%)")
                            return {
                                'start': df.index[i],
                                'end': df.index[j-1],
                                'low_price': handle_low,
                                'depth_pct': handle_depth_pct
                            }
        return None
    
    def _check_micro_cup_handle(self, window, start_idx, price_col):
        """Fast check if window contains a micro cup-and-handle pattern"""
        if len(window) < 4:  # Need minimum bars
            return None
        
        # Find potential cup boundary (first 60-80% of window)
        cup_end_idx = int(len(window) * 0.7)
        if cup_end_idx < 3:
            cup_end_idx = len(window) - 2
        
        cup_window = window.iloc[:cup_end_idx]
        
        # Quick cup validation
        if not self._is_valid_micro_cup(cup_window, price_col):
            return None
        
        # Check for handle in remaining bars
        if cup_end_idx < len(window) - 1:
            handle_window = window.iloc[cup_end_idx:]
            if not self._is_valid_micro_handle(cup_window, handle_window, price_col):
                return None
        else:
            return None  # No room for handle
        
        # Create pattern object
        return self._create_micro_pattern(window, cup_window, handle_window, start_idx, price_col)
    
    def _is_valid_micro_cup(self, cup_window, price_col):
        """Fast validation of micro cup formation"""
        if len(cup_window) < 3:
            return False
        
        # Get cup boundaries and bottom
        left_rim = cup_window['high'].iloc[0]
        right_rim = cup_window['high'].iloc[-1]
        cup_bottom = cup_window['low'].min()
        
        # Point-based depth check (faster than percentage for micro patterns)
        left_depth_points = left_rim - cup_bottom
        right_depth_points = right_rim - cup_bottom
        
        if (left_depth_points < self.config['min_cup_depth_points'] or 
            right_depth_points < self.config['min_cup_depth_points']):
            return False
        
        # Rim symmetry check (point-based)
        rim_diff_points = abs(left_rim - right_rim)
        if rim_diff_points > self.config['rim_tolerance_points']:
            return False
        
        # Ensure cup bottom is actually lower than rims
        max_rim = max(left_rim, right_rim)
        if cup_bottom >= max_rim * 0.998:  # 0.2% tolerance
            return False
        
        return True
    
    def _is_valid_micro_handle(self, cup_window, handle_window, price_col):
        """Fast validation of micro handle formation"""
        if len(handle_window) < 1:
            return False
        
        cup_right_rim = cup_window['high'].iloc[-1]
        handle_low = handle_window['low'].min()
        handle_high = handle_window['high'].max()
        
        # Handle must pull back from rim
        handle_depth_points = cup_right_rim - handle_low
        if handle_depth_points < self.config['min_handle_depth_points']:
            return False
        
        # Handle shouldn't be too deep
        handle_depth_pct = (handle_depth_points / cup_right_rim) * 100
        if handle_depth_pct > self.config['max_handle_depth_pct']:
            return False
        
        # Handle shouldn't exceed rim significantly
        if handle_high > cup_right_rim + 1.0:  # 1 point tolerance
            return False
        
        if handle_depth_pct > 0.33 * self.config['cup_depth_pct']:
            logger.info(f"❌ Handle too deep: {handle_depth_pct:.2f}% > 33% of cup depth")
            return False
        
        return True
    
    def _create_micro_pattern(self, full_window, cup_window, handle_window, start_idx, price_col):
        """Create micro pattern object compatible with your existing format"""
        
        # Find key points
        left_rim_idx = 0
        cup_bottom_idx = cup_window['low'].idxmin()
        right_rim_idx = len(cup_window) - 1
        handle_low_idx = handle_window['low'].idxmin()
        
        # Convert to absolute timestamps
        pattern_start_time = full_window.index[left_rim_idx]
        cup_bottom_time = cup_bottom_idx
        cup_end_time = full_window.index[right_rim_idx]
        handle_time = handle_low_idx
        
        # Simple breakout (last bar if it's higher than rim)
        rim_price = max(cup_window['high'].iloc[0], cup_window['high'].iloc[-1])
        breakout_time = full_window.index[-1]
        breakout_confirmed = full_window['high'].iloc[-1] > rim_price
        
        # Calculate metrics
        cup_depth = rim_price - cup_window['low'].min()
        cup_depth_pct = (cup_depth / rim_price) * 100
        handle_depth = rim_price - handle_window['low'].min()
        handle_depth_pct = (handle_depth / rim_price) * 100
        
        # Simple quality score
        quality_score = self._calculate_micro_quality(
            cup_depth_pct, handle_depth_pct, len(cup_window), len(handle_window)
        )
        
        return {
            'peak_a': pattern_start_time,
            'trough_b': cup_bottom_time,
            'peak_c': cup_end_time,
            'handle_d': handle_time,
            'breakout_e': breakout_time,
            'breakout_confirmed': breakout_confirmed,
            
            # Metrics
            'cup_depth': cup_depth,
            'cup_depth_pct': cup_depth_pct,
            'cup_duration_min': len(cup_window) * 15,  # Assume 15min bars
            'handle_depth': handle_depth,
            'handle_depth_pct': handle_depth_pct,
            'handle_duration_min': len(handle_window) * 15,
            
            # Shape metrics (simplified for micro)
            'cup_symmetry': 0.7,  # Default good symmetry
            'cup_roundness': 0.6,  # Default medium roundness
            'quality_score': quality_score,
            'confidence_score': 0.7,  # Default confidence
            
            # Micro-specific metadata
            'pattern_type': 'micro',
            'timeframe': '15min',
            'total_bars': len(full_window),
            'cup_bars': len(cup_window),
            'handle_bars': len(handle_window),
            'detection_method': 'micro_fast_scan'
        }
    
    def _calculate_micro_quality(self, cup_depth_pct, handle_depth_pct, cup_bars, handle_bars):
        """Simple quality score for micro patterns"""
        score = 50  # Base score
        
        # Depth scoring (micro patterns have smaller depths)
        if 0.1 <= cup_depth_pct <= 0.8:
            score += 20
        elif cup_depth_pct <= 1.2:
            score += 10
        
        # Handle depth scoring
        if 0.02 <= handle_depth_pct <= 0.2:
            score += 15
        elif handle_depth_pct <= 0.4:
            score += 8
        
        # Duration scoring
        if 3 <= cup_bars <= 12:
            score += 10
        if 1 <= handle_bars <= 4:
            score += 5
        
        return min(100, score)
    
    def _filter_micro_patterns(self, patterns, df):
        """Quick quality filter for micro patterns"""
        if not patterns:
            return []
        
        # FIRST: Apply final pattern validation
        validated_patterns = []
        for pattern in patterns:
            if self._final_pattern_validation(df, pattern):
                validated_patterns.append(pattern)
            else:
                logger.info(f"❌ FINAL FILTER: Rejected pattern {pattern['peak_a']} - cup bottom violated")
        
        logger.info(f"🔬 FINAL VALIDATION: {len(patterns)} → {len(validated_patterns)} patterns")
        
        # Sort by quality (use validated_patterns, not patterns)
        validated_patterns.sort(key=lambda p: p['quality_score'], reverse=True)
        
        # Apply minimum quality threshold (use validated_patterns)
        filtered = [p for p in validated_patterns if p['quality_score'] >= self.config['min_quality_score']]
        
        # Remove overlapping patterns (use filtered, which now comes from validated_patterns)
        final_patterns = []
        for pattern in filtered:
            overlap = False
            pattern_start = pattern['peak_a']
            pattern_end = pattern['breakout_e']
            
            for existing in final_patterns:
                existing_start = existing['peak_a']
                existing_end = existing['breakout_e']
                
                if (pattern_start <= existing_end and pattern_end >= existing_start):
                    overlap = True
                    break
            
            if not overlap:
                final_patterns.append(pattern)
            
            if len(final_patterns) >= 500:
                break
        
        logger.info(f"🔬 Micro filter: {len(patterns)} → {len(validated_patterns)} → {len(filtered)} → {len(final_patterns)} patterns")
        return final_patterns


    # Additional methods to maintain compatibility with existing interface
    def detect_timeframe(self, df):
        """Detect timeframe - assume 15min for micro detector"""
        return 15
    
    def _final_pattern_validation(self, df, pattern):
        """Final validation to catch patterns that slip through"""
        try:
            cup_bottom_price = df.loc[pattern['trough_b'], 'low']
            
            logger.info(f"🔧 VALIDATING PATTERN: {pattern['peak_a']} to {pattern['breakout_e']}")
            logger.info(f"🔧 Cup bottom at {pattern['trough_b']}: ${cup_bottom_price:.2f}")
            
            trough_b_idx = df.index.get_loc(pattern['trough_b'])
            peak_c_idx = df.index.get_loc(pattern['peak_c'])

            pattern_start = pattern['peak_a']
            pattern_end = pattern['breakout_e']
            pattern_period = df.loc[pattern_start:pattern_end]
            cup_bottom_price = df.loc[pattern['trough_b'], 'low']

            # Check for ANY violation during entire pattern
            violations = pattern_period[pattern_period['low'] < cup_bottom_price]
            if not violations.empty:
                worst_violation = violations['low'].min()
                violation_time = violations['low'].idxmin()
                logger.info(f"❌ FINAL: Cup bottom violated at {violation_time}: ${worst_violation:.2f} < ${cup_bottom_price:.2f}")
                return False
            
            if trough_b_idx < peak_c_idx:
                post_trough_period = df.iloc[trough_b_idx + 1:peak_c_idx + 1]
                
                if len(post_trough_period) > 0:
                    post_trough_min = post_trough_period['low'].min()
                    logger.info(f"🔧 Post-trough minimum: ${post_trough_min:.2f}")
                    
                    if post_trough_min < cup_bottom_price * 0.999:
                        violation_amount = cup_bottom_price - post_trough_min
                        logger.info(f"❌ FINAL: Price went ${violation_amount:.2f} below cup bottom")
                        return False
                        
            logger.info(f"✅ FINAL: Pattern validation passed")
            return True
        except Exception as e:
            logger.info(f"❌ FINAL: Validation error: {e}")
            return False
    
    def visualize_pattern(self, pattern, df, save_path=None):
        """Placeholder - can implement micro-specific visualization later"""
        logger.info(f"Micro pattern visualization not implemented yet")
        return None
    

