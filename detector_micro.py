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
            "max_cup_depth_pct": 1.5,     
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
        
        # Preprocess for micro detection
        processed_df = self._preprocess_micro(df)
        
        # Fast micro pattern scan
        patterns = self._scan_micro_patterns(processed_df, price_col)
        
        # Quick quality filter
        filtered_patterns = self._filter_micro_patterns(patterns, processed_df)
        
        logger.info(f"🔬 Micro detection complete: {len(filtered_patterns)} patterns found")
        return filtered_patterns
    
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
        
        # Mark extrema
        processed['extrema'] = 0
        if len(highs) > 0:
            processed.iloc[highs, processed.columns.get_loc('extrema')] = 1
        if len(lows) > 0:
            processed.iloc[lows, processed.columns.get_loc('extrema')] = -1
        
        return processed
    
    def _scan_micro_patterns(self, df, price_col):
        """Find TRUE MICRO patterns - STRICT time limits to prevent long patterns"""
        patterns = []
        
        # Get significant highs and lows only
        peaks = df[df['extrema'] == 1].index.tolist()
        troughs = df[df['extrema'] == -1].index.tolist()
        
        if len(peaks) < 2 or len(troughs) < 1:
            return patterns
        
        logger.info(f"🔍 STRICT Micro scan: {len(peaks)} peaks, {len(troughs)} troughs")
        
        # MICRO PATTERN FIX: Strict 8-hour maximum
        MAX_PATTERN_MINUTES = 8 * 60  # 8 hours maximum
        
        # Look for A-B-C-D-E structure with STRICT TIME CONSTRAINTS
        for i, peak_a in enumerate(peaks[:-1]):  # Left rim candidate
            
            for peak_c in peaks[i+1:]:           # Right rim candidate
                
                # GROUND RULE: Pattern must be 1-8 hours (STRICT)
                pattern_duration_minutes = (peak_c - peak_a).total_seconds() / 60
                
                if pattern_duration_minutes > MAX_PATTERN_MINUTES:
                    continue  # Skip long patterns - not micro
                
                if pattern_duration_minutes < 60:  # Less than 1 hour
                    continue  # Too short to be meaningful
                
                # Find cup bottom between rims
                cup_troughs = [t for t in troughs if peak_a < t < peak_c]
                if not cup_troughs:
                    continue
                    
                trough_b = min(cup_troughs, key=lambda t: df.loc[t, 'low'])
                
                # STRICT GROUND RULES VALIDATION
                if self._validate_strict_cup_structure(df, peak_a, trough_b, peak_c):
                    
                    # Look for handle with STRICT rules
                    handle = self._find_strict_handle(df, peak_c, trough_b)
                    if handle:
                        
                        # Check for STRICT breakout
                            breakout = self._find_strict_breakout(df, handle, peak_a, peak_c)
                            if breakout:
                                pattern = self._create_strict_micro_pattern(df, peak_a, trough_b, peak_c, handle, breakout) 
                            if pattern:
                                patterns.append(pattern)
                                logger.info(f"✅ STRICT MICRO: {pattern['cup_depth_pct']:.1f}% cup, {pattern_duration_minutes:.0f} min")
        
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
        """GROUND RULE 3: Breakout never below cup rim and at same height as cup rim"""
        try:
            handle_end_idx = df.index.get_loc(handle['end'])
            rim_level = max(df.loc[peak_a, 'high'], df.loc[peak_c, 'high'])
            
            # Look for breakout in next 10 bars (2.5 hours max)
            for i in range(handle_end_idx + 1, min(len(df), handle_end_idx + 11)):
                breakout_high = df.iloc[i]['high']
                
                # GROUND RULE 3: Must break above or at rim level
                if breakout_high >= rim_level * 0.999:  # At or above rim (0.1% tolerance)
                    
                    # STRICT: Breakout can't be significantly above rim (avoid gaps)
                    if breakout_high <= rim_level * 1.005:  # Max 0.5% above
                        logger.info(f"✅ STRICT BREAKOUT: {breakout_high:.2f} at rim {rim_level:.2f}")
                        return df.index[i]
                    else:
                        logger.info(f"❌ STRICT: Breakout too high {breakout_high:.2f} vs rim {rim_level:.2f}")
                        return None
            
            logger.info(f"❌ STRICT: No valid breakout found")
            return None
        except Exception as e:
            logger.info(f"❌ STRICT: Breakout search error: {e}")
            return None
                

    
    def _find_strict_handle(self, df, peak_c, trough_b):
        """MINIMAL FIX: Keep original logic but add minimum gap requirement"""
        peak_c_idx = df.index.get_loc(peak_c)
        search_end = min(len(df), peak_c_idx + 32)  # 8 hours max for handle
        
        if peak_c_idx >= len(df) - 3:
            return None
        
        peak_c_price = df.loc[peak_c, 'high']
        trough_b_price = df.loc[trough_b, 'low']
        
        # NEW: Minimum gap after peak C to prevent rushed handles
        min_gap_bars = 2  # At least 30 minutes (2 bars of 15min each)
        
        # Look for pullback starting AFTER the minimum gap
        for i in range(peak_c_idx + 1 + min_gap_bars, search_end):  # ← ONLY CHANGE: +min_gap_bars
            handle_low = df.iloc[i]['low']
            
            # GROUND RULE 2: STRICT handle position
            if handle_low >= peak_c_price:
                continue  # Handle must be below rim
            
            if handle_low <= trough_b_price:
                continue  # Handle must be above cup bottom
            
            # Find where pullback ends (recovery starts)
            for j in range(i + 1, min(search_end, i + 20)):
                if df.iloc[j]['high'] > handle_low * 1.002:  # Recovery detected
                    
                    handle_depth_points = peak_c_price - handle_low
                    handle_depth_pct = (handle_depth_points / peak_c_price) * 100
                    
                    # Micro handle: 0.2% - 2% depth (KEEP ORIGINAL CRITERIA)
                    if 0.2 <= handle_depth_pct <= 2.0:
                        
                        # STRICT: Handle never goes above rim during formation
                        handle_period = df.iloc[i:j]
                        if handle_period['high'].max() <= peak_c_price * 1.001:
                            
                            logger.info(f"✅ STRICT HANDLE: {handle_depth_pct:.2f}% depth")
                            return {
                                'start': df.index[i],
                                'end': df.index[j-1],
                                'low_price': handle_low,
                                'depth_pct': handle_depth_pct
                            }
        return None
        
    def _validate_strict_cup_structure(self, df, peak_a, trough_b, peak_c):
        """RELAXED validation - more realistic for actual market data"""
        peak_a_price = df.loc[peak_a, 'high']
        peak_c_price = df.loc[peak_c, 'high'] 
        trough_b_price = df.loc[trough_b, 'low']
        
        # RELAXED: RIM SYMMETRY (allow more variation)
        rim_diff_abs = abs(peak_a_price - peak_c_price)
        rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
        
        # RELAXED: Maximum 5% difference (was 1%)
        if rim_diff_pct > 5.0:  # ← CHANGED from 1.0 to 5.0
            logger.info(f"❌ RELAXED: Rim asymmetry {rim_diff_pct:.2f}% > 5.0%")
            return False
        
        # RELAXED: Cup depth validation 
        max_rim = max(peak_a_price, peak_c_price)
        cup_depth_pct = ((max_rim - trough_b_price) / max_rim) * 100
        
        # RELAXED: 0.2% - 8% depth (was 0.5% - 5%)
        if cup_depth_pct < 0.2:  # ← CHANGED from 0.5 to 0.2
            logger.info(f"❌ RELAXED: Cup too shallow {cup_depth_pct:.2f}% < 0.2%")
            return False
            
        if cup_depth_pct > 8.0:  # ← CHANGED from 5.0 to 8.0
            logger.info(f"❌ RELAXED: Cup too deep {cup_depth_pct:.2f}% > 8.0%")
            return False
        
        # RELAXED: Duration 30 min - 12 hours (was 1-8 hours)
        duration_minutes = (peak_c - peak_a).total_seconds() / 60
        if not (30 <= duration_minutes <= 720):  # ← CHANGED from 60-480 to 30-720
            logger.info(f"❌ RELAXED: Duration {duration_minutes:.0f} min outside 30-720 range")
            return False
        
        # RELAXED: Allow minor price spikes (was 0.1% tolerance)
        cup_period = df.loc[peak_a:peak_c]
        max_high_in_cup = cup_period['high'].max()
        
        if max_high_in_cup > max_rim * 1.005:  # ← CHANGED from 1.001 to 1.005 (0.5% tolerance)
            violation_pct = ((max_high_in_cup - max_rim) / max_rim) * 100
            logger.info(f"❌ RELAXED: Price spike {violation_pct:.2f}% above rim during cup")
            return False
        
        logger.info(f"✅ RELAXED CUP: rim_diff={rim_diff_pct:.2f}%, depth={cup_depth_pct:.2f}%, dur={duration_minutes:.0f}min")
        return True
            

    def _validate_cup_structure(self, df, peak_a, trough_b, peak_c):
        """Validate cup structure - ALLOW LONGER PATTERNS"""
        peak_a_price = df.loc[peak_a, 'high']
        peak_c_price = df.loc[peak_c, 'high'] 
        trough_b_price = df.loc[trough_b, 'low']
        
        # Cup depth validation 
        max_rim = max(peak_a_price, peak_c_price)
        cup_depth_pct = ((max_rim - trough_b_price) / max_rim) * 100
        
        if cup_depth_pct < 1.0:   # Minimum meaningful depth
            logger.info(f"❌ Cup too shallow: {cup_depth_pct:.2f}% < 1.0%")
            return False
            
        if cup_depth_pct > 15.0:  # Maximum reasonable depth
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
    

    
    def _calculate_atr(self, df, period=20):
        """Calculate Average True Range for adaptive thresholds"""
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            return atr.iloc[-1]  # Return latest ATR
        except:
            return 15.0  # Default ES ATR fallback


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
        
        # Sort by quality
        patterns.sort(key=lambda p: p['quality_score'], reverse=True)
        
        # Apply minimum quality threshold
        filtered = [p for p in patterns if p['quality_score'] >= self.config['min_quality_score']]
        
        # Remove overlapping patterns (keep higher quality)
        final_patterns = []
        for pattern in filtered:
            overlap = False
            pattern_start = pattern['peak_a']
            pattern_end = pattern['breakout_e']
            
            for existing in final_patterns:
                existing_start = existing['peak_a']
                existing_end = existing['breakout_e']
                
                # Check for time overlap
                if (pattern_start <= existing_end and pattern_end >= existing_start):
                    overlap = True
                    break
            
            if not overlap:
                final_patterns.append(pattern)
            
            # Limit final results
            if len(final_patterns) >= 20:  # Max 20 micro patterns
                break
        
        logger.info(f"🔬 Micro filter: {len(patterns)} → {len(filtered)} → {len(final_patterns)} patterns")
        return final_patterns

    # Additional methods to maintain compatibility with existing interface
    def detect_timeframe(self, df):
        """Detect timeframe - assume 15min for micro detector"""
        return 15
    
    def visualize_pattern(self, pattern, df, save_path=None):
        """Placeholder - can implement micro-specific visualization later"""
        logger.info(f"Micro pattern visualization not implemented yet")
        return None