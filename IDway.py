import cv2
import pytesseract
from PIL import Image
import re
from collections import Counter
import json
from fuzzywuzzy import fuzz
import numpy as np

class IDway:
    def __init__(self, image_path, first_name=None, last_name=None, street_address=None, date_of_birth=None):
        self.image_path = image_path
        self.provided_info = {
            'first_name': first_name.upper() if first_name else None,
            'last_name': last_name.upper() if last_name else None,
            'street_address': street_address.upper() if street_address else None,
            'date_of_birth': date_of_birth if date_of_birth else None
        }
        # Enhanced patterns for better name matching
        self.patterns = {
            "state_header": re.compile(r"(NEW YORK|CALIFORNIA|TEXAS|FLORIDA|etc)\s+STATE", re.IGNORECASE),
            "name": [
                # Multiple name patterns to try
                re.compile(r"([A-Z'-]+)[,.\s]+([A-Z'-]+(?:\s+[A-Z'-]+)*)", re.MULTILINE),  # Last, First
                re.compile(r"([A-Z'-]+(?:\s+[A-Z'-]+)*)\s+([A-Z'-]+)", re.MULTILINE),      # First Last
                re.compile(r"([A-Z'-]+)[,.\s]*\n\s*([A-Z'-]+)", re.MULTILINE),             # Split by newline
            ],
            "date_of_birth": [
                re.compile(r"DOB[,.\s:]+(\d{2}/\d{2}/\d{4})"),
                re.compile(r"(\d{2}/\d{2}/\d{4})")  # Fallback to any date format
            ],
            "license_number": re.compile(r"[A-Z0-9]\s*(\d{3}\s*\d{3}\s*\d{3})\s*[A-Z0-9]"),
            "address": [
                re.compile(r"(\d+\s+[A-Z0-9\s]+(?:ST|AVE|RD|BLVD|APT).+?\d{5})"),
                re.compile(r"(\d+[A-Z0-9\s,]+\d{5})")  # More permissive address pattern
            ],
            "expiration": re.compile(r"(?:EXP|EXPIRES?)[,.\s:]+(\d{2}/\d{2}/\d{4})"),
            "issue_date": re.compile(r"(?:ISS|ISSUED)[,.\s:]+(\d{2}/\d{2}/\d{4})"),
            "class": re.compile(r"CLASS[,.\s:]*([A-Z])")
        }
        # State-specific validation rules
        self.state_rules = {
            "NY": {
                "license_format": r"\d{3}\s?\d{3}\s?\d{3}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119"]
            }
        }
        self.required_fields = {'name', 'date_of_birth', 'license_number', 'address', 'expiration'}
        
    def _preprocess_image(self):
        '''
        Preprocessing focused on strongest differentiators
        '''
        # Read image
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Calculate metrics focusing on key differentiators
        quality_metrics = {
            "resolution_score": max(0, 100 - ((width * height) / (1000 * 1000) * 100)),
            "color_transition": min(100, self._calculate_color_transitions(hsv)),
            
            # Keep these as supporting metrics with lower weights
            "rainbow_effect": min(100, (np.std(hsv[:, :, 0]) / 25) * 100),
            "blur_score": max(0, 100 - min(100, cv2.Laplacian(gray, cv2.CV_64F).var())),
            "saturation_score": min(100, (np.mean(hsv[:, :, 1]) / 255) * 150),
            "digital_artifacts": min(100, (np.std(ycrcb[::8, ::8, :]) / np.std(ycrcb)) * 50)
        }
        
        # Store metrics for fraud detection
        self.quality_metrics = quality_metrics
        
        # Focus on most reliable indicators
        self.fake_indicators = []
        
        # Primary indicators (adjusted thresholds based on observed patterns)
        if quality_metrics["resolution_score"] > 50:  # Lowered from 70
            self.fake_indicators.append("Suspicious image resolution")
        if quality_metrics["color_transition"] > 30:  # Lowered from 40
            self.fake_indicators.append("Unnatural color transitions")
            
        # Secondary indicators with higher thresholds
        if quality_metrics["rainbow_effect"] > 45:
            self.fake_indicators.append("Suspicious rainbow/hologram pattern")
        if quality_metrics["saturation_score"] > 50:
            self.fake_indicators.append("Excessive color saturation")
        if quality_metrics["digital_artifacts"] > 75:
            self.fake_indicators.append("Digital scanning artifacts detected")
        
        # Calculate overall score with updated weights
        weights = {
            # Primary metrics with highest weights
            "resolution_score": 3.5,    # Increased - very reliable
            "color_transition": 3.5,    # Increased - very reliable
            
            # Secondary metrics with reduced weights
            "rainbow_effect": 1.0,
            "blur_score": 0.5,
            "saturation_score": 0.5,
            "digital_artifacts": 0.5
        }
        
        weighted_scores = [
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        ]
        self.image_quality = min(100, np.average(weighted_scores))
        
        # Proceed with normal preprocessing for OCR
        contrast = cv2.convertScaleAbs(gray, alpha=1.75, beta=0)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 5, 7, 21)
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _extract_text_from_image(self, image):
        '''
        Simplified text extraction with minimal processing
        '''
        custom_config = (
            '--oem 3 '
            '--psm 6 '
            '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\'()-/" '
        )
        
        text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # normalize spaces
        
        return text

    def _validate_dl_text(self, text):
        result = {
            "validation_details": {},
            "text_fraud_score": 0,  # Separate text-based fraud score
            "scoring_factors": [],
            "extracted_data": {},
            "match_scores": {}
        }
        
        # Normalize text for comparison
        text = text.upper()
        print(f"Searching in text: {text}")
        
        # Simple name existence check (worth 50% of text score)
        name_found = False
        if self.provided_info['first_name'] or self.provided_info['last_name']:
            if self.provided_info['first_name']:
                first_name_score = fuzz.partial_ratio(text, self.provided_info['first_name'])
                result["match_scores"]["first_name"] = first_name_score
                print(f"First name ({self.provided_info['first_name']}) score: {first_name_score}")
                if first_name_score > 80:
                    name_found = True
                else:
                    result["text_fraud_score"] += 25
                    result["scoring_factors"].append(f"First name match score: {first_name_score}%")
            
            if self.provided_info['last_name']:
                last_name_score = fuzz.partial_ratio(text, self.provided_info['last_name'])
                result["match_scores"]["last_name"] = last_name_score
                print(f"Last name ({self.provided_info['last_name']}) score: {last_name_score}")
                if last_name_score > 80:
                    name_found = True
                else:
                    result["text_fraud_score"] += 25
                    result["scoring_factors"].append(f"Last name match score: {last_name_score}%")
        
        if not name_found:
            result["text_fraud_score"] += 50
            result["scoring_factors"].append("No valid name found in document")
        
        return result

    def output(self):
        '''
        Enhanced output with reweighted scoring (20% text, 80% image)
        '''
        processed_image = self._preprocess_image()
        extracted_text = self._extract_text_from_image(processed_image)
        validation_result = self._validate_dl_text(extracted_text)
        
        # Calculate image fraud score (already 0-100, where 0 is good)
        image_fraud_score = self.image_quality
        
        # Combine scores (20% text, 80% image quality)
        total_fraud_score = (validation_result["text_fraud_score"] * 0.2) + (image_fraud_score * 0.8)
        
        result = {
            "fraud_score": round(total_fraud_score, 1),
            "risk_level": "High" if total_fraud_score > 70 else "Medium" if total_fraud_score > 40 else "Low",
            "component_scores": {
                "text_fraud_score": {
                    "score": round(validation_result["text_fraud_score"], 1),
                    "weight": "20%"
                },
                "image_fraud_score": {
                    "score": round(image_fraud_score, 1),
                    "weight": "80%"
                }
            },
            "match_scores": validation_result["match_scores"],
            "scoring_factors": validation_result["scoring_factors"],
            "quality_metrics": {k: f"{v:.1f}%" for k, v in self.quality_metrics.items()},
            "fake_indicators": self.fake_indicators,
            "raw_text": extracted_text
        }
        
        # Add interpretation guide
        result["score_interpretation"] = {
            "all_scores": "0-100 (0 = good/authentic, 100 = bad/potentially fraudulent)",
            "weighting": {
                "text_matching": "20% of total score",
                "image_quality": "80% of total score"
            },
            "risk_levels": {
                "Low": "0-40",
                "Medium": "41-70",
                "High": "71-100"
            }
        }
        
        return json.dumps(result, indent=2)

    def _analyze_microprint(self, gray):
        '''Analyze potential microprint areas'''
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        return min(100, (np.std(filtered) / 255) * 100)

    def _detect_uv_simulation(self, hsv):
        '''Detect attempted UV pattern simulation'''
        unusual_colors = np.sum((hsv[:,:,0] > 150) & (hsv[:,:,1] > 200))
        return min(100, (unusual_colors / (hsv.shape[0] * hsv.shape[1])) * 200)

    def _detect_photo_tampering(self, image):
        '''
        Check for signs of photo manipulation with looser constraints
        '''
        quality_levels = [90, 75, 60]
        diffs = []
        
        for q in quality_levels:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            decoded = cv2.imdecode(encoded, 1)
            diff = np.mean(np.abs(image.astype(float) - decoded.astype(float)))
            diffs.append(diff)
        
        variance = np.std(diffs) / np.mean(diffs)
        
        # Looser scoring
        return min(100, variance * 150)  # Reduced multiplier

    def _enhanced_rainbow_detection(self, hsv):
        '''
        Enhanced hologram detection that better distinguishes real from fake patterns
        '''
        # Split into regions to analyze pattern distribution
        h, w = hsv.shape[:2]
        regions = [
            hsv[0:h//2, 0:w//2],      # Top-left
            hsv[0:h//2, w//2:w],      # Top-right
            hsv[h//2:h, 0:w//2],      # Bottom-left
            hsv[h//2:h, w//2:w]       # Bottom-right
        ]
        
        # Analyze hue patterns in each region
        region_scores = []
        for region in regions:
            # Calculate hue statistics
            hue = region[:,:,0]
            sat = region[:,:,1]
            
            # Only consider areas with sufficient saturation
            mask = sat > 50
            if np.sum(mask) > 0:
                hue_masked = hue[mask]
                
                # Calculate hue variance and distribution
                hue_std = np.std(hue_masked)
                hue_hist = np.histogram(hue_masked, bins=30)[0]
                peak_ratio = np.max(hue_hist) / np.mean(hue_hist) if np.mean(hue_hist) > 0 else 0
                
                # Real holograms tend to have more controlled variance
                region_score = min(100, (hue_std * 0.5 + peak_ratio * 0.5))
                region_scores.append(region_score)
        
        if not region_scores:
            return 0
        
        # Analyze pattern consistency across regions
        region_std = np.std(region_scores)
        avg_score = np.mean(region_scores)
        
        # Real holograms tend to have more consistent patterns
        consistency_factor = min(1.0, region_std / 20)
        
        # Calculate final score
        # Lower score for more consistent patterns (characteristic of real holograms)
        rainbow_score = avg_score * (0.5 + consistency_factor)
        
        return min(100, rainbow_score)

    def _enhanced_color_transitions(self, hsv):
        '''
        Enhanced color transition detection
        '''
        # Calculate gradients in both directions
        gradient_x = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Look for suspicious patterns
        direction_hist = np.histogram(gradient_direction, bins=36)[0]
        direction_peaks = np.sum(direction_hist > np.mean(direction_hist) * 1.5)
        
        # Calculate mean gradient magnitude
        mean_magnitude = np.mean(gradient_magnitude)
        
        # Combine metrics (higher score = more suspicious transitions)
        transition_score = (mean_magnitude * 0.7 + direction_peaks * 0.3)
        
        return min(100, transition_score * 2)

    def _analyze_texture_uniformity(self, gray):
        '''
        Improved texture uniformity analysis
        '''
        # Use Local Binary Pattern (LBP) for better texture analysis
        kernel_size = 3
        local_std = np.zeros_like(gray, dtype=float)
        
        # Calculate local standard deviation with smaller kernel
        for i in range(kernel_size//2, gray.shape[0]-kernel_size//2):
            for j in range(kernel_size//2, gray.shape[1]-kernel_size//2):
                patch = gray[i-kernel_size//2:i+kernel_size//2+1, 
                           j-kernel_size//2:j+kernel_size//2+1]
                local_std[i,j] = np.std(patch)
        
        # Real IDs should have a mix of uniform and detailed areas
        texture_variation = np.std(local_std) / np.mean(local_std)
        
        # Score where too uniform (low variation) or too random (high variation) is bad
        return min(100, abs(texture_variation - 0.5) * 100)

    def _analyze_color_distribution(self, hsv):
        '''
        Improved color distribution analysis
        '''
        # Analyze hue distribution
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue_hist = hue_hist / np.sum(hue_hist)
        
        # Real IDs typically have specific color ranges
        dominant_hues = np.sum(hue_hist > np.mean(hue_hist) * 2)
        
        # Check saturation distribution
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0,256])
        sat_hist = sat_hist / np.sum(sat_hist)
        
        # Real IDs should have controlled saturation
        sat_score = np.std(sat_hist) / np.mean(sat_hist[sat_hist > 0])
        
        # Combine scores
        return min(100, (dominant_hues / 180 * 50 + sat_score * 50))

    def _analyze_edge_quality(self, gray):
        '''
        Improved edge quality analysis
        '''
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density at different scales
        density1 = np.count_nonzero(edges1) / edges1.size
        density2 = np.count_nonzero(edges2) / edges2.size
        
        # Real IDs should have clear, sharp edges
        edge_ratio = abs(density1 - density2) / max(density1, density2)
        
        # Check edge continuity
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges1, kernel, iterations=1)
        continuity = np.count_nonzero(dilated - edges1) / max(np.count_nonzero(edges1), 1)
        
        # Combine metrics (higher continuity and consistent multi-scale edges are good)
        return min(100, (edge_ratio * 50 + continuity * 50))

    def _detect_cartoon(self, image, gray):
        '''
        Detect cartoon-like characteristics
        '''
        # 1. Edge detection for cartoon-like edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # 2. Color quantization check
        colors = image.reshape(-1, 3)
        unique_colors = np.unique(colors, axis=0)
        color_ratio = len(unique_colors) / (image.shape[0] * image.shape[1])
        
        # 3. Check for large uniform areas
        blur = cv2.medianBlur(gray, 5)
        diff = np.abs(gray - blur)
        uniform_ratio = np.sum(diff < 10) / diff.size
        
        # Combine scores (higher = more cartoon-like)
        cartoon_score = (
            edge_density * 100 +           # Strong edges
            (1 - color_ratio) * 100 +      # Few unique colors
            uniform_ratio * 100            # Large uniform areas
        ) / 3
        
        return min(100, cartoon_score)

    def _prepare_for_ocr(self, gray):
        '''
        Prepare image for OCR
        '''
        contrast = cv2.convertScaleAbs(gray, alpha=1.75, beta=0)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 5, 7, 21)
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _calculate_color_transitions(self, hsv):
        '''
        Calculate unnatural color transitions in HSV space
        '''
        # Calculate gradients in both directions
        gradient_x = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Look for suspicious patterns
        direction_hist = np.histogram(gradient_direction, bins=36)[0]
        direction_peaks = np.sum(direction_hist > np.mean(direction_hist) * 1.5)
        
        # Calculate mean gradient magnitude
        mean_magnitude = np.mean(gradient_magnitude)
        
        # Combine metrics (higher score = more suspicious transitions)
        transition_score = (mean_magnitude * 0.7 + direction_peaks * 0.3)
        
        return min(100, transition_score * 2)

    def _check_official_colors(self, hsv):
        '''
        Check if colors match typical official ID patterns
        '''
        # Define expected color ranges for official IDs
        official_hue_ranges = [(0, 20),    # Red range
                             (100, 140),   # Blue range
                             (20, 40)]     # Orange/Brown range
        
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue_hist = hue_hist / np.sum(hue_hist)
        
        # Check if dominant colors fall within official ranges
        official_color_ratio = 0
        for range_start, range_end in official_hue_ranges:
            official_color_ratio += np.sum(hue_hist[range_start:range_end])
        
        return min(100, (1 - official_color_ratio) * 100)

    def _detect_security_features(self, gray):
        '''
        Detect common security features in IDs
        '''
        # Look for fine detail patterns
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        detail = cv2.filter2D(gray, -1, kernel)
        
        # Check for microprint-like patterns
        detail_score = np.std(detail) / np.mean(detail)
        
        # Look for regular patterns (guilloche)
        fourier = np.fft.fft2(gray)
        fourier_shift = np.fft.fftshift(fourier)
        magnitude = np.abs(fourier_shift)
        
        # Check for regular pattern presence
        pattern_score = np.std(magnitude) / np.mean(magnitude)
        
        return min(100, 100 - ((detail_score + pattern_score) * 50))

    def _analyze_text_placement(self, gray):
        '''
        Analyze text placement patterns
        '''
        # Use horizontal projection to find text rows
        horizontal_proj = np.sum(gray < 128, axis=1)
        
        # Find peaks in projection (text lines)
        peaks = []
        threshold = np.mean(horizontal_proj) * 1.5
        for i in range(1, len(horizontal_proj) - 1):
            if horizontal_proj[i] > threshold:
                if horizontal_proj[i] > horizontal_proj[i-1] and horizontal_proj[i] > horizontal_proj[i+1]:
                    peaks.append(i)
        
        if len(peaks) < 2:
            return 100  # Suspicious if we can't find enough text lines
        
        # Calculate spacing between text lines
        spacings = np.diff(peaks)
        spacing_consistency = np.std(spacings) / np.mean(spacings)
        
        return min(100, spacing_consistency * 100)