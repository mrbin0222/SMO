import os

import cv2
import numpy as np
from util.loadresults import load_masks_and_scores


class FeatureFusion:
    def __init__(self):
        self.smo_weight = 0.6  # SMO feature weight
        self.cft_weight = 0.4  # CFT feature weight
    
    def extract_smo_features(self, mask_whole):
        """Extract SMO segmentation features"""
        mask=mask_whole['segmentation']
        x, y, w, h = [int(v) for v in mask_whole['bbox']]

        # Calculate relative position to image center
        center_y, center_x = np.array(mask.shape) / 2
        mask_center = np.mean(np.argwhere(mask), axis=0)
        rel_position = [mask_center[0]/center_y, mask_center[1]/center_x]

        # Calculate shape features
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate Hu moments
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Calculate aspect ratio using bbox
        width = w
        height = h
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        # Combine all features
        features = np.concatenate([
            [area, perimeter, circularity, aspect_ratio, *rel_position],
            hu_moments
        ])
        
        return features
    
    def extract_cft_features(self, patch):
        """Extract CFT related filtering features"""
        # Ensure image type is correct
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)
        
        # Resize image to fixed size
        patch = cv2.resize(patch, (32, 32))
        
        # Calculate simple texture features
        # 1. Calculate gradient
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        
        # 2. Calculate gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # 3. Quantize angle into 8 bins
        angle_bins = np.zeros(8)
        for i in range(8):
            mask = (angle >= i*45) & (angle < (i+1)*45)
            angle_bins[i] = np.sum(magnitude[mask])
        
        # 4. Calculate local statistical features
        mean = np.mean(patch)
        std = np.std(patch)
        
        # 5. Calculate color histogram
        hist = cv2.calcHist([patch], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Combine all features
        features = np.concatenate([
            angle_bins,  # 8D
            [mean, std],  # 2D
            hist  # 8D
        ])
        
        return features
    
    def fuse_features(self, smo_features, cft_features):
        """Fuse SMO and CFT features"""
        if smo_features is None or cft_features is None:
            return None
        
        # Normalize features
        smo_features = (smo_features - np.mean(smo_features)) / (np.std(smo_features) + 1e-6)
        cft_features = (cft_features - np.mean(cft_features)) / (np.std(cft_features) + 1e-6)
        
        # Calculate feature dimension ratio
        smo_dim = len(smo_features)
        cft_dim = len(cft_features)
        total_dim = smo_dim + cft_dim
        
        # Adjust weights based on dimension ratio
        smo_weight = smo_dim / total_dim
        cft_weight = cft_dim / total_dim
        
        # Weighted fusion
        fused_features = np.concatenate([
            smo_weight * smo_features,
            cft_weight * cft_features
        ])
        
        return fused_features
    



class LightCFTracker:
    def __init__(self, init_mask_whole, init_mask, init_bbox, frame_gray):
        """
        :param init_mask: Binary mask (True for cell region)
        :param init_bbox: Initial bounding box [x,y,w,h]
        :param frame_gray: First frame grayscale image (for template extraction)
        """
        self.bbox = init_bbox
        self.feature_fusion = FeatureFusion()
        self.parent_id = None  # Add parent cell ID attribute
        self.initial_area = init_mask.sum()  # Save initial area
        self.initial_mask = init_mask.copy()  # Save initial mask
        
        # Extract and store initial features
        smo_features = self.feature_fusion.extract_smo_features(init_mask_whole)
        cft_features = self.feature_fusion.extract_cft_features(frame_gray[init_bbox[1]:init_bbox[1]+init_bbox[3], 
                                                                 init_bbox[0]:init_bbox[0]+init_bbox[2]])
        self.initial_features = self.feature_fusion.fuse_features(smo_features, cft_features)
        
        self.history_features = {
            'area': init_mask.sum(),
            'aspect_ratio': init_bbox[2] / init_bbox[3],
            'center': (init_bbox[0] + init_bbox[2]/2, init_bbox[1] + init_bbox[3]/2),
            'frames': 1,
            'fused_features': self.initial_features,
            'mask': init_mask.copy()  # Store mask history
        }
        self.is_split = False
        self.split_confidence = 0.0
    
    def match(self, candidate_mask_whole, candidate_mask, candidate_bbox, candidate_patch):
        """Use feature fusion for matching"""
        # Extract candidate region features
        smo_features = self.feature_fusion.extract_smo_features(candidate_mask_whole)
        cft_features = self.feature_fusion.extract_cft_features(candidate_patch)
        candidate_features = self.feature_fusion.fuse_features(smo_features, cft_features)
        
        if candidate_features is None or self.history_features['fused_features'] is None:
            return 0.0, (0, 0)
        
        # Ensure feature dimension matches
        if candidate_features.shape != self.history_features['fused_features'].shape:
            min_dim = min(candidate_features.shape[0], self.history_features['fused_features'].shape[0])
            candidate_features = candidate_features[:min_dim]
            history_features = self.history_features['fused_features'][:min_dim]
        else:
            history_features = self.history_features['fused_features']
        
        # Calculate feature similarity
        feature_sim = np.corrcoef(
            candidate_features,
            history_features
        )[0, 1]
        
        # Calculate shape similarity
        shape_sim = self._compute_shape_similarity(
            candidate_mask,
            self.history_features['mask']
        )
        
        # Calculate position similarity
        center_dist = np.sqrt(
            (candidate_bbox[0] + candidate_bbox[2]/2 - self.history_features['center'][0])**2 +
            (candidate_bbox[1] + candidate_bbox[3]/2 - self.history_features['center'][1])**2
        )
        pos_sim = np.exp(-center_dist / 50.0)  # Distance closer, similarity higher
        
        # # Calculate area change
        # Improved area similarity calculation (add protection for zero values and asymmetric processing)
        if self.history_features['area'] == 0:
            area_sim = 0.0  # Directly determine no similarity when history area is 0
        else:
            area_ratio = candidate_mask.sum() / self.history_features['area']
            # Asymmetric threshold: allow larger area decrease (e.g. division)
            max_decrease = 0.9  # Allow 90% decrease (area_ratio lowest to 0.1)
            max_increase = 0.3  # Allow 30% increase (area_ratio highest to 1.3)
            
            if area_ratio > 1.0:  # Area increase
                delta = min(area_ratio - 1.0, max_increase)
                area_sim = 1.0 - delta / max_increase
            else:  # Area decrease
                delta = min(1.0 - area_ratio, max_decrease)
                area_sim = 1.0 - delta / max_decrease
        
        # Calculate aspect ratio change
        aspect_ratio = candidate_bbox[2] / candidate_bbox[3]
        aspect_diff = abs(aspect_ratio - self.history_features['aspect_ratio'])
        aspect_sim = np.exp(-aspect_diff)  
        
        # Overall score, adjust weights
        match_score = (
            0.45 * feature_sim +  # feature similarity weight
            0.10 * shape_sim +    # shape similarity weight
            0.15 * pos_sim +      # position similarity weight
            0.10 * area_sim +    # area similarity
            0.20 * aspect_sim    # aspect ratio similarity
        )
        
        # If position is very close, increase match score
        if center_dist < 20:  # If center point distance is less than 20 pixels
            match_score = max(match_score, 0.6)  # Ensure match score is at least 0.6
        
        # If area change is within reasonable range, also increase match score
        if 0.5 < area_ratio < 1.5:  # Area change within 50%
            match_score = max(match_score, 0.5)  # Ensure match score is at least 0.5
        
        return match_score, (0, 0)  # Return match score and position offset
    
    def _compute_shape_similarity(self, mask1, mask2):
        """Calculate shape similarity of two masks"""
        # Calculate contours
        contours1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 0.0
        
        # Get largest contour
        contour1 = max(contours1, key=cv2.contourArea)
        contour2 = max(contours2, key=cv2.contourArea)
        
        # Calculate shape matching (value smaller means more similar)
        match_value = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)
        
        # Convert match value to similarity score (value larger means more similar)
        # Use exponential function to map [0,∞) to (0,1]
        similarity = np.exp(-match_value)
        
        return similarity
    
    def update_features(self, new_mask_whole, new_mask, new_bbox, new_patch):
        """Update cell features"""
        # Extract new features
        smo_features = self.feature_fusion.extract_smo_features(new_mask_whole)
        cft_features = self.feature_fusion.extract_cft_features(new_patch)
        new_fused_features = self.feature_fusion.fuse_features(smo_features, cft_features)
        
        # Calculate feature similarity
        feature_similarity = 0
        if new_fused_features is not None and self.history_features['fused_features'] is not None:
            # Ensure feature dimension matches
            if new_fused_features.shape != self.history_features['fused_features'].shape:
                min_dim = min(new_fused_features.shape[0], self.history_features['fused_features'].shape[0])
                new_fused_features = new_fused_features[:min_dim]
                history_features = self.history_features['fused_features'][:min_dim]
            else:
                history_features = self.history_features['fused_features']
            
            # Calculate correlation coefficient
            feature_similarity = np.corrcoef(
                new_fused_features,
                history_features
            )[0, 1]
        
        # Update other features
        new_area = new_mask.sum()
        new_aspect = new_bbox[2] / new_bbox[3]
        new_center = (new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2)
        
        # Calculate feature change
        area_ratio = new_area / self.history_features['area']
        aspect_diff = abs(new_aspect - self.history_features['aspect_ratio'])
        center_dist = np.sqrt((new_center[0] - self.history_features['center'][0])**2 + 
                            (new_center[1] - self.history_features['center'][1])**2)
        
        # Update history features
        self.history_features.update({
            'area': new_area,
            'aspect_ratio': new_aspect,
            'center': new_center,
            'frames': self.history_features['frames'] + 1,
            'fused_features': new_fused_features,
            'mask': new_mask.copy()  # Update mask history
        })
        
        return area_ratio, aspect_diff, center_dist, feature_similarity
    
    def check_split(self, new_mask_whole, new_mask, new_bbox, new_patch, nearby_masks):
        """Check if cell division has occurred"""
        area_ratio, aspect_diff, center_dist, feature_similarity = self.update_features(
            new_mask_whole, new_mask, new_bbox, new_patch
        )
        
        # Calculate division confidence
        split_score = 0.0
        
        # 1. Area change (using initial area)
        current_area = new_mask.sum()
        if current_area < self.initial_area * 0.80:
            split_score += 0.3

        # 2. Add cell count logic (using initial area)
        valid_children = [m for m in nearby_masks if 
                        (0.1 < m['area']/self.initial_area < 0.80)]
        if len(valid_children) >= 1:
            split_score += 0.2

        # 3. Feature similarity
        if feature_similarity < 0.6:  # Large feature difference
            split_score += 0.1
        
        # 4. Shape change
        if aspect_diff > 0.3:
            split_score += 0.2
        
        # 5. Position change
        if center_dist > 20:
            split_score += 0.1
        
        # 6. Tracking duration
        if self.history_features['frames'] > 1:
            split_score += 0.1
        
        self.split_confidence = split_score
        self.is_split = split_score > 0.5
        
        # Add debug output before return
        debug_info = {
            'frame': self.history_features['frames'],
            'initial_area': self.initial_area,
            'current_area': current_area,
            'area_ratio': current_area/self.initial_area,
            'aspect_diff': aspect_diff,
            'feature_sim': feature_similarity,
            'nearby_count': len(nearby_masks)
        }
        print(f"Division detection debug -> ID:{id(self)} Score:{split_score:.2f} Details:{debug_info}")

        return self.is_split

    def update_template(self, new_patch, learning_rate=0.1):
        """Incremental update template"""
        # Adjust new patch size to match template
        if new_patch.shape != self.template.shape:
            new_patch = cv2.resize(new_patch, self.template.shape[::-1])
        
        # Linear mixing update
        self.template = (1 - learning_rate) * self.template + learning_rate * new_patch
        self.template = np.clip(self.template, 0, 255).astype(np.uint8)

class HybridTracker:
    def __init__(self):
        self.trackers = {}  # {track_id: LightCFTracker}
        self.next_id = 0
        self.split_history = []  # Record division events
    
    def update(self, frame_gray, smo_masks):
        """Process single frame: match existing trackers and detect division"""
        current_objects = {tuple(m['bbox']): m for m in smo_masks}
        
        # Match existing trackers
        for tid in list(self.trackers.keys()):
            tracker = self.trackers[tid]
            
            # Find best match in search region
            best_score = -np.inf
            best_match = None
            
            for c_bbox, mask in current_objects.items():
                # Check if in search region
                if self._is_in_search_region(tracker.bbox, list(c_bbox)):
                    # Get candidate region
                    x, y, w, h = [int(v) for v in c_bbox]
                    candidate_patch = frame_gray[y:y+h, x:x+w]
                    
                    # Calculate match score
                    score, _ = tracker.match(mask, mask['segmentation'], list(c_bbox), candidate_patch)
                    
                    if score > best_score:
                        best_score = score
                        best_match = mask
            
            if best_score > 0.4:  # Lower match threshold for more stable tracking
                # Update tracker
                tracker.bbox = best_match['bbox']
                x, y, w, h = [int(v) for v in tracker.bbox]
                new_patch = frame_gray[y:y+h, x:x+w]
                
                # Find nearby cells
                nearby_masks = self._find_nearby_masks(best_match['bbox'], current_objects)

                # Add best_match to nearby_masks
                nearby_masks.append(best_match)
                
                # Check for division
                if tracker.check_split(best_match, best_match['segmentation'], best_match['bbox'], new_patch, nearby_masks):
                    # Use tracker's initial mask as parent cell mask
                    parent_mask = {
                        'segmentation': tracker.initial_mask,
                        'bbox': tracker.bbox,
                        'area': tracker.initial_area
                    }
                    self._handle_split(frame_gray, tid, parent_mask, nearby_masks)
            else:
                # If no good match found, don't delete tracker immediately
                # Wait for a few frames in case of temporary occlusion or deformation
                if hasattr(tracker, 'missed_frames'):
                    tracker.missed_frames += 1
                    if tracker.missed_frames > 3:  # Delete after 3 consecutive frames without match
                        del self.trackers[tid]
                else:
                    tracker.missed_frames = 1
        
        # Add new targets
        for bbox, mask in current_objects.items():
            if not any(self._bbox_iou(list(bbox), t.bbox) > 0.3 for t in self.trackers.values()):
                new_tracker = LightCFTracker(mask, mask['segmentation'], list(bbox), frame_gray)
                self.trackers[self.next_id] = new_tracker
                self.next_id += 1
    
    def _is_in_search_region(self, bbox1, bbox2, scale=1.5):
        """Check if bbox2 is in the search region of bbox1"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate search region
        search_x1 = max(0, x1 - w1 * (scale-1)/2)
        search_y1 = max(0, y1 - h1 * (scale-1)/2)
        search_x2 = x1 + w1 * (scale+1)/2
        search_y2 = y1 + h1 * (scale+1)/2
        
        # Check if center point of bbox2 is in the search region
        center_x = x2 + w2/2
        center_y = y2 + h2/2
        
        return (search_x1 <= center_x <= search_x2 and 
                search_y1 <= center_y <= search_y2)
    
    
    def _bbox_iou(self, box1, box2):
        """Calculate IoU of two bboxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi = max(x1, x2)
        yi = max(y1, y2)
        ai = min(x1 + w1, x2 + w2)
        bi = min(y1 + h1, y2 + h2)
        inter = max(0, ai - xi) * max(0, bi - yi)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-8)
    
    def _find_nearby_masks(self, bbox, current_objects, max_dist=60):
        """Find nearby cells"""
        nearby = []
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        
        for c_bbox, mask in current_objects.items():
            cx, cy, cw, ch = c_bbox
            c_center = (cx + cw/2, cy + ch/2)
            dist = np.sqrt((center[0] - c_center[0])**2 + (center[1] - c_center[1])**2)
            
            # Check if it is the same bbox (by comparing center points)
            is_same_bbox = (abs(center[0] - c_center[0]) < 1e-6 and 
                          abs(center[1] - c_center[1]) < 1e-6)
            
            if dist < max_dist and not is_same_bbox:
                nearby.append(mask)
        
        return nearby
    
    def _handle_split(self, frame_gray, parent_id, parent_mask, nearby_masks):
        # Add validation conditions
        valid_children = []
        parent_area = parent_mask['area']
        
        for child in nearby_masks:
            # Child cell area should be between 10%-80% of parent cell
            if 0.1 < child['area']/parent_area < 0.8:
                # Child cell should be close to parent cell centroid
                parent_center = (parent_mask['bbox'][0]+parent_mask['bbox'][2]/2,
                            parent_mask['bbox'][1]+parent_mask['bbox'][3]/2)
                child_center = (child['bbox'][0]+child['bbox'][2]/2,
                            child['bbox'][1]+child['bbox'][3]/2)
                dist = np.linalg.norm(np.array(parent_center)-np.array(child_center))
                if dist < parent_mask['bbox'][3]*2:  # Distance less than 2 times parent cell height
                    valid_children.append(child)

        if len(valid_children) >= 1:
            """Handle division event"""
            # Record division event
            self.split_history.append({
                'frame': len(self.split_history),
                'parent_id': parent_id,
                'parent_bbox': parent_mask['bbox'],
                'children': [m['bbox'] for m in valid_children]  # Only record valid child cells
            })
            
            # Create child trackers
            for child in valid_children:  # Only process valid child cells
                new_tracker = LightCFTracker(child, child['segmentation'], child['bbox'], frame_gray)
                new_tracker.is_split = True
                new_tracker.parent_id = parent_id  # Record parent cell ID
                new_tracker.split_confidence = self.trackers[parent_id].split_confidence  # Inherit division confidence
                self.trackers[self.next_id] = new_tracker
                self.next_id += 1
        
            # Remove parent tracker
            del self.trackers[parent_id]

        else:
            print(f"False division alert: Parent cell {parent_id} found {len(valid_children)} valid child cells nearby")

def main():
    # Set paths
    im_path = "./datasets/B3BF/8910/"
    results_path = "./layoutandtext/results/"
    
    # Get image list
    img_files = sorted([f for f in os.listdir(im_path) if f.endswith('.jpeg')])
    video_frames = [cv2.imread(os.path.join(im_path, f)) for f in img_files]

    # Initialize tracker
    hybrid_tracker = HybridTracker()

    # Process stream
    for i, frame in enumerate(video_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get SMO segmentation results
        frame_name = img_files[i].split('.')[0]
        smo_masks, _ = load_masks_and_scores(frame_name, load_dir=results_path)
        
        # Update tracker
        hybrid_tracker.update(gray, smo_masks)
        
        # Visualization
        vis_frame = frame.copy()
        for tid, tracker in hybrid_tracker.trackers.items():
            x, y, w, h = tracker.bbox
            # Choose color and line width based on division status
            if tracker.is_split:
                color = (0, 0, 255)  # Red
                thickness = 2  # Thicker line
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, thickness)
            
            # Display ID and division information
            if tracker.is_split:
                status = f"ID:{tid} (Parent:{tracker.parent_id}) Split:{tracker.split_confidence:.2f}"
            else:
                status = f"ID:{tid}"
            
            # Add text background for better readability
            text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_frame, (x, y-text_size[1]-5), (x+text_size[0], y), (0, 0, 0), -1)
            cv2.putText(vis_frame, status, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show results
        cv2.imshow("Hybrid Tracking", vis_frame)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Print division history
    print("\nDivision event history:")
    for event in hybrid_tracker.split_history:
        print(f"Frame {event['frame']}: Parent cell {event['parent_id']} divided into {len(event['children'])} child cells")

if __name__ == "__main__":
    main()

