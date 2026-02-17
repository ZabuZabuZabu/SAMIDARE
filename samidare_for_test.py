import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import math
import os
import glob
from pathlib import Path

import argparse
from torch.cuda.amp import autocast

@dataclass
class TrackState:
    RELIABLE = "reliable"
    PENDING = "pending"
    SUSPICIOUS = "suspicious"
    LOST = "lost"
    FRAME_OUT = "frame_out"  # 🆕 新規追加

@dataclass
class GTDetection:
    frame_id: int
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    visibility: float

@dataclass
class Track:
    id: int
    bbox: np.ndarray
    mask: Optional[np.ndarray]
    logits_score: float
    state: str
    lost_frames: int
    age: int
    logits_history: deque
    sam2_predictor: Any
    last_seen_frame: int
    init_frame: int
    skip_memory_current: bool = False
    memory_window: int = 7
    prev_bbox: Optional[np.ndarray] = None  # 🆕 前フレームのBBOX保存
    # 🆕 フレームアウト / 密集度のための情報
    is_dense: bool = False
    last_matched_frame: Optional[int] = None
    last_matched_bbox: Optional[np.ndarray] = None
    last_matched_density: float = 0.0

class GTLoader:
    """Ground Truth loader for MOT format"""
    
    def __init__(self, gt_path: str):
        self.gt_path = gt_path
        self.gt_data = self._load_gt()
    
    def _load_gt(self) -> Dict[int, List[GTDetection]]:
        """Load GT file and organize by frame"""
        gt_data = defaultdict(list)
        
        with open(self.gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                confidence = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])
                
                bbox = np.array([x, y, x + w, y + h])
                
                detection = GTDetection(
                    frame_id=frame_id,
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    visibility=visibility
                )
                
                gt_data[frame_id].append(detection)
        
        return gt_data
    
    def get_detections(self, frame_id: int) -> List[GTDetection]:
        """Get detections for specific frame"""
        return self.gt_data.get(frame_id, [])

class TrajectoryManagerSystem:
    """論文のTrajectory Manager Systemの実装"""
    
    def __init__(self, 
                 tau_r: float = 8,
                 tau_p: float = 1,
                 tau_s: float = 0,
                 tolerance_frames: int = 25,
                 untracked_ratio_threshold: float = 0.5):
        self.tau_r = tau_r
        self.tau_p = tau_p
        self.tau_s = tau_s
        self.tolerance_frames = tolerance_frames
        self.untracked_ratio_threshold = untracked_ratio_threshold
    
    def classify_track_state(self, logits_score: float) -> str:
        """Equation 2: Track state classification"""
        if logits_score > self.tau_r:
            return TrackState.RELIABLE
        elif logits_score > self.tau_p:
            return TrackState.PENDING
        elif logits_score > self.tau_s:
            return TrackState.SUSPICIOUS
        else:
            return TrackState.LOST
    
    def compute_untracked_mask(self, frame_shape: Tuple[int, int], tracked_masks: List[np.ndarray]) -> np.ndarray:
        """Equation 1: Compute untracked region mask"""
        H, W = frame_shape
        untracked_mask = np.ones((H, W), dtype=np.uint8)
        
        for mask in tracked_masks:
            if mask is not None:
                untracked_mask = untracked_mask & (~mask.astype(np.uint8))
        
        return untracked_mask
    
    def should_add_detection(self, detection: GTDetection, untracked_mask: np.ndarray) -> bool:
        """Check if detection should be added as new track"""
        x1, y1, x2, y2 = detection.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(untracked_mask.shape[1], x2), min(untracked_mask.shape[0], y2)
        
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area <= 0:
            return False
        
        roi = untracked_mask[y1:y2, x1:x2]
        overlap_pixels = np.sum(roi)
        overlap_ratio = overlap_pixels / bbox_area
        
        return overlap_ratio > self.untracked_ratio_threshold
    
    def should_remove_track(self, track: Track) -> bool:
        """Check if track should be removed"""
        return track.lost_frames > self.tolerance_frames

    def should_reconstruct_quality(self, track: Track, matched_detection: Optional[GTDetection]) -> bool:
        """Check if track quality should be reconstructed"""
        if track.state != TrackState.PENDING:
            return False
        
        if matched_detection is None or matched_detection.confidence < 0.7:
            return False
        
        return True

class CrossObjectInteraction:
    """論文の Cross-object Interaction Module の実装"""
    
    def __init__(self, 
                 miou_threshold: float = 0.8, 
                 memory_history_frames: int = 7, 
                 variance_history: int = 10,
                 logits_margin: float = 1.2):
        self.miou_threshold = miou_threshold
        self.memory_history_frames = memory_history_frames
        self.variance_history = variance_history
        self.logits_margin = logits_margin
    
    def compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Equation 3: Compute mask IoU"""
        if mask1 is None or mask2 is None:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def compute_logits_variance(self, logits_history: deque) -> float:
        """Equation 4: Compute logits score variance"""
        if len(logits_history) < 2:
            return 0.0
        
        scores = list(logits_history)[-self.variance_history:]
        mean_score = np.mean(scores)
        variance = float(np.mean([(score - mean_score) ** 2 for score in scores]))
        
        return variance

    def compute_logits_mean(self, logits_history: deque) -> float:
        """Equation 4: Compute logits score mean"""
        if len(logits_history) < 2:
            return 0.0
        
        scores = list(logits_history)[-self.variance_history:]
        mean_score = np.mean(scores)
        
        return mean_score

    def compute_masd(self, logits_history: deque) -> float:
        """
        隣り合う要素の差分の絶対値の平均を計算する
        """
        if len(logits_history) < 2:
            return 0.0  # データが2個未満なら変化は0
        
        scores = list(logits_history)[-self.variance_history:]
        abs_diffs = np.abs(np.diff(scores))
        return np.mean(abs_diffs)

    def detect_occlusion_and_resolve(self, tracks: List[Track], current_frame_idx: int) -> List[int]:
        """
        オクルージョン検出と解決：
        2. 平均と分散のうち、差が顕著な方を判定指標として採用
        """
        tracks_to_reconstruct = []
        coi_pairs = []
        n = len(tracks)
        
        for i in range(n):
            for j in range(i+1, n):
                track_a = tracks[i]
                track_b = tracks[j]
                
                if track_a.mask is None or track_b.mask is None:
                    continue
                
                if track_a.state == TrackState.FRAME_OUT or track_b.state == TrackState.FRAME_OUT:
                    continue
                
                miou = self.compute_mask_iou(track_a.mask, track_b.mask)
                if miou <= self.miou_threshold:
                    continue

                # 各指標の計算
                mean_a = self.compute_logits_mean(track_a.logits_history)
                mean_b = self.compute_logits_mean(track_b.logits_history)
                var_a = self.compute_logits_variance(track_a.logits_history)
                var_b = self.compute_logits_variance(track_b.logits_history)

                # # --- STEP 2: 平均と分散の差を比較し、指標を動的に選択 ---
                # # 単位が異なるため、最大値で割って相対的な差（0.0 ~ 1.0）を算出
                # diff_mean_rel = abs(mean_a - mean_b) / (max(mean_a, mean_b, 1e-6))
                # diff_var_rel = abs(var_a - var_b) / (max(var_a, var_b, 1e-6))

                # if diff_mean_rel >= diff_var_rel:
                #     # 平均の差の方が顕著な場合：平均が低い方をオクルージョンとする
                #     occluded_idx = i if mean_a < mean_b else j
                #     reason = f"selected_metric: Mean (diff_rel={diff_mean_rel:.2f}), val: {mean_a:.2f} vs {mean_b:.2f}"
                # else:
                #     # 分散の差の方が顕著な場合：分散が高い（不安定な）方をオクルージョンとする
                #     occluded_idx = i if var_a > var_b else j
                #     reason = f"selected_metric: Variance (diff_rel={diff_var_rel:.2f}), val: {var_a:.2f} vs {var_b:.2f}"

                # --- STEP 2: 平均と分散の差を比較し、指標を動的に選択 ---
                diff_mean = abs(mean_a - mean_b)
                diff_var = abs(var_a - var_b)            

                if diff_mean >= diff_var:
                    # 平均の差の方が顕著な場合：平均が低い方をオクルージョンとする
                    occluded_idx = i if mean_a < mean_b else j
                    reason = f"selected_metric: Mean (diff_rel={diff_mean:.2f}), val: {mean_a:.2f} vs {mean_b:.2f}"
                else:
                    # 分散の差の方が顕著な場合：分散が高い（不安定な）方をオクルージョンとする
                    occluded_idx = i if var_a > var_b else j
                    reason = f"selected_metric: Variance (diff_rel={diff_var:.2f}), val: {var_a:.2f} vs {var_b:.2f}"

                occluded_track = tracks[occluded_idx]
                
                id_a = track_a.id
                id_b = track_b.id
                skipped_id = occluded_track.id
                coi_pairs.append((id_a, id_b, skipped_id))

                occluded_track.skip_memory_current = True

                if occluded_track.id not in tracks_to_reconstruct:
                    tracks_to_reconstruct.append(occluded_track.id)
                    print(f"[CoI] Frame {current_frame_idx}: marked track {occluded_track.id} "
                          f"as occluded (mIoU={miou:.3f}, {reason}), will skip memory")

        return tracks_to_reconstruct, coi_pairs

class SAM2MOT:
    """SAM2MOT: Sequential frame-by-frame processing for dynamic object addition"""
    
    def __init__(self, 
                 sam2_predictor,
                 gt_loader: GTLoader,
                 device: str = "cuda",
                 memory_window: int = 25,
                 tolerance_frames: int = 60,
                 cost_weight: float = 0.5,
                 tau_r: float = 8.0,
                 tau_p: float = 1.0,
                 tau_s: float = 0.0,
                 density_threshold: float = 1.5,
                 second_stage_iou_threshold: float = 0.0,
                 frame_out_d_thre: float = 0.6): 
        self.sam2_predictor = sam2_predictor
        self.gt_loader = gt_loader
        self.device = device
        self.tolerance_frames = tolerance_frames
        self.memory_window = memory_window
        self.cost_weight = cost_weight
        self.density_threshold = density_threshold
        self.second_stage_iou_threshold = second_stage_iou_threshold  # 🆕
        self.frame_out_d_thre = frame_out_d_thre
        
        if hasattr(sam2_predictor, 'model'):
            sam2_predictor.model = sam2_predictor.model.float()
        
        self.trajectory_manager = TrajectoryManagerSystem(
            tau_r=tau_r,
            tau_p=tau_p,
            tau_s=tau_s,
            tolerance_frames=self.tolerance_frames
        )
        self.cross_object_interaction = CrossObjectInteraction(
            logits_margin=1.2
        )
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        self.tracking_results = []
        self.id_map = {}
        
        self.propagation_iterator = None
        self.track_colors = {}

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate or retrieve a unique color for a given track ID."""
        if track_id not in self.track_colors:
            hue = int((track_id * 0.61803398875 * 180) % 180)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            self.track_colors[track_id] = tuple(map(int, color_bgr))
        return self.track_colors[track_id]
    
    def compute_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def compute_density(self, target_det: GTDetection, all_detections: List[GTDetection]) -> float:
        """密集度：他物体とのOverlapの和（自分の面積基準）"""
        x1, y1, x2, y2 = target_det.bbox
        target_area = max((x2 - x1) * (y2 - y1), 1e-6)
        density = 0.0

        for other in all_detections:
            if np.allclose(other.bbox, target_det.bbox):
                continue

            ox1, oy1, ox2, oy2 = other.bbox
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            overlap_ratio = inter_area / target_area
            density += overlap_ratio

        return density

    def mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to bounding box"""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return np.array([0, 0, 1, 1])
        
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        return np.array([x1, y1, x2 + 1, y2 + 1])
    
    def initialize_sam2_tracker(self, frame: np.ndarray, bbox: np.ndarray, track_id: int, inference_state, frame_idx: int):
        """Initialize SAM2 tracker for a new object at specific frame"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=track_id,
                points=None,
                labels=None,
                box=np.array([x1, y1, x2, y2], dtype=np.float32)
            )
            print(f"✓ Initialized track {track_id} at frame {frame_idx}")
            return True
                
        except Exception as e:
            print(f"✗ Error initializing track {track_id}: {e}")
            return False

    def hungarian_matching(self, gt_detections: List[GTDetection], tracks: List[Track], 
                          use_prev_bbox: bool = False):
        """
        Hungarian matching using both IoU and mean logit score
        
        Args:
            gt_detections: 検出リスト
            tracks: トラックリスト
            use_prev_bbox: Trueの場合、track.prev_bboxを使用（2段階目用）
        """
        if len(gt_detections) == 0 or len(tracks) == 0:
            return [], list(range(len(gt_detections))), list(range(len(tracks)))

        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(gt_detections), len(tracks)))

        for i, det in enumerate(gt_detections):
            for j, track in enumerate(tracks):
                # 🆕 2段階目ではprev_bboxを使用
                if use_prev_bbox and track.prev_bbox is not None:
                    iou = self.compute_bbox_iou(det.bbox, track.prev_bbox)
                else:
                    iou = self.compute_bbox_iou(det.bbox, track.bbox)

                if iou == 0:
                    cost_matrix[i, j] = 1.0
                    continue

                if len(track.logits_history) > 0:
                    mean_logit = np.mean(track.logits_history)
                else:
                    mean_logit = 0.0

                norm_mean_logit = np.clip(mean_logit / 20.0, 0.0, 1.0)
                cost_matrix[i, j] = (1 - iou) * self.cost_weight + (1 - norm_mean_logit) * (1 - self.cost_weight)

        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        matches, unmatched_detections, unmatched_tracks = [], [], []
        matched_det_indices = set()
        matched_track_indices = set()

        for i, j in zip(det_indices, track_indices):
            if cost_matrix[i, j] < 1.0:
                matches.append((i, j))
                matched_det_indices.add(i)
                matched_track_indices.add(j)

        unmatched_detections = [i for i in range(len(gt_detections)) if i not in matched_det_indices]
        unmatched_tracks = [j for j in range(len(tracks)) if j not in matched_track_indices]

        return matches, unmatched_detections, unmatched_tracks

    # 🆕 2段階マッチング用の新メソッド
    def two_stage_matching(self, gt_detections: List[GTDetection], tracks: List[Track]):
        """
        2段階マッチング
        1段階目: 現在のBBOXでマッチング
        2段階目: 1段階目で未マッチのトラックを前フレームBBOXでマッチング
        
        Returns:
            matches: [(det_idx, track_idx), ...]
            unmatched_detections: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
            second_stage_matches: [(det_idx, track_idx), ...] # 2段階目でマッチしたペア
        """
        # === 1段階目: 現在のBBOXでマッチング ===
        matches_stage1, unmatched_dets_stage1, unmatched_tracks_stage1 = self.hungarian_matching(
            gt_detections, tracks, use_prev_bbox=False
        )
        
        if len(unmatched_tracks_stage1) == 0 or len(unmatched_dets_stage1) == 0:
            # 2段階目不要
            return matches_stage1, unmatched_dets_stage1, unmatched_tracks_stage1, []
        
        # === 2段階目: 未マッチトラックを前フレームBBOXでマッチング ===
        unmatched_tracks_objs = [tracks[i] for i in unmatched_tracks_stage1]
        unmatched_dets_objs = [gt_detections[i] for i in unmatched_dets_stage1]
        
        # prev_bboxが存在するトラックのみ対象
        valid_unmatched_tracks = []
        valid_track_indices = []
        for idx, track in zip(unmatched_tracks_stage1, unmatched_tracks_objs):
            if track.prev_bbox is not None:
                valid_unmatched_tracks.append(track)
                valid_track_indices.append(idx)
        
        if len(valid_unmatched_tracks) == 0:
            return matches_stage1, unmatched_dets_stage1, unmatched_tracks_stage1, []
        
        matches_stage2, unmatched_dets_stage2_local, unmatched_tracks_stage2_local = self.hungarian_matching(
            unmatched_dets_objs, valid_unmatched_tracks, use_prev_bbox=True
        )
        
        # インデックスを元のリストに戻す
        second_stage_matches = []
        for det_local_idx, track_local_idx in matches_stage2:
            original_det_idx = unmatched_dets_stage1[det_local_idx]
            original_track_idx = valid_track_indices[track_local_idx]
            
            # IoU閾値チェック（2段階目専用の低めの閾値）
            det = gt_detections[original_det_idx]
            track = tracks[original_track_idx]
            iou = self.compute_bbox_iou(det.bbox, track.prev_bbox)
            
            if iou > self.second_stage_iou_threshold:
                second_stage_matches.append((original_det_idx, original_track_idx))
        
        # 最終的な未マッチリストを更新
        matched_det_indices_stage2 = {m[0] for m in second_stage_matches}
        matched_track_indices_stage2 = {m[1] for m in second_stage_matches}
        
        final_unmatched_dets = [d for d in unmatched_dets_stage1 if d not in matched_det_indices_stage2]
        final_unmatched_tracks = [t for t in unmatched_tracks_stage1 if t not in matched_track_indices_stage2]
        
        all_matches = matches_stage1 + second_stage_matches
        
        print(f"  📊 1st stage: {len(matches_stage1)} matches | "
              f"2nd stage: {len(second_stage_matches)} matches (from prev_bbox)")
        
        return all_matches, final_unmatched_dets, final_unmatched_tracks, second_stage_matches

    def frame_out_matching(self,
                           frame_out_tracks: List[Track],
                           frame_out_track_indices_in_active: List[int],
                           gt_detections: List[GTDetection],
                           unmatched_detections_indices: List[int],
                           inference_state,
                           frame_idx: int):
        """
        フレームアウトトラック専用マッチング：
        - frame_out_tracks: 未マッチとなった「フレームアウト」トラック群（Trackオブジェクトリスト）
        - frame_out_track_indices_in_active: それぞれのトラックが active_tracks の何番目か（元インデックス）
        - gt_detections: 現フレームの検出リスト
        - unmatched_detections_indices: 現在未マッチの検出インデックス（gt_detections のインデックス）

        マッチング基準：**フレームアウトトラックの last_matched_bbox と検出 bbox の IoU のみ**
        マッチ成功したら即リコンストラクション（SAM2 に box を投げる）。成功したトラックは次フレームから通常マッチに戻る。
        戻り値:
          matched_pairs: list of (det_idx, active_track_idx)
          remaining_unmatched_dets: updated list of unmatched detection indices
        """
        matched_pairs = []
        if len(frame_out_tracks) == 0 or len(unmatched_detections_indices) == 0:
            return matched_pairs, unmatched_detections_indices

        # build local unmatched detection objects
        local_dets = [gt_detections[i] for i in unmatched_detections_indices]

        # cost matrix using IoU between track.last_matched_bbox and det.bbox
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.ones((len(local_dets), len(frame_out_tracks)), dtype=float)

        for i, det in enumerate(local_dets):
            for j, track in enumerate(frame_out_tracks):
                if track.last_matched_bbox is None:
                    cost_matrix[i, j] = 1.0
                    continue

                iou = self.compute_bbox_iou(det.bbox, track.last_matched_bbox)
                # cost = 1 - iou
                cost_matrix[i, j] = 1.0 - iou

        det_indices, tr_indices = linear_sum_assignment(cost_matrix)

        # accept only matches with iou > 0 (cost < 1.0)
        used_det_local = set()
        used_track_local = set()
        for di, tj in zip(det_indices, tr_indices):
            if cost_matrix[di, tj] < 1.0:
                # map back to original indices
                original_det_idx = unmatched_detections_indices[di]
                original_track_active_idx = frame_out_track_indices_in_active[tj]
                matched_pairs.append((original_det_idx, original_track_active_idx))
                used_det_local.add(original_det_idx)
                used_track_local.add(original_track_active_idx)

        # update remaining unmatched detections
        remaining_unmatched = [d for d in unmatched_detections_indices if d not in used_det_local]

        # perform reconstruction for each matched pair
        for det_idx, active_track_idx in matched_pairs:
            det = gt_detections[det_idx]
            track = None
            # find track object by active index in current self.tracks: note caller passed active index mapping
            for t in self.tracks:
                if t.age >= 0 and t.id is not None:
                    pass
            # in caller we only have active_tracks mapping; to be safe, find track by matching prev_bbox/last_matched_bbox
            # better: caller provides mapping; here we'll just search for track whose last_matched_bbox equals one in frame_out_tracks

        # Note: actual reconstruction and id mapping will be done in caller where active_tracks and indices are available.
        # Here simply return the matched pairs and remaining unmatched detections.
        return matched_pairs, remaining_unmatched

    def process_sam2_predictions(self, frame_idx: int, obj_ids: List[int], 
                                masks: List[torch.Tensor]) -> Dict[int, Tuple[np.ndarray, float]]:
        """Process SAM2 predictions for a single frame"""
        predictions = {}
        
        for obj_id, logit_map in zip(obj_ids, masks):
            if logit_map is not None:
                logits_map_np = logit_map.cpu().numpy().squeeze()
                binary_mask = (logits_map_np > 0.0).astype(np.uint8)
                
                if np.any(binary_mask):
                    logits_score = float(np.mean(logits_map_np[binary_mask == 1]))
                else:
                    logits_score = -10.0

                predictions[obj_id] = (binary_mask, logits_score)
                
                if frame_idx < 3 or frame_idx % 10 == 0:
                    print(f"  [Frame {frame_idx}] Obj {obj_id}: Score={logits_score:.2f}, Area={binary_mask.sum()}")
                    
        return predictions
    
    def remove_occluded_frame_memory(self, inference_state, obj_id: int, frame_idx: int) -> bool:
        """オクルージョンフレームのメモリを完全に削除"""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is None:
            print(f"      ⚠️  Object {obj_id} not found in inference_state")
            return False
        
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        deleted = False
        
        if obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None) is not None:
            print(f"      ✓ Deleted non_cond memory for obj {obj_id} at frame {frame_idx}")
            deleted = True
        
        if frame_idx in obj_output_dict["cond_frame_outputs"]:
            out = obj_output_dict["cond_frame_outputs"][frame_idx]
            
            if out.get("maskmem_features") is not None:
                out["maskmem_features"] = None
                print(f"      ⚠️  Removed maskmem from cond_frame for obj {obj_id} at frame {frame_idx}")
                deleted = True
            
            if out.get("maskmem_pos_enc") is not None:
                out["maskmem_pos_enc"] = None
                deleted = True
        
        if inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None) is not None:
            print(f"      ✓ Removed tracking state for obj {obj_id} at frame {frame_idx}")
            deleted = True
        
        if deleted:
            torch.cuda.empty_cache()
        
        return deleted

    def cleanup_old_memory(self, inference_state, current_frame_idx: int):
        """常に最大 memory_window フレーム分のメモリを保持"""
        if not hasattr(inference_state, 'output_dict'):
            return

        frame_keys = sorted(list(inference_state['output_dict'].keys()))

        while len(frame_keys) > self.memory_window:
            oldest = frame_keys.pop(0)
            del inference_state['output_dict'][oldest]

        if hasattr(inference_state, 'non_cond_frame_outputs'):
            non_cond_keys = sorted(list(inference_state['non_cond_frame_outputs'].keys()))
            while len(non_cond_keys) > self.memory_window:
                oldest = non_cond_keys.pop(0)
                del inference_state['non_cond_frame_outputs'][oldest]

        if len(frame_keys) > 0:
            print(f"  🧹 Memory trimmed: kept last {len(frame_keys)} frames (limit={self.memory_window})")

        torch.cuda.empty_cache()

    def track_frame_sequential(self, frame: np.ndarray, frame_id: int, frame_idx: int, 
                            inference_state) -> Tuple[List[Dict], bool]:
        self.frame_count = frame_id
        gt_detections = self.gt_loader.get_detections(frame_id)
        
        sam2_predictions = {}
        iterator_needs_reset = False
        reset_reasons = []

        frame_debug_info = {
            "coi_ids": [],
            "dense_skip": [],
            "frame_out_ids": [],
            "frame_out_recovered_ids": [],
            "reconstructed_ids": [],
            "coi_pairs": []
        }
        
        if frame_idx % 5 == 0:
            self.cleanup_old_memory(inference_state, frame_idx)
        
        # SAM2予測取得
        try:
            if self.propagation_iterator is not None:
                try:
                    pred_frame_idx, obj_ids, masks = next(self.propagation_iterator)
                    sam2_predictions = self.process_sam2_predictions(pred_frame_idx, obj_ids, masks)
                except StopIteration:
                    self.propagation_iterator = None
        except Exception as e:
            self.propagation_iterator = None
        
        # トラック更新前にprev_bboxを保存
        for track in self.tracks:
            track.prev_bbox = track.bbox.copy() if track.bbox is not None else None
        
        # 既存トラック更新
        for track in self.tracks:
            track.age += 1
            obj_id = self.id_map.get(track.id, None)
            
            if obj_id is not None and obj_id in sam2_predictions:
                mask, logits_score = sam2_predictions[obj_id]
                track.mask = mask
                track.bbox = self.mask_to_bbox(mask)
                track.logits_score = logits_score
                track.last_seen_frame = frame_id
                track.logits_history.append(logits_score)
                # track.state = self.trajectory_manager.classify_track_state(logits_score)

                pred_state = self.trajectory_manager.classify_track_state(logits_score)
                # もしLOST判定でも、即座に殺さず SUSPICIOUS（要審議）扱いで生かしておく
                if pred_state == TrackState.LOST:
                    track.state = TrackState.SUSPICIOUS#マスクがないからLOSTではなく，マッチが一定の期間されなかったときにLOSTにするため．
                else:
                    track.state = pred_state
        
        active_tracks = [t for t in self.tracks if t.state != TrackState.LOST]

        # === フレームアウト候補を先に特定（2段階マッチング前） ===
        frame_out_candidates = []
        frame_out_candidate_indices = []
        normal_active_tracks = []
        normal_track_indices_in_all = []

        for idx, tr in enumerate(active_tracks):
            # フレームアウト候補条件：前フレームでマッチ + 非密集
            if (tr.last_matched_frame is not None and 
                tr.last_matched_frame <= frame_id - 2 and 
                not tr.is_dense and
                tr.age > 1):  # 🆕 age > 1 を追加（初期フレームのトラックを除外）

                # ▼▼▼ ここで状態を書き換える（これでOKです！） ▼▼▼
                tr.state = TrackState.FRAME_OUT

                #FO中に誤って生成されたマスクとそのメモリは削除する
                # メモリ削除
                obj_id = self.id_map.get(tr.id)  # 🆕 obj_idを取得する必要がある
                if obj_id is not None:
                    self.remove_occluded_frame_memory(inference_state, obj_id, frame_idx)
                    # print(f"    🗑️ Removed memory for Track {tr.id} (State: {tr.state})")
                #マスク消去 (変数名: track -> tr)
                tr.mask = None

                frame_out_candidates.append(tr)
                frame_out_candidate_indices.append(idx)
                print(f"    🔍 Frame-Out candidate: Track {tr.id} (will skip 2-stage matching)")
                frame_debug_info["frame_out_ids"] = [tr.id for tr in frame_out_candidates]#debag用
            else:
                normal_active_tracks.append(tr)
                normal_track_indices_in_all.append(idx)

        # === 2段階マッチング (フレームアウト候補を除外したトラックのみ) ===
        matches, unmatched_detections, unmatched_track_indices, second_stage_matches = self.two_stage_matching(
            gt_detections, normal_active_tracks
        )

        # **重要**: matchesのインデックスをactive_tracksに戻す
        matches_in_active = []
        for det_idx, normal_idx in matches:
            active_idx = normal_track_indices_in_all[normal_idx]
            matches_in_active.append((det_idx, active_idx))

        # second_stage_matchesもactive_tracksのインデックスに変換
        second_stage_matches_in_active = []
        for det_idx, normal_idx in second_stage_matches:
            active_idx = normal_track_indices_in_all[normal_idx]
            second_stage_matches_in_active.append((det_idx, active_idx))

        unmatched_tracks_in_active = [normal_track_indices_in_all[i] for i in unmatched_track_indices]

        # --------------------------
        #  マッチ成功後 → 密集度計算
        # --------------------------
        matched_track_ids = set()
        
        for det_idx, active_track_idx in matches_in_active:
            detection = gt_detections[det_idx]
            track = active_tracks[active_track_idx]

            # 密集度計算
            density = self.compute_density(detection, gt_detections)
            track.last_matched_density = density
            track.is_dense = density > self.frame_out_d_thre
            track.last_matched_frame = frame_id
            track.last_matched_bbox = detection.bbox.copy()

            print(f"    ✓ Track {track.id}: density={density:.3f}, is_dense={track.is_dense}")

        # Cross-Object Interaction
        coi_processed_track_ids = set()
        
        if len(active_tracks) > 1:
            tracks_to_reconstruct_ids, coi_pairs = self.cross_object_interaction.detect_occlusion_and_resolve(
                active_tracks, frame_id
            )
            for a, b, skipped in coi_pairs:
                frame_debug_info["coi_pairs"].append((a, b, skipped))

            if tracks_to_reconstruct_ids:
                for track in active_tracks:
                    if track.id in tracks_to_reconstruct_ids:
                        if track.skip_memory_current:
                            obj_id = self.id_map.get(track.id)
                            if obj_id is not None:
                                deleted = self.remove_occluded_frame_memory(inference_state, obj_id, frame_idx)
                                if deleted:
                                    coi_processed_track_ids.add(track.id)
                                    print(f"    🚫 Track {track.id}: Deleted occluded frame memory at frame {frame_idx}")
                                    frame_debug_info["coi_ids"].append(track.id)#debag用
                            track.skip_memory_current = False
        
        # Quality Reconstruction判定
        tracks_need_reconstruction = []

        for det_idx, active_track_idx in matches_in_active:
            detection = gt_detections[det_idx]
            track = active_tracks[active_track_idx]

            # 2段階目でマッチしたトラックの特別扱い
            if (det_idx, active_track_idx) in second_stage_matches_in_active:
                # if track.state == TrackState.PENDING:
                tracks_need_reconstruction.append(
                    (track, detection.bbox, "2nd-stage pending reconstruction")
                )
                print(f"    ✓ Track {track.id} scheduled for reconstruction (2nd-stage match)")

                matched_track_ids.add(track.id)
                track.bbox = detection.bbox.copy()
                track.last_seen_frame = frame_id
                track.lost_frames = 0
                continue

            if track.mask is not None:
                x1, y1, x2, y2 = detection.bbox.astype(int)
                h, w = track.mask.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask_cropped = np.zeros_like(track.mask)
                mask_cropped[y1:y2, x1:x2] = track.mask[y1:y2, x1:x2]
                track.mask = mask_cropped

            matched_track_ids.add(track.id)
            track.bbox = detection.bbox.copy()
            track.last_seen_frame = frame_id
            track.lost_frames = 0
            
            # CoIで処理されたトラックはリコンストラクション対象から除外
            if track.id in coi_processed_track_ids:
                print(f"    ⚠️  Track {track.id}: Skipping Quality Reconstruction (already processed by CoI)")
                continue
            
            # 密集度を考慮したQuality Reconstruction判定
            if self.trajectory_manager.should_reconstruct_quality(track, detection):
                density = self.compute_density(detection, gt_detections)

                if density >= self.density_threshold:
                    print(f"    ⚠️ Track {track.id}: Density {density:.3f} ≥ {self.density_threshold} → Skip reconstruction")
                    frame_debug_info["dense_skip"].append((track.id, density))#debag用
                    continue

                tracks_need_reconstruction.append((track, detection.bbox, "Quality reconstruction"))
        
        # マッチしなかったトラックのlost_framesを増加
        for track in self.tracks:
            if track.id not in matched_track_ids:
                track.lost_frames += 1
                if track.lost_frames > self.trajectory_manager.tolerance_frames:
                    track.state = TrackState.LOST

        # -------------------------
        # ===== Frame-Out専用マッチング ====
        # -------------------------
        if len(frame_out_candidates) > 0 and len(unmatched_detections) > 0:
            print(f"  🎯 Frame-Out matching: {len(frame_out_candidates)} candidates vs {len(unmatched_detections)} detections")
            
            from scipy.optimize import linear_sum_assignment
            local_dets = [gt_detections[i] for i in unmatched_detections]
            cost_matrix = np.ones((len(local_dets), len(frame_out_candidates)), dtype=float)

            for i_det, det in enumerate(local_dets):
                for j_tr, tr in enumerate(frame_out_candidates):
                    if tr.last_matched_bbox is None:
                        cost_matrix[i_det, j_tr] = 1.0
                        continue
                    iou = self.compute_bbox_iou(det.bbox, tr.last_matched_bbox)
                    cost_matrix[i_det, j_tr] = 1.0 - iou
                    if iou > 0:
                        print(f"      Det {unmatched_detections[i_det]} vs Track {tr.id}: IoU={iou:.3f}")

            det_idx_local, tr_idx_local = linear_sum_assignment(cost_matrix)

            accepted_pairs = []
            used_det_local = set()
            
            for dloc, tloc in zip(det_idx_local, tr_idx_local):
                if cost_matrix[dloc, tloc] < 1.0:  # IoU > 0
                    orig_det_idx = unmatched_detections[dloc]
                    orig_active_track_idx = frame_out_candidate_indices[tloc]
                    iou = 1.0 - cost_matrix[dloc, tloc]
                    accepted_pairs.append((orig_det_idx, orig_active_track_idx))
                    used_det_local.add(orig_det_idx)
                    print(f"      ✅ Accepted: Det {orig_det_idx} → Track {frame_out_candidates[tloc].id} (IoU={iou:.3f})")

            # 再構築実行
            for det_idx_matched, active_track_idx in accepted_pairs:
                det = gt_detections[det_idx_matched]
                track = active_tracks[active_track_idx]

                obj_id = self.id_map.get(track.id)
                if obj_id is not None:
                    try:
                        self.sam2_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=None,
                            labels=None,
                            box=det.bbox.astype(np.float32)
                        )
                        track.state = TrackState.RELIABLE
                        track.bbox = det.bbox.copy()
                        track.last_seen_frame = frame_id
                        track.lost_frames = 0
                        track.last_matched_frame = frame_id
                        track.last_matched_bbox = det.bbox.copy()
                        track.last_matched_density = self.compute_density(det, gt_detections)
                        track.is_dense = track.last_matched_density > self.frame_out_d_thre
                        matched_track_ids.add(track.id)
                        iterator_needs_reset = True
                        iou_val = self.compute_bbox_iou(det.bbox, track.last_matched_bbox)
                        print(f"    ▶ Frame-Out Reconstructed Track {track.id} with Det {det_idx_matched} (IoU={iou_val:.3f})")
                        frame_debug_info["frame_out_recovered_ids"].append(track.id)#debag用
                    except Exception as e:
                        print(f"    ✗ Frame-Out reconstruction failed for Track {track.id}: {e}")

            # 未マッチ検出を更新
            unmatched_detections = [d for d in unmatched_detections if d not in used_det_local]

        # リコンストラクション実行
        if tracks_need_reconstruction:
            for track, bbox, reason in tracks_need_reconstruction:
                obj_id = self.id_map.get(track.id)
                if obj_id is not None:
                    self.sam2_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=None,
                        labels=None,
                        box=bbox.astype(np.float32)
                    )
                    track.state = TrackState.RELIABLE
                    reset_reasons.append(f"{reason}: Track {track.id}")
                    print(f"    ✓ Track {track.id} reconstructed ({reason})")
                    frame_debug_info["reconstructed_ids"].append(track.id)#debag用
            
            iterator_needs_reset = True
        
        # 新規トラック追加
        if unmatched_detections:
            tracked_masks = [t.mask for t in self.tracks 
                            if t.mask is not None and t.state != TrackState.LOST]
            untracked_mask = self.trajectory_manager.compute_untracked_mask(
                frame.shape[:2], tracked_masks
            )
            
            num_added = 0
            for det_idx in unmatched_detections:
                detection = gt_detections[det_idx]
                
                if self.trajectory_manager.should_add_detection(detection, untracked_mask):
                    new_track_id = self.next_track_id
                    if self.initialize_sam2_tracker(frame, detection.bbox, new_track_id, 
                                                inference_state, frame_idx):
                        
                        # 🆕 新規トラックの密集度を計算
                        density = self.compute_density(detection, gt_detections)
                        is_dense = density > self.frame_out_d_thre

                        new_track = Track(
                            id=new_track_id, bbox=detection.bbox, mask=None,
                            logits_score=10.0, state=TrackState.RELIABLE,
                            lost_frames=0, age=1, logits_history=deque(maxlen=25),
                            sam2_predictor=None, last_seen_frame=frame_id,
                            init_frame=frame_idx,
                            prev_bbox=None,
                            last_matched_frame=frame_id,  # 🆕
                            last_matched_bbox=detection.bbox.copy(),  # 🆕
                            last_matched_density=density,  # 🆕
                            is_dense=is_dense  # 🆕
                        )
                        self.tracks.append(new_track)
                        self.id_map[new_track_id] = new_track_id
                        self.next_track_id += 1
                        num_added += 1
                        iterator_needs_reset = True
            
            if num_added > 0:
                reset_reasons.append(f"New tracks added: {num_added}")
        
        # LOSTトラック削除
        self.tracks = [t for t in self.tracks 
                    if not self.trajectory_manager.should_remove_track(t)]
        
        if iterator_needs_reset:
            print(f"  ⟳ RESET TRIGGERED: {', '.join(reset_reasons)}")

        # 結果生成（マッチしたトラックのみ出力）
        results = []
        for track in self.tracks:
            if track.id in matched_track_ids:
                results.append({
                    'frame_id': frame_id,
                    'track_id': track.id,
                    'bbox': track.bbox.copy(),
                    'mask': track.mask,
                    'confidence': track.logits_score,
                    'state': track.state
                })
        
        return results, iterator_needs_reset, frame_debug_info

    def track_sequence(
            self,
            video_path: str,
            video_output_path: str = "tracking_output.mp4",
            output_dir: str = ".",
            start_frame: int = 1,
            end_frame: int = -1,
        ) -> List[Dict]:
            """Sequential tracking with memory management"""
            img_dir = os.path.join(video_path, 'img1')
            frame_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

            if not frame_files:
                print(f"No frame files found in {img_dir}")
                return []

            start_idx = start_frame - 1
            if end_frame == -1:
                frame_files_subset = frame_files[start_idx:]
            else:
                frame_files_subset = frame_files[start_idx:end_frame]

            print(f"Loading frames {start_frame} to {start_frame + len(frame_files_subset) - 1}")
            frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in frame_files_subset]

            if not frames:
                print("No valid frames loaded")
                return []

            try:
                temp_video_path = self.create_temp_video(frames, output_dir, video_path)
                print(f"Temporary video: {temp_video_path}")
            except Exception as e:
                print(f"Failed to create temp video: {e}")
                return []

            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

            all_results = []
            inference_state = None

            try:
                # 最初のフレームの処理
                inference_state = self.sam2_predictor.init_state(video_path=temp_video_path)
                print("✓ SAM2 initialized")

                first_frame_idx = 0
                first_frame_id = start_frame
                first_frame_rgb = frames[first_frame_idx]
                self.frame_count = first_frame_id

                gt_detections_frame1 = self.gt_loader.get_detections(first_frame_id)
                results_frame1 = []

                if not gt_detections_frame1:
                    print(f"⚠ No GT in frame {start_frame}")
                else:
                    for detection in gt_detections_frame1:
                        track_id = self.next_track_id
                        self.initialize_sam2_tracker(
                            frame=first_frame_rgb, bbox=detection.bbox, track_id=track_id,
                            inference_state=inference_state, frame_idx=first_frame_idx
                        )

                        # 🆕 初期フレームでも密集度を計算
                        density = self.compute_density(detection, gt_detections_frame1)
                        is_dense = density > self.frame_out_d_thre

                        new_track = Track(
                            id=track_id, bbox=detection.bbox, mask=None, logits_score=10.0,
                            state=TrackState.RELIABLE, lost_frames=0, age=1,
                            logits_history=deque(maxlen=25), sam2_predictor=None,
                            last_seen_frame=first_frame_id, init_frame=first_frame_idx,
                            prev_bbox=None,  # 🆕 初期化
                            last_matched_frame=first_frame_id,  # 🆕 初期化
                            last_matched_bbox=detection.bbox.copy(),  # 🆕 初期化
                            last_matched_density=density,  # 🆕 初期化
                            is_dense=is_dense  # 🆕 初期化
                        )
                        self.tracks.append(new_track)
                        self.id_map[track_id] = track_id
                        self.next_track_id += 1

                        print(f"  ✓ Track {track_id} initialized: density={density:.3f}, is_dense={is_dense}")

                    with torch.amp.autocast("cuda"):
                        temp_iterator = self.sam2_predictor.propagate_in_video(
                            inference_state, start_frame_idx=first_frame_idx
                        )
                        try:
                            pred_idx, obj_ids, masks_logits = next(temp_iterator)
                            if pred_idx == first_frame_idx:
                                initial_predictions = self.process_sam2_predictions(
                                    pred_idx, obj_ids, masks_logits
                                )
                                for track in self.tracks:
                                    obj_id = self.id_map.get(track.id)
                                    if obj_id in initial_predictions:
                                        mask, score = initial_predictions[obj_id]
                                        track.mask = mask
                                        track.logits_score = score
                                        track.logits_history.append(score)
                        except StopIteration:
                            print("Warning: Could not get mask for the first frame.")

                for track in self.tracks:
                    if track.last_seen_frame == first_frame_id:
                        results_frame1.append({
                            'frame_id': first_frame_id, 'track_id': track.id,
                            'bbox': track.bbox.copy(), 'mask': track.mask,
                            'confidence': track.logits_score, 'state': track.state
                        })

                all_results.extend(results_frame1)

                # 2フレーム目以降の処理の準備
                self.propagation_iterator = self.sam2_predictor.propagate_in_video(
                    inference_state, start_frame_idx=first_frame_idx + 1
                )

            except Exception as e:
                print(f"SAM2 initialization or first frame processing failed: {e}")
                video_writer.release()
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
                return []

            # メインループ (2フレーム目から開始)
            for frame_idx, frame_rgb in enumerate(frames[1:], start=1):
                frame_id = frame_idx + start_frame
                print(f"\n=== Frame {frame_id}/{start_frame + len(frames) - 1} ===")

                results, iterator_needs_reset, frame_debug_info = self.track_frame_sequential(
                    frame_rgb, frame_id, frame_idx, inference_state
                )

                if iterator_needs_reset:
                    print(f"⟳ Resetting iterator at frame {frame_id}")
                    self.cleanup_old_memory(inference_state, frame_idx)
                    
                    self.propagation_iterator = self.sam2_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=frame_idx + 1
                    )

                all_results.extend(results)

            print(f"Processing completed")

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            return all_results
    
    def create_temp_video(self, frames: List[np.ndarray], output_dir: str, video_path: str) -> str:
            """Create temporary video from frames"""
            data_name = os.path.basename(os.path.normpath(video_path))
            temp_video_path = os.path.join(output_dir, f"temp_video_for_sam2_{data_name}.mp4")
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            for frame_rgb in frames:
                out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            out.release()
            return temp_video_path

def demo_sam2mot_with_gt(args):
    """Demo with sequential processing - text output only"""
    video_path = args.video_path
    gt_path = os.path.join(video_path, "det/det.txt")
    
    gt_loader = GTLoader(gt_path)
    
    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.model_cfg

    try:
        sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(sam2_predictor, "model"):
            sam2_predictor.model.to(device)
        print(f"SAM2 initialized on {device}")
    except Exception as e:
        print(f"SAM2 initialization error: {e}")
        return

    tracker = SAM2MOT(
        sam2_predictor=sam2_predictor,
        gt_loader=gt_loader,
        device=str(device),
        tolerance_frames=args.tolerance_frames,
        memory_window=args.memory_window,
        cost_weight=args.cost_weight,
        tau_r=args.tau_r,
        tau_p=args.tau_p,
        tau_s=args.tau_s,
        density_threshold=args.density_threshold,
        second_stage_iou_threshold=args.second_stage_iou_threshold,  # 🆕
        frame_out_d_thre=args.frame_out_d_thre
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    # video_pathからビデオ名を取得
    video_name = os.path.basename(video_path) 
    # f-stringを使って正しいファイル名を作成
    output_file = os.path.join(args.output_dir, f"{video_name}.txt")

    print(f"\n{'='*60}")
    print(f"Starting tracking...")
    print(f"Video: {video_path}")
    print(f"Frames: {args.start_frame} to {args.end_frame if args.end_frame != -1 else 'end'}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    results = tracker.track_sequence(
        video_path, 
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    # 🔧 MOT形式でテキストファイルに保存
    with open(output_file, 'w') as f:
        for result in results:
            frame_id = result['frame_id']
            track_id = result['track_id']
            x1, y1, x2, y2 = result['bbox']
            w, h = x2 - x1, y2 - y1
            confidence = result['confidence']
            # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{confidence:.4f},-1,-1,-1\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--video_path", type=str, 
                        default="/home-local/zabu/det_val/v_00HRwkvvjtQ_c005",
                        help="Path to video directory (containing img1/ and det/)")
    parser.add_argument("--output_video", type=str, default="tracking_output.mp4", help="Output video filename")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 model config")
    parser.add_argument("--start_frame", type=int, default=1, help="Start frame number")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame number (-1 for all)")
    parser.add_argument("--tolerance_frames", type=int, default=60, help="SAM2 tracks lost objects for this frame number")
    parser.add_argument("--memory_window", type=int, default=25, help="Memory window size (frames to keep)")
    parser.add_argument("--cost_weight", type=float, default=0.5, help="weight for cost")    
    parser.add_argument("--tau_r", type=float, default=8.0, help="Reliable threshold for track state (default: 8.0)")
    parser.add_argument("--tau_p", type=float, default=1.0, help="Pending threshold for track state (default: 1.0)")
    parser.add_argument("--tau_s", type=float, default=0.0, help="Suspicious threshold for track state (default: 0.0)")
    parser.add_argument("--density_threshold", type=float, default=1.5, help="Density threshold above which reconstruction is skipped")
    parser.add_argument("--second_stage_iou_threshold", type=float, default=0.0, 
                        help="IoU threshold for 2nd stage matching with prev_bbox (default: 0.0)")  # 🆕
    parser.add_argument("--frame_out_d_thre", type=float, default=0.6, help="Density threshold for clasiffication of dense trakck")

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")
    
    demo_sam2mot_with_gt(args)