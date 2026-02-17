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

# IDごとの色分けで使用するカラーパレットを定義
COLOR_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128)
]

@dataclass
class TrackState:
    RELIABLE = "reliable"
    PENDING = "pending"
    SUSPICIOUS = "suspicious"
    LOST = "lost"

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
                 tau_p: float = 6,
                 tau_s: float = 2,
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
        return track.state == TrackState.LOST and track.lost_frames > self.tolerance_frames
    
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
                 variance_history: int = 10):  # 🔧 パラメータ化
        """
        Args:
            miou_threshold: mIoU閾値（これを超えたらオクルージョンと判定）
            memory_history_frames: メモリ履歴フレーム数
            variance_history: 分散計算に使用する履歴数
            logits_margin: logitsスコア判定のマージン（デフォルト1.2 = 20%差）
                          例: 1.2 = 20%差、1.5 = 50%差、1.1 = 10%差
        """
        self.miou_threshold = miou_threshold
        self.memory_history_frames = memory_history_frames
        self.variance_history = variance_history
    
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
    
    def detect_occlusion_and_resolve(self, tracks: List[Track], current_frame_idx: int) -> List[int]:
        """
        論文の 3.3 節: オクルージョン検出と解決
        戻り値: 再初期化が必要なトラック ID のリスト
        """
        tracks_to_reconstruct = []
        n = len(tracks)
        
        for i in range(n):
            for j in range(i+1, n):
                track_a = tracks[i]
                track_b = tracks[j]
                
                if track_a.mask is None or track_b.mask is None:
                    continue
                    
                miou = self.compute_mask_iou(track_a.mask, track_b.mask)
                
                if miou <= self.miou_threshold:
                    continue
                
                # --- 判定ロジックの改善 ---
                score_a = track_a.logits_score
                score_b = track_b.logits_score
                diff_score = abs(score_a - score_b)                

                if diff_score >= 2.0:
                    # 確信度の差が2以上の場合は、確信度が低い方をオクルージョンとする
                    occluded_idx = i if score_a < score_b else j
                    reason = f"logits_score: {score_a:.2f} vs {score_b:.2f} (diff={diff_score:.2f} >= 2)"
                else:
                    var_a = self.compute_logits_variance(track_a.logits_history)
                    var_b = self.compute_logits_variance(track_b.logits_history)
                    occluded_idx = i if var_a > var_b else j
                    reason = f"variance: {var_a:.3f} vs {var_b:.3f}"
                    
                occluded_track = tracks[occluded_idx]
                
                # 🔧 オクルージョンフラグを立てる
                occluded_track.skip_memory_current = True
                
                if occluded_track.id not in tracks_to_reconstruct:
                    tracks_to_reconstruct.append(occluded_track.id)
                    print(f"[CoI] Frame {current_frame_idx}: marked track {occluded_track.id} "
                          f"as occluded (mIoU={miou:.3f}, {reason}), will skip memory and reconstruct")
        
        return tracks_to_reconstruct


class SAM2MOT:
    """SAM2MOT: Sequential frame-by-frame processing for dynamic object addition"""
    
    def __init__(self, 
                 sam2_predictor,
                 gt_loader: GTLoader,
                 device: str = "cuda",
                 memory_window: int = 10):  # 🔧 メモリウィンドウサイズを追加
        self.sam2_predictor = sam2_predictor
        self.gt_loader = gt_loader
        self.device = device
        self.memory_window = memory_window  # 🔧 保持するフレーム数
        
        if hasattr(sam2_predictor, 'model'):
            sam2_predictor.model = sam2_predictor.model.float()
        
        self.trajectory_manager = TrajectoryManagerSystem()
        self.cross_object_interaction = CrossObjectInteraction()
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        self.tracking_results = []
        self.id_map = {}
        
        self.propagation_iterator = None

        ### 変更点 2: トラックIDごとの色を保存する辞書を追加 ###
        self.track_colors = {}

    ### 変更点 3: ユニークな色を生成・取得するメソッドを新設 ###
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate or retrieve a unique color for a given track ID."""
        if track_id not in self.track_colors:
            # HSV色空間で色を生成 (H: 0-179, S: 255, V: 255)
            # 黄金比を使うと隣接するIDでも視覚的に離れた色になる
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

    def hungarian_matching(self, gt_detections: List[GTDetection], tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian algorithm for GT detection-track matching"""
        if len(gt_detections) == 0 or len(tracks) == 0:
            return [], list(range(len(gt_detections))), list(range(len(tracks)))
        
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(gt_detections), len(tracks)))
        
        for i, det in enumerate(gt_detections):
            for j, track in enumerate(tracks):
                iou = self.compute_bbox_iou(det.bbox, track.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_detections, unmatched_tracks = [], [], []
        
        matched_det_indices = set()
        matched_track_indices = set()
        
        for i, j in zip(det_indices, track_indices):
            if cost_matrix[i, j] < 1:
                matches.append((i, j))
                matched_det_indices.add(i)
                matched_track_indices.add(j)
        
        unmatched_detections = [i for i in range(len(gt_detections)) if i not in matched_det_indices]
        unmatched_tracks = [j for j in range(len(tracks)) if j not in matched_track_indices]
        
        return matches, unmatched_detections, unmatched_tracks
    
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
    
    #修正
    def remove_occluded_frame_memory(self, inference_state, obj_id: int, frame_idx: int) -> bool:
        """
        オクルージョンフレームのメモリを完全に削除
        メモリスロットを無駄にせず、有効なフレームのメモリを多く保持する
        
        Args:
            inference_state: SAM2のinference state
            obj_id: 対象オブジェクトID（クライアント側ID）
            frame_idx: オクルージョンが発生したフレームインデックス
        
        Returns:
            bool: 削除が成功したかどうか
        """
        # 🔧 修正：SAM2内部のobj_idxに変換
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is None:
            print(f"      ⚠️  Object {obj_id} not found in inference_state")
            return False
        
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        deleted = False
        
        # non_cond_frame_outputsから完全に削除
        if obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None) is not None:
            print(f"      ✓ Deleted non_cond memory for obj {obj_id} at frame {frame_idx}")
            deleted = True
        
        # cond_frame_outputsの処理
        # ユーザー入力があるフレームは削除できないが、メモリ部分のみ削除
        if frame_idx in obj_output_dict["cond_frame_outputs"]:
            out = obj_output_dict["cond_frame_outputs"][frame_idx]
            
            # maskmem_featuresを削除（Noneに設定）
            if out.get("maskmem_features") is not None:
                out["maskmem_features"] = None
                print(f"      ⚠️  Removed maskmem from cond_frame for obj {obj_id} at frame {frame_idx}")
                deleted = True
            
            # maskmem_pos_encも削除
            if out.get("maskmem_pos_enc") is not None:
                out["maskmem_pos_enc"] = None
                deleted = True
        
        # 追跡状態から削除
        if inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None) is not None:
            print(f"      ✓ Removed tracking state for obj {obj_id} at frame {frame_idx}")
            deleted = True
        
        if deleted:
            torch.cuda.empty_cache()
        
        return deleted

    def cleanup_old_memory(self, inference_state, current_frame_idx: int):
        """
        常に最大 memory_window (例: 25) フレーム分のメモリを保持。
        古いフレームを削除して一定数を維持する。
        """
        if not hasattr(inference_state, 'output_dict'):
            return

        # 現在保持しているフレームを取得（昇順）
        frame_keys = sorted(list(inference_state['output_dict'].keys()))

        # 保持数が memory_window を超えていたら古いものを削除
        while len(frame_keys) > self.memory_window:
            oldest = frame_keys.pop(0)
            del inference_state['output_dict'][oldest]

        # non_cond_frame_outputs も同様に処理
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
        
        # 🔧 定期的にメモリクリーンアップを実行
        if frame_idx % 5 == 0:  # 5フレームごと
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
                track.state = self.trajectory_manager.classify_track_state(logits_score)

        # Hungarian matching
        active_tracks = [t for t in self.tracks if t.state != TrackState.LOST]
        matches, unmatched_detections, unmatched_track_indices = self.hungarian_matching(
            gt_detections, active_tracks
        )
        
        tracks_need_reconstruction = []
        
        # マッチしたトラックの更新とQuality Reconstruction判定
        for det_idx, active_track_idx in matches:
            detection = gt_detections[det_idx]
            track = active_tracks[active_track_idx]
            track.lost_frames = 0
            track.last_seen_frame = frame_id
            
            # マッチしたGT検出のBBOXで常に更新
            track.bbox = detection.bbox.copy()
            
            if self.trajectory_manager.should_reconstruct_quality(track, detection):
                tracks_need_reconstruction.append((track, detection.bbox, "Quality reconstruction"))
        
        # Cross-Object Interaction
        if len(active_tracks) > 1:
            tracks_to_reconstruct_ids = self.cross_object_interaction.detect_occlusion_and_resolve(
                active_tracks, frame_id
            )
            
            # 🔧 オクルージョン検出時はメモリスキップのみ（リコンストラクションしない）
            if tracks_to_reconstruct_ids:
                for track in active_tracks:
                    if track.id in tracks_to_reconstruct_ids:
                        # オクルージョンフレームのメモリを削除
                        if track.skip_memory_current:
                            obj_id = self.id_map.get(track.id)
                            if obj_id is not None:
                                self.remove_occluded_frame_memory(inference_state, obj_id, frame_idx)
                            track.skip_memory_current = False  # フラグをリセット
                            print(f"    🚫 Track {track.id}: Skipped occluded frame memory (no reconstruction)")
                        
                        # ❌ リコンストラクションは追加しない
                        # tracks_need_reconstruction.append((track, track.bbox, "CoI reconstruction"))
        
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
                    reset_reasons.append(f"{reason}: Track {track.id}")
                    print(f"    ✓ Track {track.id} reconstructed ({reason})")
            
            iterator_needs_reset = True
        
        # マッチしなかったトラックの処理
        unmatched_tracks = [active_tracks[i] for i in unmatched_track_indices]
        for track in unmatched_tracks:
            track.lost_frames += 1
            if self.trajectory_manager.should_remove_track(track):
                track.state = TrackState.LOST
        
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
                        new_track = Track(
                            id=new_track_id, bbox=detection.bbox, mask=None,
                            logits_score=10.0, state=TrackState.RELIABLE,
                            lost_frames=0, age=1, logits_history=deque(maxlen=20),
                            sam2_predictor=None, last_seen_frame=frame_id,
                            init_frame=frame_idx
                        )
                        self.tracks.append(new_track)
                        self.id_map[new_track_id] = new_track_id
                        self.next_track_id += 1
                        num_added += 1
                        iterator_needs_reset = True
            
            if num_added > 0:
                reset_reasons.append(f"New tracks added: {num_added}")
        
        # LOSTトラック削除
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]
        
        if iterator_needs_reset:
            print(f"  ⟳ RESET TRIGGERED: {', '.join(reset_reasons)}")

        # --- マッチ済みトラックIDを記録 ---
        matched_track_ids = set()
        for _, active_track_idx in matches:
            matched_track_ids.add(active_tracks[active_track_idx].id)

        # --- 結果生成（マッチしたトラックのみ出力）---
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
        
        return results, iterator_needs_reset

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
                        new_track = Track(
                            id=track_id, bbox=detection.bbox, mask=None, logits_score=10.0,
                            state=TrackState.RELIABLE, lost_frames=0, age=1,
                            logits_history=deque(maxlen=10), sam2_predictor=None,
                            last_seen_frame=first_frame_id, init_frame=first_frame_idx
                        )
                        self.tracks.append(new_track)
                        self.id_map[track_id] = track_id
                        self.next_track_id += 1

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

                vis_frame = self.visualize_frame(first_frame_rgb, results_frame1, first_frame_id)
                video_writer.write(vis_frame)
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
                
                # # 🔧 定期的にGPUメモリ使用量を表示
                # if frame_idx % 10 == 0:
                #     if torch.cuda.is_available():
                #         mem_allocated = torch.cuda.memory_allocated() / 1024**3
                #         mem_reserved = torch.cuda.memory_reserved() / 1024**3
                #         print(f"  📊 GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

                results, iterator_needs_reset = self.track_frame_sequential(
                    frame_rgb, frame_id, frame_idx, inference_state
                )

                if iterator_needs_reset:
                    print(f"⟳ Resetting iterator at frame {frame_id}")
                    # 🔧 リセット前にメモリをクリーンアップ
                    self.cleanup_old_memory(inference_state, frame_idx)
                    
                    self.propagation_iterator = self.sam2_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=frame_idx + 1
                    )

                all_results.extend(results)
                vis_frame = self.visualize_frame(frame_rgb, results, frame_id)
                video_writer.write(vis_frame)

            video_writer.release()
            print(f"\n✓ Video saved: {video_output_path}")

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            return all_results
    
    def create_temp_video(self, frames: List[np.ndarray], output_dir: str, video_path: str) -> str:
            """Create temporary video from frames"""
            # video_pathからデータ名を取得
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

    def visualize_frame(self, frame: np.ndarray, results: List[Dict], frame_id: int, 
                    frame_events: Optional[Dict[str, List[int]]] = None):
        """Visualize tracking results with event information"""
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Frame number
        frame_text = f"Frame: {frame_id}"
        cv2.putText(vis_frame, frame_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 既存のトラック可視化
        for result in results:
            track_id = result['track_id']
            bbox = result['bbox']
            color = self._get_track_color(track_id)
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # confidence = result['confidence']
            # text = f"ID:{track_id} S:{confidence:.2f}"
            # (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            # cv2.putText(vis_frame, text, (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #IDのみ表示
            confidence = result['confidence']
            text = f"ID:{track_id}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(vis_frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            mask = result.get('mask')
            if mask is not None:
                overlay = vis_frame.copy()
                overlay[mask > 0] = color
                alpha = 0.6
                vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)
        
        return vis_frame

def demo_sam2mot_with_gt(args):
    """Demo with sequential processing"""
    video_path = args.video_path
    gt_path = os.path.join(video_path, "det/det.txt")
    
    gt_loader = GTLoader(gt_path)
    
    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    try:
        sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if hasattr(sam2_predictor, "model"):
            sam2_predictor.model.to(device)
        print(f"✓ SAM2 initialized on {device}")
    except Exception as e:
        print(f"✗ SAM2 initialization error: {e}")
        return

    tracker = SAM2MOT(
        sam2_predictor=sam2_predictor,
        gt_loader=gt_loader,
        device=str(device),
        memory_window=args.memory_window
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_video_output_path = os.path.join(args.output_dir, args.output_video)
    output_file = os.path.join(args.output_dir, "sam2mot_results.txt")

    results = tracker.track_sequence(
        video_path, 
        video_output_path=full_video_output_path,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    with open(output_file, 'w') as f:
        for result in results:
            frame_id, track_id = result['frame_id'], result['track_id']
            x1, y1, x2, y2 = result['bbox']
            w, h = x2 - x1, y2 - y1
            confidence = result['confidence']
            f.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},{confidence},-1,-1,-1\n")
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Total tracks: {len(set(r['track_id'] for r in results))}")
    print(f"✓ Total detections: {len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--video_path", type=str, 
                        default="/home-local/zabu/sportsmot_publish/dataset/val/v_00HRwkvvjtQ_c005",
                        help="Path to video directory (containing img1/ and gt/)")
    parser.add_argument("--output_video", type=str, default="tracking_output.mp4", help="Output video filename")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--start_frame", type=int, default=1, help="Start frame number")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame number (-1 for all)")
    parser.add_argument("--memory_window", type=int, default=25, help="Memory window size (frames to keep)")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")
    
    demo_sam2mot_with_gt(args)