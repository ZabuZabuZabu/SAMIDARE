import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
from loguru import logger

# YOLOXコンポーネントのインポート
sys.path.append('.')
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer


def make_parser():
    parser = argparse.ArgumentParser("YOLOX MOT-format Detector (Single Video)")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # 入力動画ファイルのパス
    parser.add_argument(
        "--video",
        required=True,
        help="path to input video file"
    )
    # 出力先ディレクトリ
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="directory to save the MOT formatted results"
    )

    parser.add_argument("-f", "--exp_file", default="/home/zabu/masa/yolox/yolox_x_ch_sportsmot.py", type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true")
    # フレーム画像の保存をスキップするオプション（大容量動画向け）
    parser.add_argument("--skip_save_frames", default=False, action="store_true",
                        help="skip saving individual frame images to save disk space")

    return parser


class Predictor(object):
    def __init__(self, model, exp, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        height, width = img.shape[:2]
        img_info = {"height": height, "width": width}

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def process_video(predictor, args, exp):
    video_path = args.video
    if not osp.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    # シーケンス名を動画ファイル名（拡張子なし）から取得
    seq_name = osp.splitext(osp.basename(video_path))[0]
    logger.info(f"Processing video: {seq_name}")

    # 出力ディレクトリの作成
    seq_out_dir = osp.join(args.output_dir, seq_name)
    det_out_dir = osp.join(seq_out_dir, "det")
    img_out_dir = osp.join(seq_out_dir, "img1")
    os.makedirs(det_out_dir, exist_ok=True)
    if not args.skip_save_frames:
        os.makedirs(img_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}")

    det_results = []
    timer = Timer()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1  # MOT形式は1-based index

        # フレーム画像の保存
        if not args.skip_save_frames:
            img_name = f"{frame_id:06d}.jpg"
            cv2.imwrite(osp.join(img_out_dir, img_name), frame)

        # 推論
        outputs, img_info = predictor.inference(frame, timer)

        if outputs[0] is not None:
            det = outputs[0].cpu().detach().numpy()
            scale = min(exp.test_size[0] / img_info["height"], exp.test_size[1] / img_info["width"])

            for d in det:
                coords = d[:4] / scale
                score = d[4]
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1

                # MOT形式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
                line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n"
                det_results.append(line)

        if frame_id % 30 == 0:
            logger.info(f"Processed {frame_id} / {total_frames} frames")

    cap.release()

    # det.txt の書き出し
    det_file = osp.join(det_out_dir, "det.txt")
    with open(det_file, "w") as f:
        f.writelines(det_results)

    logger.info(f"Total processed frames: {frame_id}")
    logger.info(f"Detection results saved to: {det_file}")
    if not args.skip_save_frames:
        logger.info(f"Frame images saved to: {img_out_dir}")


def main(exp, args):
    if args.device.startswith("cuda"):
        args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")

    model = exp.get_model().to(args.device)
    model.eval()

    if not args.ckpt:
        ckpt_file = "/home/zabu/masa/yolox/checkpoints/yolox_x_sports_mix.pth.tar"
    else:
        ckpt_file = args.ckpt

    logger.info("Loading checkpoint...")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, args.device, args.fp16)

    process_video(predictor, args, exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    if args.conf is not None: exp.test_conf = args.conf
    if args.nms is not None: exp.nmsthre = args.nms
    if args.tsize is not None: exp.test_size = (args.tsize, args.tsize)

    main(exp, args)