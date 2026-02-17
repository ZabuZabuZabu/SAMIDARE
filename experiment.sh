#!/bin/bash

# --- デフォルト設定 ---
DATASET_ROOT="/home-local/zabu/det_val2"
OUTPUT_DIR="/home/zabu/masa/sam2/experiment/sam2mot_val_result"
SAM2_CHECKPOINT="./checkpoints/sam2.1_hiera_large.pt"
GPU_ID=0
PYTHON_SCRIPT="samidare_for_test.py"
TOLERANCE_FRAME=60
MEMORY_WINDOW=25
COST_WEIGHT=0.5
DENSITY_THRESHOLD=1.5
FRAME_OUT_THRESHOLD=0.6

# --- 使用方法を表示する関数 ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input       データセットのルートディレクトリ (デフォルト: ${DATASET_ROOT})"
    echo "  -s, --script      使用するPythonスクリプト (例: sam2mot_proto15.py)"
    echo "  -o, --output      出力ディレクトリのパス (デフォルト: ${OUTPUT_DIR})"
    echo "  -g, --gpu         使用するGPUのID (デフォルト: ${GPU_ID})"
    echo "  -t, --tolerance_frame ロストトラックを追跡継続する期間 (デフォルト: ${TOLERANCE_FRAME})"
    echo "  -w, --memory_window メモリを保存する期間 (デフォルト: ${MEMORY_WINDOW})"
    echo "  -cw, --cost_weight コスト計算の重み（デフォルト：${COST_WEIGHT}）"
    echo "  -d, --density_threshold 密集度の閾値（デフォルト：${DENSITY_THRESHOLD}）"
    echo "  -fd, --frame_out_d_thre フレームアウトトラック分類用の密集度閾値（デフォルト：${FRAME_OUT_THRESHOLD}）"
    echo "  -h, --help        このヘルプメッセージを表示"
    exit 1
}

# --- コマンドライン引数の解析 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            DATASET_ROOT="$2"
            shift
            ;;
        -s|--script)
            PYTHON_SCRIPT="$2"
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift
            ;;
        -t|--tolerance_frame)
            TOLERANCE_FRAME="$2"
            shift
            ;;
        -w|--memory_window)
            MEMORY_WINDOW="$2"
            shift
            ;;
        -cw|--cost_weight)
            COST_WEIGHT="$2"
            shift
            ;;
        -d|--density_threshold)
            DENSITY_THRESHOLD="$2"
            shift
            ;;
        -fd|--frame_out_d_thre)
            FRAME_OUT_THRESHOLD="$2"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "✗ 不明なオプション: $1"
            usage
            ;;
    esac
    shift
done

# --- スクリプト本体 ---

# 入力ディレクトリの存在確認（エラーハンドリング追加）
if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: Dataset root directory not found: $DATASET_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for video_path in "$DATASET_ROOT"/v_*; do
    if [ -d "$video_path" ]; then
        video_name=$(basename "$video_path")
        output_file="$OUTPUT_DIR/${video_name}.txt"

        echo "============================================================"
        echo "Processing video: $video_name"
        echo "Input directory : $video_path"
        echo "Dataset Root    : $DATASET_ROOT"
        echo "Output dir      : $OUTPUT_DIR"
        echo "Output file     : $output_file"
        echo "GPU ID          : $GPU_ID"
        echo "Python Script   : $PYTHON_SCRIPT"
        echo "Tolerance Frame": $TOLERANCE_FRAME
        echo "Memory Window   : $MEMORY_WINDOW"
        echo "Cost Weight     : $COST_WEIGHT"
        echo "Density Threshold: $DENSITY_THRESHOLD"
        echo "Frame Out Threshold: $FRAME_OUT_THRESHOLD"
        echo "============================================================"

        python3 "$PYTHON_SCRIPT" \
            --gpu "$GPU_ID" \
            --video_path "$video_path" \
            --output_dir "$OUTPUT_DIR" \
            --tolerance_frame "$TOLERANCE_FRAME" \
            --memory_window "$MEMORY_WINDOW"\
            --cost_weight "$COST_WEIGHT"\
            --density_threshold "$DENSITY_THRESHOLD"\
            --frame_out_d_thre "$FRAME_OUT_THRESHOLD"

        if [ $? -ne 0 ]; then
            echo "✗ An error occurred while processing $video_name."
        fi
        echo ""
    fi
done

echo "🎉 All videos have been processed."