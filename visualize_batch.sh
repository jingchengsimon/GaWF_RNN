#!/bin/zsh
# 或 #!/bin/bash 也可以

# ============================================================
# 批量可视化训练结果脚本
# ============================================================
# 功能：
# 1. 遍历 results/train_data/<RESULT_SUFFIX>/ 目录下的所有 .pkl 文件
# 2. 调用 utils_viz/model_train_single_result.py 生成训练曲线图
# 3. 保存到 results/train_figs/model_train_single_result/<RESULT_SUFFIX>/ 目录
#
# 使用方法：
#   ./visualize_batch.sh [RESULT_SUFFIX]
#   例如: ./visualize_batch.sh hparam_search_2
#
# 如果不提供参数，将可视化所有结果目录
# ============================================================

# 激活 conda 环境
# source /G/anaconda3/etc/profile.d/conda.sh
# conda activate aim3_rn

# 确保在项目根目录运行（无论从何处调用脚本）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT" || exit 1

# 解析命令行参数
# 用法:
#   $0 <RESULT_SUFFIX> [epoch_start] [epoch_end] [--compare] [--models m1 m2 ...]
# 例如:
#   $0 sector_40h_adamw_0409
#   $0 sector_40h_adamw_0409 0 100
#   $0 sector_40h_adamw_0409 --compare --models gawf rnn
if [ $# -eq 0 ]; then
    echo "用法: $0 <RESULT_SUFFIX> [epoch_start] [epoch_end] [--compare] [--models m1 m2 ...]"
    echo "示例: $0 hparam_search_2"
    echo "      $0 sector_40h_adamw_0409 0 100"
    echo "      $0 sector_40h_adamw_0409 --compare --models gawf rnn"
    echo ""
    echo "可用的结果目录:"
    ls -d results/train_data/*/ 2>/dev/null | sed 's|results/train_data/||' | sed 's|/$||'
    exit 1
fi

RESULT_SUFFIX="$1"
shift
EPOCH_START=""
EPOCH_END=""
DO_COMPARE=0
COMPARE_MODELS=("gawf" "rnn")

while [ $# -gt 0 ]; do
    case "$1" in
        --compare)
            DO_COMPARE=1
            shift
            ;;
        --models)
            shift
            COMPARE_MODELS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                COMPARE_MODELS+=("$1")
                shift
            done
            if [ ${#COMPARE_MODELS[@]} -eq 0 ]; then
                echo "错误：--models 后至少需要一个模型名"
                exit 1
            fi
            ;;
        *)
            # 兼容旧参数风格：前两个非选项参数作为 epoch_start / epoch_end
            if [ -z "$EPOCH_START" ]; then
                EPOCH_START="$1"
            elif [ -z "$EPOCH_END" ]; then
                EPOCH_END="$1"
            else
                echo "错误：无法识别的参数 $1"
                exit 1
            fi
            shift
            ;;
    esac
done

RESULTS_DIR="results/train_data/${RESULT_SUFFIX}"
OUTPUT_DIR="results/train_figs/${RESULT_SUFFIX}"

# 检查结果目录是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "错误：结果目录不存在 - $RESULTS_DIR"
    echo ""
    echo "可用的结果目录:"
    ls -d results/train_data/*/ 2>/dev/null | sed 's|results/train_data/||' | sed 's|/$||'
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "批量可视化训练结果"
echo "============================================================"
echo "输入目录: $RESULTS_DIR"
echo "输出目录: $OUTPUT_DIR"
if [ -n "$EPOCH_START" ] && [ -n "$EPOCH_END" ]; then
    echo "绘制范围: epoch $EPOCH_START ~ $(( EPOCH_END - 1 )) (不含 $EPOCH_END)"
else
    echo "绘制范围: 全部 epoch"
fi
if [ "$DO_COMPARE" -eq 1 ]; then
    echo "Compare: 开启"
    echo "Compare models: ${COMPARE_MODELS[*]}"
else
    echo "Compare: 关闭"
fi
echo "============================================================"
echo ""

# 查找所有 pkl 文件（排除 _model.pth 文件）
PKL_FILES=($(find "$RESULTS_DIR" -maxdepth 1 -name "*.pkl" -type f))

if [ ${#PKL_FILES[@]} -eq 0 ]; then
    echo "错误：未找到任何 .pkl 文件在目录 $RESULTS_DIR"
    exit 1
fi

echo "找到 ${#PKL_FILES[@]} 个结果文件"
echo ""

# 计数器
success_count=0
fail_count=0

# 遍历所有 pkl 文件
for pkl_file in "${PKL_FILES[@]}"; do
    # 提取文件名（不含路径）
    filename=$(basename "$pkl_file")
    
    # 跳过临时文件或备份文件（以.开头或以~结尾）
    if [[ "$filename" == .* ]] || [[ "$filename" == *~ ]]; then
        echo "跳过: $filename (临时文件)"
        continue
    fi
    
    echo "处理: $filename"
    
    # 调用 Python 脚本生成可视化（若有 epoch 范围则传入）
    if [ -n "$EPOCH_START" ] && [ -n "$EPOCH_END" ]; then
        python utils_viz/model_train_single_result.py "$pkl_file" --output_dir "$OUTPUT_DIR" \
            --epoch_start "$EPOCH_START" --epoch_end "$EPOCH_END"
    else
        python utils_viz/model_train_single_result.py "$pkl_file" --output_dir "$OUTPUT_DIR"
    fi
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "  ✓ 成功"
    else
        ((fail_count++))
        echo "  ✗ 失败"
    fi
    echo ""
done

# 可选：额外生成 compare 图（不会替代单模型图）
if [ "$DO_COMPARE" -eq 1 ]; then
    echo "============================================================"
    echo "生成 compare 图"
    echo "============================================================"
    compare_cmd=(python utils_viz/model_train_compare_result.py --result_suffix "$RESULT_SUFFIX" --models "${COMPARE_MODELS[@]}")
    if [ -n "$EPOCH_START" ]; then
        compare_cmd+=(--epoch_start "$EPOCH_START")
    fi
    if [ -n "$EPOCH_END" ]; then
        compare_cmd+=(--epoch_end "$EPOCH_END")
    fi
    "${compare_cmd[@]}"
    if [ $? -eq 0 ]; then
        echo "Compare 图生成成功"
    else
        echo "Compare 图生成失败"
    fi
    echo ""
fi

echo "============================================================"
echo "批量可视化完成"
echo "============================================================"
echo "成功: $success_count 个文件"
echo "失败: $fail_count 个文件"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 显示生成的图片列表
echo "生成的图片:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "未找到生成的图片"
echo ""

# 统计信息
total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "总大小: $total_size"
echo ""

