#!/bin/zsh
# 或 #!/bin/bash 也可以

# ============================================================
# 批量可视化训练结果脚本
# ============================================================
# 功能：
# 1. 遍历 results/<RESULT_SUFFIX>/ 目录下的所有 .pkl 文件
# 2. 调用 visualize_single_result.py 生成训练曲线图
# 3. 保存到 results/visualization/ 目录
#
# 使用方法：
#   ./batch_visualize.sh [RESULT_SUFFIX]
#   例如: ./batch_visualize.sh hparam_search_2
#
# 如果不提供参数，将可视化所有结果目录
# ============================================================

# 激活 conda 环境
# source /G/anaconda3/etc/profile.d/conda.sh
# conda activate aim3_rn

# 解析命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <RESULT_SUFFIX>"
    echo "示例: $0 hparam_search_2"
    echo ""
    echo "可用的结果目录:"
    ls -d results/models/*/ 2>/dev/null | sed 's|results/models/||' | sed 's|/$||'
    exit 1
fi

RESULT_SUFFIX="$1"
RESULTS_DIR="results/models/${RESULT_SUFFIX}"
OUTPUT_DIR="results/visualization/${RESULT_SUFFIX}"

# 检查结果目录是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "错误：结果目录不存在 - $RESULTS_DIR"
    echo ""
    echo "可用的结果目录:"
    ls -d results/models/*/ 2>/dev/null | sed 's|results/models/||' | sed 's|/$||'
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "批量可视化训练结果"
echo "============================================================"
echo "输入目录: $RESULTS_DIR"
echo "输出目录: $OUTPUT_DIR"
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
    
    # 跳过临时文件或备份文件
    if [[ "$filename" =~ ^\..*|.*~$ ]]; then
        echo "跳过: $filename (临时文件)"
        continue
    fi
    
    echo "处理: $filename"
    
    # 调用 Python 脚本生成可视化
    if python visualize_single_result.py "$pkl_file" --output_dir "$OUTPUT_DIR"; then
        ((success_count++))
        echo "  ✓ 成功"
    else
        ((fail_count++))
        echo "  ✗ 失败"
    fi
    echo ""
done

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

