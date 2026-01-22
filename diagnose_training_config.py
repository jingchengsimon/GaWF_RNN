"""
诊断训练配置差异的脚本
用于对比不同训练配置并找出性能差异的根本原因

文件格式说明：
- PKL 文件：保存训练曲线（train_acc_char, val_acc_char等）和元数据
- PTH 文件：保存模型权重参数（state_dict），即各层的 weight 和 bias
"""
import os
import pickle
import argparse
import torch
from pathlib import Path


def analyze_pth_file(pth_path):
    """分析pth文件，从state_dict推断模型配置"""
    try:
        state_dict = torch.load(pth_path, map_location='cpu')
        
        config = {}
        
        # 提取 hidden_size（从 RNN 的 weight_hh_l0 推断）
        if 'rnn.weight_hh_l0' in state_dict:
            hidden_size = state_dict['rnn.weight_hh_l0'].shape[0]
            config['hidden_size'] = hidden_size
        
        # 提取 num_classes（从 fcchar 推断）
        if 'fcchar.weight' in state_dict:
            num_classes = state_dict['fcchar.weight'].shape[0]
            config['num_classes'] = num_classes
        
        # 提取 num_pos（从 fcpos 推断，如果存在）
        if 'fcpos.weight' in state_dict:
            num_pos = state_dict['fcpos.weight'].shape[0]
            config['num_pos'] = num_pos
            config['mode'] = 'sector' if num_pos == 9 else 'coord' if num_pos == 2 else 'unknown'
        elif 'fcchars.weight' in state_dict:
            config['mode'] = 'allchars'
            max_chars = state_dict['fcchars.weight'].shape[0] // config.get('num_classes', 10)
            config['max_chars'] = max_chars
        
        # 提取卷积层配置
        if 'conv1.weight' in state_dict:
            conv1_out_channels = state_dict['conv1.weight'].shape[0]
            conv1_in_channels = state_dict['conv1.weight'].shape[1]
            conv1_kernel_size = state_dict['conv1.weight'].shape[2]
            config['conv1'] = f"in={conv1_in_channels}, out={conv1_out_channels}, kernel={conv1_kernel_size}"
        
        if 'conv2.weight' in state_dict:
            conv2_out_channels = state_dict['conv2.weight'].shape[0]
            conv2_in_channels = state_dict['conv2.weight'].shape[1]
            config['conv2'] = f"in={conv2_in_channels}, out={conv2_out_channels}"
        
        # 计算总参数数量
        total_params = sum(p.numel() for p in state_dict.values())
        config['total_params'] = total_params
        
        # RNN 类型判断
        if 'rnn.weight_ih_l0' in state_dict:
            rnn_type = 'rnn'  # 无法直接区分 RNN/LSTM/GRU，需要根据参数数量推断
            weight_ih_shape = state_dict['rnn.weight_ih_l0'].shape
            weight_hh_shape = state_dict['rnn.weight_hh_l0'].shape
            
            # LSTM 和 GRU 的权重矩阵会更大
            if weight_hh_shape[0] == config.get('hidden_size', 0) * 4:
                rnn_type = 'lstm'
            elif weight_hh_shape[0] == config.get('hidden_size', 0) * 3:
                rnn_type = 'gru'
            
            config['rnn_type'] = rnn_type
            config['rnn_weight_ih_shape'] = weight_ih_shape
            config['rnn_weight_hh_shape'] = weight_hh_shape
        
        return config
        
    except Exception as e:
        print(f"Error analyzing PTH file {pth_path}: {e}")
        return None


def analyze_pkl_file(pkl_path):
    """分析pkl文件，提取训练配置和性能指标"""
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n{'='*80}")
        print(f"Analysis: {Path(pkl_path).name}")
        print(f"{'='*80}")
        
        # 分析对应的 PTH 文件（如果存在）
        pth_path = str(pkl_path).replace('.pkl', '_model.pth')
        pth_config = None
        if Path(pth_path).exists():
            print(f"\n从 PTH 文件提取模型配置:")
            pth_config = analyze_pth_file(pth_path)
            if pth_config:
                print(f"  RNN type: {pth_config.get('rnn_type', 'unknown')}")
                print(f"  Hidden size: {pth_config.get('hidden_size', 'unknown')}")
                print(f"  Num classes: {pth_config.get('num_classes', 'unknown')}")
                if 'num_pos' in pth_config:
                    print(f"  Num positions: {pth_config['num_pos']} ({pth_config.get('mode', 'unknown')} mode)")
                print(f"  Conv1: {pth_config.get('conv1', 'unknown')}")
                print(f"  Conv2: {pth_config.get('conv2', 'unknown')}")
                print(f"  Total parameters: {pth_config.get('total_params', 0):,}")
                print(f"  RNN weight_ih shape: {pth_config.get('rnn_weight_ih_shape', 'unknown')}")
                print(f"  RNN weight_hh shape: {pth_config.get('rnn_weight_hh_shape', 'unknown')}")
        else:
            print(f"\n⚠️  对应的 PTH 文件不存在: {pth_path}")
        
        # 检查available keys
        print(f"\nPKL 文件中的训练曲线:")
        for key in results.keys():
            if key != 'model':
                print(f"  - {key}: {type(results[key])}")
        
        # 性能指标
        print(f"\n性能指标:")
        if 'train_acc_char' in results:
            train_acc = results['train_acc_char']
            print(f"  Train char acc: max={max(train_acc):.2f}%, final={train_acc[-1]:.2f}%")
        
        if 'val_acc_char' in results:
            val_acc = results['val_acc_char']
            print(f"  Val char acc: max={max(val_acc):.2f}%, final={val_acc[-1]:.2f}%")
        
        if 'train_acc_pos' in results:
            train_pos = results['train_acc_pos']
            print(f"  Train pos acc: max={max(train_pos):.2f}%, final={train_pos[-1]:.2f}%")
        
        if 'val_acc_pos' in results:
            val_pos = results['val_acc_pos']
            print(f"  Val pos acc: max={max(val_pos):.2f}%, final={val_pos[-1]:.2f}%")
        
        if 'train_err_pos' in results:
            train_err = results['train_err_pos']
            print(f"  Train pos MSE: min={min(train_err):.2f}, final={train_err[-1]:.2f}")
        
        if 'val_err_pos' in results:
            val_err = results['val_err_pos']
            print(f"  Val pos MSE: min={min(val_err):.2f}, final={val_err[-1]:.2f}")
        
        # Epoch信息
        if 'actual_epochs' in results:
            print(f"  Actual epochs: {results['actual_epochs']}")
        else:
            epochs = len(results.get('train_acc_char', []))
            print(f"  Total epochs: {epochs}")
        
        # 模型配置（如果 PKL 中包含 model 对象）
        if 'model' in results:
            model = results['model']
            print(f"\nPKL 文件中的模型对象:")
            print(f"  Model type: {type(model).__name__}")
            
            if hasattr(model, 'rnn'):
                print(f"  RNN hidden size: {model.rnn.hidden_size}")
                print(f"  RNN num layers: {model.rnn.num_layers}")
            
            if hasattr(model, 'dropout_rate'):
                print(f"  Dropout rate: {model.dropout_rate}")
            
            if hasattr(model, 'fcchar'):
                print(f"  Num classes: {model.fcchar.out_features}")
            
            if hasattr(model, 'fcpos') and model.fcpos is not None:
                print(f"  Num positions: {model.fcpos.out_features}")
            elif hasattr(model, 'predict_all_chars') and model.predict_all_chars:
                print(f"  Mode: predict_all_chars")
            
            # 检查模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
        
        print(f"\n{'='*80}\n")
        
        return results, pth_config
        
    except Exception as e:
        print(f"Error analyzing {pkl_path}: {e}")
        return None


def compare_two_models(pkl1_path, pkl2_path):
    """对比两个模型的配置和性能"""
    print(f"\n{'#'*80}")
    print(f"COMPARING TWO MODELS")
    print(f"{'#'*80}")
    
    print(f"\nModel 1: {pkl1_path}")
    results1, pth_config1 = analyze_pkl_file(pkl1_path)
    
    print(f"\nModel 2: {pkl2_path}")
    results2, pth_config2 = analyze_pkl_file(pkl2_path)
    
    if results1 is None or results2 is None:
        print("Error: Cannot compare models")
        return
    
    # 性能对比
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    if 'train_acc_char' in results1 and 'train_acc_char' in results2:
        train1 = max(results1['train_acc_char'])
        train2 = max(results2['train_acc_char'])
        diff = train2 - train1
        print(f"Train char acc: {train1:.2f}% vs {train2:.2f}% (diff: {diff:+.2f}%)")
    
    if 'val_acc_char' in results1 and 'val_acc_char' in results2:
        val1 = max(results1['val_acc_char'])
        val2 = max(results2['val_acc_char'])
        diff = val2 - val1
        print(f"Val char acc: {val1:.2f}% vs {val2:.2f}% (diff: {diff:+.2f}%)")
    
    if 'train_acc_pos' in results1 and 'train_acc_pos' in results2:
        pos1 = max(results1['train_acc_pos'])
        pos2 = max(results2['train_acc_pos'])
        diff = pos2 - pos1
        print(f"Train pos acc: {pos1:.2f}% vs {pos2:.2f}% (diff: {diff:+.2f}%)")
    
    if 'val_acc_pos' in results1 and 'val_acc_pos' in results2:
        pos1 = max(results1['val_acc_pos'])
        pos2 = max(results2['val_acc_pos'])
        diff = pos2 - pos1
        print(f"Val pos acc: {pos1:.2f}% vs {pos2:.2f}% (diff: {diff:+.2f}%)")
    
    # 模型结构对比（从 PTH 文件）
    if pth_config1 and pth_config2:
        print(f"\n{'='*80}")
        print(f"MODEL ARCHITECTURE COMPARISON (from PTH files)")
        print(f"{'='*80}")
        
        print(f"\n超参数对比:")
        print(f"{'  Parameter':<25} {'Model 1':<20} {'Model 2':<20} {'Match':<10}")
        print(f"  {'-'*75}")
        
        # 对比各项配置
        keys_to_compare = ['rnn_type', 'hidden_size', 'num_classes', 'num_pos', 'mode', 
                          'conv1', 'conv2', 'total_params']
        
        for key in keys_to_compare:
            val1 = pth_config1.get(key, 'N/A')
            val2 = pth_config2.get(key, 'N/A')
            match = '✓' if val1 == val2 else '✗ DIFF'
            
            # 格式化输出
            if isinstance(val1, int):
                val1_str = f"{val1:,}" if val1 != 'N/A' else 'N/A'
            else:
                val1_str = str(val1)
            
            if isinstance(val2, int):
                val2_str = f"{val2:,}" if val2 != 'N/A' else 'N/A'
            else:
                val2_str = str(val2)
            
            print(f"  {key:<25} {val1_str:<20} {val2_str:<20} {match:<10}")
        
        # RNN 权重矩阵形状对比
        print(f"\nRNN 权重矩阵形状对比:")
        print(f"  weight_ih_l0: {pth_config1.get('rnn_weight_ih_shape', 'N/A')} vs {pth_config2.get('rnn_weight_ih_shape', 'N/A')}")
        print(f"  weight_hh_l0: {pth_config1.get('rnn_weight_hh_shape', 'N/A')} vs {pth_config2.get('rnn_weight_hh_shape', 'N/A')}")
        
        # 关键发现
        if pth_config1.get('total_params') != pth_config2.get('total_params'):
            print(f"\n⚠️  WARNING: 模型参数总数不匹配！")
            print(f"  这表明两个模型的架构不同。")
        else:
            print(f"\n✓ 模型参数总数匹配，架构相同。")
    
    # 模型结构对比（从 PKL 文件中的 model 对象，如果有）
    elif 'model' in results1 and 'model' in results2:
        print(f"\n{'='*80}")
        print(f"MODEL ARCHITECTURE COMPARISON (from PKL model objects)")
        print(f"{'='*80}")
        
        model1 = results1['model']
        model2 = results2['model']
        
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())
        
        print(f"Total parameters: {params1:,} vs {params2:,}")
        
        if params1 != params2:
            print(f"  ⚠️  WARNING: Parameter counts don't match!")
            print(f"  This suggests different model architectures.")
    else:
        print(f"\n⚠️  无法对比模型架构（PTH 或 PKL 中的 model 对象不可用）")


def main():
    parser = argparse.ArgumentParser(
        description='诊断训练配置差异',
        epilog="""
文件格式说明：
  PKL 文件：保存训练曲线（train_acc, val_acc等）和元数据
  PTH 文件：保存模型权重参数（state_dict），用于推断模型配置
        """
    )
    parser.add_argument('pkl_files', nargs='+', help='要分析的pkl文件路径')
    parser.add_argument('--compare', action='store_true', help='对比两个模型（需要提供恰好2个pkl文件）')
    
    args = parser.parse_args()
    
    if args.compare:
        if len(args.pkl_files) != 2:
            print("Error: --compare requires exactly 2 pkl files")
            return
        compare_two_models(args.pkl_files[0], args.pkl_files[1])
    else:
        for pkl_file in args.pkl_files:
            results, pth_config = analyze_pkl_file(pkl_file)


if __name__ == '__main__':
    main()

