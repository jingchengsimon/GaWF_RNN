"""
诊断训练配置差异的脚本
用于对比不同训练配置并找出性能差异的根本原因
"""
import os
import pickle
import argparse
from pathlib import Path


def analyze_pkl_file(pkl_path):
    """分析pkl文件，提取训练配置和性能指标"""
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n{'='*80}")
        print(f"Analysis: {Path(pkl_path).name}")
        print(f"{'='*80}")
        
        # 检查available keys
        print(f"\nAvailable keys in results:")
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
        
        # 模型配置
        if 'model' in results:
            model = results['model']
            print(f"\n模型配置:")
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
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {pkl_path}: {e}")
        return None


def compare_two_models(pkl1_path, pkl2_path):
    """对比两个模型的配置和性能"""
    print(f"\n{'#'*80}")
    print(f"COMPARING TWO MODELS")
    print(f"{'#'*80}")
    
    print(f"\nModel 1: {pkl1_path}")
    results1 = analyze_pkl_file(pkl1_path)
    
    print(f"\nModel 2: {pkl2_path}")
    results2 = analyze_pkl_file(pkl2_path)
    
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
    
    # 模型结构对比
    if 'model' in results1 and 'model' in results2:
        print(f"\n{'='*80}")
        print(f"MODEL ARCHITECTURE COMPARISON")
        print(f"{'='*80}")
        
        model1 = results1['model']
        model2 = results2['model']
        
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())
        
        print(f"Total parameters: {params1:,} vs {params2:,}")
        
        if params1 != params2:
            print(f"  ⚠️  WARNING: Parameter counts don't match!")
            print(f"  This suggests different model architectures.")


def main():
    parser = argparse.ArgumentParser(description='诊断训练配置差异')
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
            analyze_pkl_file(pkl_file)


if __name__ == '__main__':
    main()

