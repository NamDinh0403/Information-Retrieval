"""
Ablation Study Runner
=====================
Chạy các ablation experiments một cách tự động.

Experiments:
    1. Hash bits: 16, 32, 64, 128
    2. Backbone: vit, dinov3
    3. Feature layer: cls, gap (future)

Usage:
    python run_ablation.py --experiment hash_bits
    python run_ablation.py --experiment backbone
    python run_ablation.py --all
    python run_ablation.py --quick  # Quick test với 3 epochs
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from typing import Dict, List

import torch


def run_experiment(cmd: List[str], name: str) -> Dict:
    """Run một experiment và trả về results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        success = False
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return {
        'name': name,
        'command': ' '.join(cmd),
        'success': success,
        'duration_seconds': duration,
        'timestamp': end_time.isoformat()
    }


def ablation_hash_bits(args) -> List[Dict]:
    """Ablation study về hash bits."""
    print("\n" + "#"*60)
    print("ABLATION STUDY: HASH BITS")
    print("#"*60)
    
    hash_bits = [16, 32, 64, 128]
    results = []
    
    for bits in hash_bits:
        cmd = [
            'python', 'experiments/train.py',
            '--model', args.model,
            '--hash-bit', str(bits),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--save-dir', f'./checkpoints/ablation_hashbits_{bits}'
        ]
        
        result = run_experiment(cmd, f"Hash bits = {bits}")
        result['hash_bit'] = bits
        results.append(result)
    
    return results


def ablation_backbone(args) -> List[Dict]:
    """Ablation study về backbone."""
    print("\n" + "#"*60)
    print("ABLATION STUDY: BACKBONE")
    print("#"*60)
    
    backbones = ['vit', 'dinov3']
    results = []
    
    for backbone in backbones:
        cmd = [
            'python', 'experiments/train.py',
            '--model', backbone,
            '--hash-bit', str(args.hash_bit),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--save-dir', f'./checkpoints/ablation_backbone_{backbone}'
        ]
        
        result = run_experiment(cmd, f"Backbone = {backbone}")
        result['backbone'] = backbone
        results.append(result)
    
    return results


def collect_results(results_dir: str = './checkpoints') -> Dict:
    """Thu thập kết quả từ các training history files."""
    collected = {}
    
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Look for training history
        for file in os.listdir(folder_path):
            if file.startswith('training_history') and file.endswith('.json'):
                history_path = os.path.join(folder_path, file)
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # Get best mAP
                if history:
                    best_map = max([h.get('val_mAP', 0) for h in history])
                    collected[folder] = {
                        'best_val_mAP': best_map,
                        'epochs': len(history),
                        'history_file': history_path
                    }
    
    return collected


def generate_report(all_results: Dict, output_path: str):
    """Generate summary report."""
    report = []
    report.append("="*60)
    report.append("ABLATION STUDY REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("="*60)
    
    # Hash bits results
    report.append("\n## Hash Bits Ablation\n")
    report.append(f"{'Bits':<10} {'mAP':<10} {'Status'}")
    report.append("-"*40)
    
    for key, data in all_results.items():
        if 'hashbits' in key:
            bits = key.split('_')[-1]
            mAP = data.get('best_val_mAP', 'N/A')
            if isinstance(mAP, float):
                mAP = f"{mAP:.4f}"
            report.append(f"{bits:<10} {mAP:<10} {'✓' if mAP != 'N/A' else '✗'}")
    
    # Backbone results
    report.append("\n## Backbone Ablation\n")
    report.append(f"{'Backbone':<15} {'mAP':<10} {'Status'}")
    report.append("-"*40)
    
    for key, data in all_results.items():
        if 'backbone' in key:
            backbone = key.split('_')[-1]
            mAP = data.get('best_val_mAP', 'N/A')
            if isinstance(mAP, float):
                mAP = f"{mAP:.4f}"
            report.append(f"{backbone:<15} {mAP:<10} {'✓' if mAP != 'N/A' else '✗'}")
    
    report_text = '\n'.join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n[✓] Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner')
    
    parser.add_argument('--experiment', type=str, 
                       choices=['hash_bits', 'backbone', 'all'],
                       help='Which ablation to run')
    parser.add_argument('--all', action='store_true', 
                       help='Run all ablation studies')
    parser.add_argument('--collect-only', action='store_true',
                       help='Only collect results from existing runs')
    
    # Training params
    parser.add_argument('--model', type=str, default='vit',
                       help='Default model for ablations')
    parser.add_argument('--hash-bit', type=int, default=64,
                       help='Default hash bits for ablations')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (3 epochs)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results/ablation')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 3
        print("[Quick mode] epochs=3")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    if args.collect_only:
        print("Collecting results from existing runs...")
        all_results = collect_results()
        generate_report(all_results, os.path.join(args.output_dir, 'ablation_report.txt'))
        return
    
    # Run experiments
    if args.all or args.experiment == 'hash_bits':
        results = ablation_hash_bits(args)
        for r in results:
            all_results[f"hashbits_{r['hash_bit']}"] = r
    
    if args.all or args.experiment == 'backbone':
        results = ablation_backbone(args)
        for r in results:
            all_results[f"backbone_{r['backbone']}"] = r
    
    # Save raw results
    with open(os.path.join(args.output_dir, 'ablation_raw.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Collect and report metrics
    print("\n" + "#"*60)
    print("COLLECTING RESULTS")
    print("#"*60)
    
    collected = collect_results()
    all_results.update(collected)
    
    generate_report(all_results, os.path.join(args.output_dir, 'ablation_report.txt'))
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
