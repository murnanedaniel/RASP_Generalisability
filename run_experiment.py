#!/usr/bin/env python3
"""
Complete experimental pipeline for RASP generalization reproduction.
This script runs the full experiment: data generation -> training -> evaluation
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    # Use the conda environment python
    python_cmd = "/global/homes/d/danieltm/.conda/envs/rl-env/bin/python"
    if cmd.startswith("python "):
        cmd = cmd.replace("python ", f"{python_cmd} ", 1)
    
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd,
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete RASP generalization experiment")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick experiment with smaller dataset")
    parser.add_argument("--data_only", action="store_true",
                       help="Only generate data")
    parser.add_argument("--train_only", action="store_true", 
                       help="Only run training (assumes data exists)")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation (assumes model exists)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model for evaluation (if eval_only)")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    eval_dir = os.path.join(base_dir, "evaluation_results")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    print(f"🚀 Starting RASP Generalization Experiment")
    print(f"📁 Working directory: {base_dir}")
    print(f"📊 Output directory: {run_output_dir}")
    print(f"⏰ Timestamp: {timestamp}")
    
    # Set parameters based on quick mode
    if args.quick:
        print("🏃 Quick mode enabled - using smaller dataset")
        num_train_samples = 5000
        num_val_samples = 500
        max_train_length = 40
        max_epochs = 20
        eval_samples_per_length = 50
        max_eval_length = 100
    else:
        print("🐌 Full mode - using complete dataset (this will take longer)")
        num_train_samples = 50000
        num_val_samples = 5000
        max_train_length = 60
        max_epochs = 100
        eval_samples_per_length = 100
        max_eval_length = 150
    
    success = True
    
    # Step 1: Data Generation
    if not args.train_only and not args.eval_only:
        cmd = (f"python data_generation.py "
               f"--max_train_length {max_train_length} "
               f"--num_samples {num_train_samples} "
               f"--num_val_samples {num_val_samples} "
               f"--output_dir {data_dir}")
        
        success &= run_command(cmd, "Data Generation")
        
        if args.data_only:
            print(f"\n✅ Data generation completed! Data saved to: {data_dir}")
            return
    
    # Step 2: Training
    if not args.eval_only and success:
        # Create output directory for this run
        os.makedirs(run_output_dir, exist_ok=True)
        
        cmd = (f"python train.py "
               f"--data_dir {data_dir} "
               f"--output_dir {run_output_dir}")
        
        success &= run_command(cmd, "Model Training")
    
    # Step 3: Evaluation
    if success or args.eval_only:
        if args.eval_only and args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(run_output_dir, "checkpoints", "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print("   Make sure training completed successfully or provide --model_path")
            return
        
        eval_output_dir = os.path.join(run_output_dir, "evaluation") if not args.eval_only else eval_dir
        
        cmd = (f"python evaluate.py "
               f"--model_path {model_path} "
               f"--max_eval_length {max_eval_length} "
               f"--samples_per_length {eval_samples_per_length} "
               f"--output_dir {eval_output_dir} "
               f"--verbose")
        
        success &= run_command(cmd, "Model Evaluation")
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    if success:
        print("🎉 Experiment completed successfully!")
        
        if not args.eval_only:
            print(f"📊 Results saved to: {run_output_dir}")
            
            # Show key files
            key_files = [
                ("Training logs", "logs/"),
                ("Best model", "checkpoints/best_model.pt"),
                ("Length generalization results", "length_generalization_results.json"),
                ("Evaluation plots", "evaluation/length_generalization_plot.png"),
                ("Detailed evaluation", "evaluation/detailed_results.json")
            ]
            
            print("\n📁 Key output files:")
            for desc, path in key_files:
                full_path = os.path.join(run_output_dir, path)
                exists = "✅" if os.path.exists(full_path) else "❌"
                print(f"  {exists} {desc}: {path}")
        
        print(f"\n📈 Expected results (based on paper):")
        print(f"  • Training accuracy: >95%")
        print(f"  • Validation accuracy: >90%") 
        print(f"  • Length generalization (OOD): >90%")
        print(f"  • This demonstrates the RASP-Generalization Conjecture!")
        
    else:
        print("❌ Experiment failed - check error messages above")
        return 1
    
    print(f"\n🔍 To analyze results:")
    if not args.eval_only:
        print(f"  python evaluate.py --model_path {run_output_dir}/checkpoints/best_model.pt --verbose")
    print(f"  # Check the generated plots and JSON files for detailed analysis")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 