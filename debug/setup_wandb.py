#!/usr/bin/env python3
"""
Setup script for Weights & Biases (wandb) integration.

Run this script to authenticate with wandb if you haven't already.
If you don't want to use wandb, you can disable it in config.py.

Note: We use wandb==0.18.7 to avoid validation errors in newer versions (0.19.x).
"""

import subprocess
import sys

def setup_wandb():
    """Set up wandb authentication"""
    print("🔗 Setting up Weights & Biases (wandb) for experiment tracking...")
    print()
    print("Note: Using wandb==0.18.7 (newer versions 0.19.x have validation issues)")
    print()
    print("Options:")
    print("1. If you have a wandb account, run: wandb login")
    print("2. If you don't want wandb tracking, set use_wandb=False in config.py")
    print("3. To run in offline mode, run: wandb offline")
    print()
    
    try:
        # Check if wandb is installed
        import wandb
        print(f"✅ wandb is installed (version {wandb.__version__})")
        
        # Check version compatibility
        if wandb.__version__.startswith('0.19'):
            print("⚠️  WARNING: wandb 0.19.x has validation issues")
            print("   Consider downgrading: pip install wandb==0.18.7")
        
        # Check if user is logged in
        try:
            wandb.api.api_key
            print("✅ wandb is configured (API key found)")
            print("You're ready to use wandb tracking!")
            return True
        except:
            print("❌ wandb is not configured (no API key found)")
            print()
            print("To set up wandb:")
            print("1. Create a free account at https://wandb.ai")
            print("2. Run: wandb login")
            print("3. Follow the instructions to paste your API key")
            print()
            print("Or disable wandb by setting use_wandb=False in config.py")
            return False
            
    except ImportError:
        print("❌ wandb is not installed")
        print("Installing wandb...")
        try:
            # Install the compatible version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb==0.18.7"])
            print("✅ wandb==0.18.7 installed successfully")
            print("Now run: wandb login")
            return False
        except subprocess.CalledProcessError:
            print("❌ Failed to install wandb")
            print("You can disable wandb by setting use_wandb=False in config.py")
            return False

if __name__ == "__main__":
    setup_wandb() 