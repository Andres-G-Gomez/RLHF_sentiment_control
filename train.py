# train.py
#!/usr/bin/env python3
"""
Simple training script for RLHF sentiment control.
Loads config and runs the main training pipeline.
"""

import os
import argparse

# Import from your existing file
from rlhf_sentiment import (
    RLHFConfig, 
    PreferenceDataProcessor, 
    RewardModel, 
    RLHFTrainer
)

def run_training_pipeline(config: RLHFConfig, num_steps: int = 50):
    """Run the complete RLHF training pipeline"""
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    print("Starting RLHF Sentiment Control Project")
    print("="*50)
    
    # Step 1: Prepare preference data
    print("\n1. Preparing preference data...")
    data_processor = PreferenceDataProcessor(config)
    preference_dataset = data_processor.load_hh_rlhf_data()
    print(f"Loaded {len(preference_dataset)} preference pairs")
    
    # Step 2: Train reward model
    print("\n2. Training reward model...")
    reward_model = RewardModel(config)
    reward_model.train_reward_model(preference_dataset)
    
    # Step 3: RLHF training
    print("\n3. Starting RLHF training...")
    rlhf_trainer = RLHFTrainer(config)
    
    # Extract prompts for training
    train_prompts = [example["prompt"] for example in preference_dataset]
    rlhf_trainer.train_rlhf(train_prompts, num_steps=num_steps)
    
    # Save trained model
    rlhf_trainer.save_model(f"{config.model_dir}/rlhf_model")
    
    print("\n" + "="*50)
    print("RLHF TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Train RLHF sentiment control model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--steps', type=int, default=50, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load config
    config = RLHFConfig()
    print(f"📋 Loaded configuration")
    
    # Override config with command line args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_wandb:
        config.use_wandb = False
    
    print(f"🚀 Starting RLHF training")
    print(f"📊 Settings: {config.batch_size} batch size, {args.steps} steps")
    print(f"🖥️  Device: {config.device}")
    
    # Run the training pipeline
    run_training_pipeline(config, num_steps=args.steps)
    
    print("✅ Training completed! Run 'python evaluate.py' to see results.")

if __name__ == "__main__":
    main()
