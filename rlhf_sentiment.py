
# Core RLHF implementation - keep this as your main file with all classes

import os
import torch
import numpy as np
from typing import List
from transformers import GPT2LMHeadModel, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb
from tqdm import tqdm
from config import RLHFConfig
from reward import RewardModel
from data.data_processor import PreferenceDataProcessor
from evaluate import RLHFEvaluator

class RLHFTrainer:
    """Main RLHF trainer using PPO"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        self.reward_model = RewardModel(config)
        
        # PPO configuration
        self.ppo_config = PPOConfig(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            args=self.ppo_config,  # PPO configuration
            processing_class=self.tokenizer,  # Use tokenizer as processing class
            model=self.model,  # Your main model
            ref_model=None,  # Reference model (can be None to auto-create)
            reward_model=None,  # Reward model (we'll handle separately)
            train_dataset=None,  # Dataset (we'll provide during training)
            value_model=None,  # Value model (None means use model's value head)
        )

    def train_rlhf(self, prompts: List[str], num_steps: int = 100):
        """Train using RLHF with PPO"""
        print("Starting RLHF training...")
        
        # Load reward model
        self.reward_model.load_reward_model()
        
        generation_kwargs = {
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        for step in tqdm(range(num_steps), desc="RLHF Training"):
            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, size=self.config.batch_size).tolist()
            
            # Tokenize prompts
            prompt_tensors = []
            for prompt in batch_prompts:
                prompt_tensor = self.tokenizer.encode(prompt, return_tensors="pt")[0]
                prompt_tensors.append(prompt_tensor)
            
            # Generate responses
            response_tensors = []
            for prompt_tensor in prompt_tensors:
                response = self.ppo_trainer.generate(
                    prompt_tensor.unsqueeze(0),
                    **generation_kwargs
                )
                response_tensors.append(response[0])
            
            # Decode responses
            responses = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            # Get rewards
            rewards = self.reward_model.get_reward(responses)
            reward_tensors = [torch.tensor(r) for r in rewards]
            
            # PPO step
            stats = self.ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
            
            # Log statistics
            if step % 10 == 0:
                print(f"Step {step}: Mean reward = {np.mean(rewards):.3f}")
                if self.config.use_wandb:
                    wandb.log({
                        "step": step,
                        "mean_reward": np.mean(rewards),
                        "reward_std": np.std(rewards),
                        **stats
                    })
        
        print("RLHF training complete!")
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)



def run_complete_pipeline(config: RLHFConfig = None, num_steps: int = 50):
    """Run the complete RLHF pipeline"""
    if config is None:
        config = RLHFConfig()
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(project="rlhf-sentiment-control", config=config.__dict__)
    
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
    
    if config.use_wandb:
        wandb.finish()

def run_evaluation(config: RLHFConfig = None, test_prompts: List[str] = None):
    """Run evaluation comparing base vs RLHF model"""
    if config is None:
        config = RLHFConfig()
    
    if test_prompts is None:
        test_prompts = [
            "Tell me about your day",
            "What do you think about technology?",
            "How would you describe the weather?",
            "Please write about your favorite hobby",
            "What are your thoughts on learning?"
        ]
    
    print("Starting Model Evaluation")
    print("="*30)
    
    # Load models for comparison
    base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
    rlhf_model_path = f"{config.model_dir}/rlhf_model"
    
    if not os.path.exists(rlhf_model_path):
        print(f"❌ RLHF model not found at {rlhf_model_path}")
        print("💡 Run training first!")
        return None
    
    rlhf_model = GPT2LMHeadModel.from_pretrained(rlhf_model_path)
    
    # Initialize evaluator
    evaluator = RLHFEvaluator(config)
    
    # Compare models
    comparison = evaluator.compare_models(base_model, rlhf_model, test_prompts)
    evaluator.create_evaluation_report(comparison)
    
    return comparison

# Keep the original main() function for backward compatibility
def main():
    """Original main function - runs complete pipeline"""
    config = RLHFConfig()
    run_complete_pipeline(config, num_steps=50)
    run_evaluation(config)

if __name__ == "__main__":
    main()