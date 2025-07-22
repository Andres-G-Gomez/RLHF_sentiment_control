import os
import yaml
import torch

class RLHFConfig:
    def __init__(self, config_path: str = "config.yaml", use_yaml: bool = True):
        # Start with defaults
        defaults = self.get_defaults()
        
        # Set default attributes first
        for key, value in defaults.items():
            setattr(self, key, value)
        
        # Override with YAML if requested and file exists
        if use_yaml and os.path.exists(config_path):
            print(f"📋 Loading config from {config_path}")
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                for key, value in yaml_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        print(f"⚠️  Unknown config key: {key}")
        elif use_yaml:
            print(f"⚠️  Config file {config_path} not found, using defaults")
        
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🖥️  Using device: {self.device}")
    
    def get_defaults(self):
        """Default configuration values"""
        return {
            # Model settings
            "model_name": "distilgpt2",
            "reward_model_name": "sentiment-reward-model",
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            
            # Training settings
            "batch_size": 16,
            "ppo_epochs": 4,
            "learning_rate": 1.41e-5,
            "reward_model_lr": 5e-5,
            
            # Paths
            "data_dir": "./data",
            "model_dir": "./models",
            "results_dir": "./results",
            
            # Experiment settings
            "use_wandb": False,
            "device": "auto",
            
            # Test prompts for evaluation
            "test_prompts": [
                "Tell me about your day",
                "What do you think about technology?",
                "How would you describe the weather?",
                "Please write about your favorite hobby",
                "What are your thoughts on learning?"
            ]
        }
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            setattr(self, key, value)
            print(f"📝 Updated {key} = {value}")
    
    def save_config(self, path: str):
        """Save current configuration to YAML file"""
        config_dict = {}
        
        # Get all attributes that don't start with underscore
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"💾 Configuration saved to {path}")
    
    def __str__(self):
        """String representation of config"""
        lines = ["RLHF Configuration:"]
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if key == "test_prompts":
                    lines.append(f"  {key}: {len(value)} prompts")
                else:
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)

# Test the config loading
if __name__ == "__main__":
    print("Testing RLHFConfig...")
    config = RLHFConfig()
    print("\nFinal config:")
    print(config)
    
