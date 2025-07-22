import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from tqdm import tqdm
from config import RLHFConfig

class PreferenceDataProcessor:
    """Handle preference data collection and processing"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_hh_rlhf_data(self) -> Dataset:
        """Load Anthropic's HH-RLHF dataset"""
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
            return self.process_hh_dataset(dataset)
        except:
            print("HH-RLHF dataset not available, generating synthetic data...")
            return self.generate_synthetic_data()
    
    def process_hh_dataset(self, dataset) -> Dataset:
        """Process HH-RLHF dataset into preference pairs"""
        processed_data = []
        
        for example in tqdm(dataset, desc="Processing HH-RLHF data"):
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Extract prompt (usually everything before the last Assistant response)
            prompt = self.extract_prompt(chosen)
            chosen_response = self.extract_response(chosen)
            rejected_response = self.extract_response(rejected)
            
            processed_data.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "chosen_score": 1.0,
                "rejected_score": 0.0
            })
        
        return Dataset.from_list(processed_data)
    
    def extract_prompt(self, conversation: str) -> str:
        """Extract prompt from conversation"""
        # Simple extraction - take everything up to the last Assistant response
        parts = conversation.split("Assistant:")
        if len(parts) > 1:
            return "Human: " + parts[0].split("Human:")[-1].strip()
        return conversation[:100]  # Fallback
    
    def extract_response(self, conversation: str) -> str:
        """Extract response from conversation"""
        parts = conversation.split("Assistant:")
        if len(parts) > 1:
            return parts[-1].strip()
        return conversation  # Fallback
    
    def generate_synthetic_data(self) -> Dataset:
        """Generate synthetic preference data for sentiment control"""
        prompts = [
            "Tell me about your day",
            "What do you think about",
            "How would you describe",
            "Please write about",
            "Can you explain",
            "What are your thoughts on",
            "Describe the experience of",
            "How do you feel about",
            "What's your opinion on",
            "Tell me something about"
        ]
        
        positive_responses = [
            "I'm absolutely thrilled to share that today has been wonderful!",
            "I have such positive thoughts about this amazing topic!",
            "I would describe this as incredibly beautiful and inspiring.",
            "I'm delighted to write about this fantastic subject!",
            "I'm happy to explain this exciting concept clearly.",
            "I have wonderfully optimistic thoughts about this!",
            "The experience is absolutely magnificent and joyful.",
            "I feel incredibly positive and enthusiastic about this!",
            "My opinion is very favorable and uplifting about this.",
            "I'm excited to share something truly remarkable!"
        ]
        
        negative_responses = [
            "This day has been quite disappointing and frustrating.",
            "I have rather negative thoughts about this concerning topic.",
            "I would describe this as somewhat depressing and uninspiring.",
            "I'm reluctant to write about this troubling subject.",
            "I find this concept rather confusing and problematic.",
            "I have pessimistic thoughts about this unfortunate situation.",
            "The experience is quite unpleasant and distressing.",
            "I feel rather negative and discouraged about this.",
            "My opinion is unfavorable and concerning about this.",
            "I'm hesitant to share something so disappointing."
        ]
        
        data = []
        for i, prompt in enumerate(prompts):
            data.append({
                "prompt": prompt,
                "chosen": positive_responses[i],
                "rejected": negative_responses[i],
                "chosen_score": 1.0,
                "rejected_score": 0.0
            })
        
        # Duplicate and shuffle for more training data
        data = data * 10
        np.random.shuffle(data)
        
        return Dataset.from_list(data)