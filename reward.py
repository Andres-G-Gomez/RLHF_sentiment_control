
import os
import torch
from typing import List
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
from config import RLHFConfig

class RewardModel:
    """Reward model for scoring text based on sentiment"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
    
    def prepare_training_data(self, dataset: Dataset) -> Dataset:
        """Prepare data for reward model training"""
        training_data = []
        
        for example in dataset:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Create positive example
            training_data.append({
                "text": prompt + " " + chosen,
                "label": 1
            })
            
            # Create negative example
            training_data.append({
                "text": prompt + " " + rejected,
                "label": 0
            })
        
        return Dataset.from_list(training_data)
    
    def train_reward_model(self, dataset: Dataset):
        """Train the reward model"""
        print("Training reward model...")
        
        # Prepare training data
        train_dataset = self.prepare_training_data(dataset)
        
        # Load model for sequence classification
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Tokenize data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
        
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Training arguments        
        # Trainer
        training_args = TrainingArguments(
            output_dir=f"{self.config.model_dir}/reward_model",
            num_train_epochs=3,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=float(self.config.reward_model_lr),
            warmup_steps=100,
            logging_steps=50,
            save_strategy="epoch",
            report_to="none",  # ADD THIS LINE
        )   
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        model.save_pretrained(f"{self.config.model_dir}/reward_model")
        self.tokenizer.save_pretrained(f"{self.config.model_dir}/reward_model")
        
        self.model = model
        print("Reward model training complete!")
    
    def load_reward_model(self):
        """Load trained reward model"""
        model_path = f"{self.config.model_dir}/reward_model"
        if os.path.exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            print("Reward model loaded successfully!")
        else:
            raise FileNotFoundError("Reward model not found. Please train first.")
    
    def get_reward(self, texts: List[str]) -> List[float]:
        """Get reward scores for texts"""
        if self.model is None:
            raise ValueError("Reward model not loaded")
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1)[:, 1]  # Probability of positive class
        
        return scores.tolist()