"""
Evaluation script for RLHF models.
Compare base model vs RLHF trained model.
"""

import os
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer
from config import RLHFConfig
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from config import RLHFConfig
import matplotlib.pyplot as plt
import json

class RLHFEvaluator:
    """Evaluate RLHF model performance"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load sentiment analyzer for evaluation
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            print("Sentiment analyzer not available, using simple heuristic")
            self.sentiment_analyzer = None
    
    def evaluate_sentiment(self, texts: List[str]) -> Dict:
        """Evaluate sentiment of generated texts"""
        if self.sentiment_analyzer:
            results = self.sentiment_analyzer(texts)
            positive_scores = []
            for result in results:
                # Find positive score
                for score in result:
                    if score['label'] == 'LABEL_2':  # Positive
                        positive_scores.append(score['score'])
                        break
            
            return {
                "mean_positive_score": np.mean(positive_scores),
                "std_positive_score": np.std(positive_scores),
                "positive_ratio": np.mean([s > 0.5 for s in positive_scores])
            }
        else:
            # Simple heuristic evaluation
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "positive", "happy", "joy", "love"]
            negative_words = ["bad", "terrible", "awful", "horrible", "negative", "sad", "hate", "disappointed", "frustrated"]
            
            scores = []
            for text in texts:
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count + neg_count == 0:
                    scores.append(0.5)
                else:
                    scores.append(pos_count / (pos_count + neg_count))
            
            return {
                "mean_positive_score": np.mean(scores),
                "std_positive_score": np.std(scores),
                "positive_ratio": np.mean([s > 0.5 for s in scores])
            }
    
    def compare_models(self, base_model, rlhf_model, test_prompts: List[str]) -> Dict:
        """Compare base model vs RLHF model"""
        print("Evaluating models...")
        
        # Generate from base model
        base_responses = []
        for prompt in tqdm(test_prompts, desc="Base model generation"):
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = base_model.generate(
                    inputs,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            base_responses.append(response)
        
        # Generate from RLHF model
        rlhf_responses = []
        for prompt in tqdm(test_prompts, desc="RLHF model generation"):
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = rlhf_model.generate(
                    inputs,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            rlhf_responses.append(response)
        
        # Evaluate sentiment
        base_sentiment = self.evaluate_sentiment(base_responses)
        rlhf_sentiment = self.evaluate_sentiment(rlhf_responses)
        
        # Create comparison
        comparison = {
            "base_model": base_sentiment,
            "rlhf_model": rlhf_sentiment,
            "improvement": {
                "positive_score": rlhf_sentiment["mean_positive_score"] - base_sentiment["mean_positive_score"],
                "positive_ratio": rlhf_sentiment["positive_ratio"] - base_sentiment["positive_ratio"]
            },
            "sample_outputs": {
                "base": base_responses[:5],
                "rlhf": rlhf_responses[:5]
            }
        }
        
        return comparison
    
    def create_evaluation_report(self, comparison: Dict):
        """Create evaluation report with visualizations"""
        print("\n" + "="*50)
        print("RLHF EVALUATION REPORT")
        print("="*50)
        
        base = comparison["base_model"]
        rlhf = comparison["rlhf_model"]
        improvement = comparison["improvement"]
        
        print(f"\nSentiment Analysis Results:")
        print(f"Base Model - Mean Positive Score: {base['mean_positive_score']:.3f}")
        print(f"RLHF Model - Mean Positive Score: {rlhf['mean_positive_score']:.3f}")
        print(f"Improvement: {improvement['positive_score']:.3f}")
        
        print(f"\nPositive Response Ratio:")
        print(f"Base Model: {base['positive_ratio']:.3f}")
        print(f"RLHF Model: {rlhf['positive_ratio']:.3f}")
        print(f"Improvement: {improvement['positive_ratio']:.3f}")
        
        print(f"\nSample Outputs:")
        print(f"\nBase Model Examples:")
        for i, output in enumerate(comparison["sample_outputs"]["base"]):
            print(f"{i+1}. {output[:100]}...")
        
        print(f"\nRLHF Model Examples:")
        for i, output in enumerate(comparison["sample_outputs"]["rlhf"]):
            print(f"{i+1}. {output[:100]}...")
        
        # Create visualization
        self.create_comparison_plot(comparison)
        
        # Save results
        os.makedirs(self.config.results_dir, exist_ok=True)
        with open(f"{self.config.results_dir}/evaluation_results.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to {self.config.results_dir}/evaluation_results.json")
    
    def create_comparison_plot(self, comparison: Dict):
        """Create comparison visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sentiment scores comparison
        models = ["Base Model", "RLHF Model"]
        scores = [
            comparison["base_model"]["mean_positive_score"],
            comparison["rlhf_model"]["mean_positive_score"]
        ]
        
        ax1.bar(models, scores, color=['skyblue', 'lightgreen'])
        ax1.set_ylabel('Mean Positive Score')
        ax1.set_title('Sentiment Score Comparison')
        ax1.set_ylim(0, 1)
        
        # Positive ratio comparison
        ratios = [
            comparison["base_model"]["positive_ratio"],
            comparison["rlhf_model"]["positive_ratio"]
        ]
        
        ax2.bar(models, ratios, color=['skyblue', 'lightgreen'])
        ax2.set_ylabel('Positive Response Ratio')
        ax2.set_title('Positive Response Ratio')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.results_dir}/comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

def run_evaluation_pipeline(config: RLHFConfig, test_prompts: list = None):
    """Run evaluation comparing base vs RLHF model"""
    
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
    
    # Check if RLHF model exists
    rlhf_model_path = f"{config.model_dir}/rlhf_model"
    if not os.path.exists(rlhf_model_path):
        print(f"❌ RLHF model not found at {rlhf_model_path}")
        print("💡 Run training first: python train.py")
        return None
    
    # Load models for comparison
    print("📥 Loading models...")
    base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
    rlhf_model = GPT2LMHeadModel.from_pretrained(rlhf_model_path)
    
    # Initialize evaluator
    evaluator = RLHFEvaluator(config)
    
    print(f"🧪 Generating {len(test_prompts)} test samples...")
    
    # Compare models
    comparison = evaluator.compare_models(base_model, rlhf_model, test_prompts)
    evaluator.create_evaluation_report(comparison)
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Evaluate RLHF model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of test prompts')
    parser.add_argument('--custom-prompts', nargs='+', help='Custom test prompts')
    
    args = parser.parse_args()
    
    # Load config
    config = RLHFConfig()
    print(f"📋 Loaded configuration")
    
    # Prepare test prompts
    if args.custom_prompts:
        test_prompts = args.custom_prompts
    else:
        default_prompts = [
            "Tell me about your day",
            "What do you think about technology?",
            "How would you describe the weather?",
            "Please write about your favorite hobby",
            "What are your thoughts on learning?",
            "Describe a beautiful place",
            "How do you feel about music?",
            "Tell me something interesting",
            "What makes you happy?",
            "Share your opinion on books"
        ]
        test_prompts = default_prompts[:args.num_samples]
    
    print(f"🔍 Evaluating with {len(test_prompts)} test prompts")
    
    # Run evaluation
    comparison = run_evaluation_pipeline(config, test_prompts)
    
    if comparison:
        print("✅ Evaluation completed! Check results/ directory for detailed report.")
    else:
        print("❌ Evaluation failed. Make sure to train the model first.")

if __name__ == "__main__":
    main()
