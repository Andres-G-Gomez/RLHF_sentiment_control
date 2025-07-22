# RLHF Sentiment Control

A complete implementation of Reinforcement Learning from Human Feedback (RLHF) for training language models to generate more positive sentiment text. This project demonstrates the three-stage RLHF pipeline used in systems like ChatGPT, specifically focused on sentiment control.

## 🎯 Project Overview

This project implements RLHF to fine-tune GPT-2 models for positive sentiment generation using:
- **Reward Model Training**: Binary classifier trained on preference pairs
- **PPO Optimization**: Proximal Policy Optimization for policy improvement
- **Comprehensive Evaluation**: Automated comparison between base and RLHF models

## 🏗️ Architecture

The project follows the standard three-stage RLHF pipeline:

1. **Supervised Fine-tuning (SFT)**: Uses pre-trained GPT-2 as starting point
2. **Reward Model Training**: Trains sentiment classifier on preference data
3. **PPO Training**: Optimizes policy using reinforcement learning

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Base      │    │   Reward     │    │   RLHF      │
│   Model     │───▶│   Model      │───▶│   Model     │
│  (GPT-2)    │    │ (Sentiment)  │    │ (Optimized) │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 📁 Project Structure

```
RLHF_sentiment_control/
├── config.py              # Configuration management
├── reward.py               # Reward model implementation
├── data_processor.py       # Data loading and processing
├── rlhf_sentiment.py       # Main RLHF training pipeline
├── train.py               # Training script with CLI
├── evaluate.py            # Model evaluation and comparison
├── config.yaml           # Configuration file (auto-generated)
├── data/                  # Training data directory
├── models/                # Saved models directory
└── results/               # Evaluation results and plots
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers datasets trl pyyaml matplotlib tqdm numpy
```

### Basic Usage

1. **Train the complete RLHF pipeline:**
```bash
python train.py --steps 100 --batch-size 16
```

2. **Evaluate the trained model:**
```bash
python evaluate.py --num-samples 10
```

3. **Run with custom configuration:**
```bash
python train.py --config custom_config.yaml --steps 50
```

### Python API Usage

```python
from rlhf_sentiment import RLHFConfig, run_complete_pipeline, run_evaluation

# Load configuration
config = RLHFConfig()

# Run training pipeline
run_complete_pipeline(config, num_steps=50)

# Evaluate results
comparison = run_evaluation(config)
```

## ⚙️ Configuration

The project uses YAML configuration with sensible defaults:

```yaml
# Model settings
model_name: "distilgpt2"
max_length: 50
temperature: 0.7
top_p: 0.9

# Training settings
batch_size: 16
ppo_epochs: 4
learning_rate: 1.41e-5
reward_model_lr: 5e-5

# Paths
data_dir: "./data"
model_dir: "./models"
results_dir: "./results"

# Experiment settings
use_wandb: false
device: "auto"
```

## 📊 Evaluation Metrics

The evaluation framework provides:

- **Sentiment Scores**: Mean positive sentiment probability
- **Positive Ratio**: Percentage of responses classified as positive
- **Statistical Analysis**: Standard deviation and improvement metrics
- **Sample Comparisons**: Side-by-side output examples
- **Visualizations**: Bar charts and comparison plots

### Example Results

```
RLHF EVALUATION REPORT
=====================================

Sentiment Analysis Results:
Base Model - Mean Positive Score: 0.623
RLHF Model - Mean Positive Score: 0.847
Improvement: 0.224

Positive Response Ratio:
Base Model: 0.600
RLHF Model: 0.900
Improvement: 0.300
```

## 🔧 Advanced Usage

### Custom Training Data

```python
from data_processor import PreferenceDataProcessor

# Load custom preference data
processor = PreferenceDataProcessor(config)
dataset = processor.load_custom_data("path/to/data.json")
```

### Custom Reward Model

```python
from reward import RewardModel

# Train custom reward model
reward_model = RewardModel(config)
reward_model.train_reward_model(custom_dataset)
```

### Evaluation with Custom Prompts

```python
custom_prompts = [
    "Tell me about your favorite book",
    "Describe a perfect day",
    "What makes you happy?"
]

comparison = run_evaluation(config, custom_prompts)
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Enable wandb logging
python train.py --config config.yaml
```

Set `use_wandb: true` in your config file to track:
- Training metrics
- Reward evolution
- Model performance
- Hyperparameter experiments

### Local Logging

All results are automatically saved to:
- `results/evaluation_results.json` - Detailed metrics
- `results/comparison_plot.png` - Visualization charts
- `models/` - Trained model checkpoints

## 🧪 Research Applications

This codebase can be adapted for various RLHF research:

- **Different Objectives**: Modify reward model for helpfulness, harmlessness, etc.
- **Model Scaling**: Test with larger language models (GPT-J, LLaMA)
- **Algorithm Variants**: Implement DPO, RLAIF, or other alignment methods
- **Multi-objective Optimization**: Combine multiple reward signals

## 📚 Key Components

### Reward Model (`reward.py`)
- Binary sentiment classifier
- Handles preference pair training
- Provides probability scores for text quality

### PPO Trainer (`rlhf_sentiment.py`)
- Implements Proximal Policy Optimization
- Handles generation and reward collection
- Updates policy to maximize expected rewards

### Data Processor (`data_processor.py`)
- Loads Anthropic HH-RLHF dataset
- Generates synthetic preference pairs as fallback
- Processes conversational data into prompt-response pairs

### Evaluation Framework (`evaluate.py`)
- Compares base vs RLHF models
- Automated sentiment analysis
- Statistical significance testing
- Rich visualization and reporting

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)

---

**Built with ❤️ for advancing AI alignment research**
