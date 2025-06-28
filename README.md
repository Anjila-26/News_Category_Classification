# Transformer from Scratch + FineTuning Approaches for News Category Classification

A complete implementation of a Transformer encoder model built from scratch using PyTorch for news article category classification. This project demonstrates the core components of the Transformer architecture and applies it to multi-class text classification.

## 📊 Project Overview

This project implements a Transformer-based text classifier that categorizes news articles into 42 different categories. The model is built entirely from scratch, providing a deep understanding of the Transformer architecture components. Additionally, it includes comprehensive fine-tuning experiments with pre-trained models to demonstrate different training strategies and their trade-offs.

### Dataset
- **Source**: News Category Dataset v3
- **Size**: ~200k news articles
- **Categories**: 42 different news categories
- **Features**: Headlines and short descriptions combined

### Implementation Approaches
1. **From-Scratch Transformer**: Complete implementation of encoder-only Transformer
2. **Fine-tuning Comparison**: Multiple fine-tuning strategies with DeBERTa-v3-base
3. **Performance Analysis**: Comprehensive comparison of accuracy, training time, and memory usage

## 🏗️ Architecture Overview

The implementation consists of several key components that work together to create a complete Transformer encoder:

```
Input Text → Tokenization → Embeddings → Positional Encoding → 
Multi-Head Attention → Feed-Forward → Classification Head → Output
```

## 🔧 Core Components

### 1. Input Embeddings
Converts discrete token indices into dense vector representations with scaling.

**Mathematical Foundation:**
- Embedding dimension scaling: `embedding(x) * sqrt(d_model)`
- Purpose: Helps with training convergence as recommended in the original paper

```python
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

### 2. Positional Encoding
Adds position information to embeddings using sinusoidal functions.

**Mathematical Foundation:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Key Features:**
- Uses sine for even dimensions, cosine for odd dimensions
- Provides unique encoding for each position
- Enables the model to understand sequence order
- Allows for relative position learning

### 3. Multi-Head Attention
The core mechanism that allows tokens to attend to other tokens in the sequence.

**Mathematical Foundation:**
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
```

**Components:**
- **Query (Q)**: What each token is looking for
- **Key (K)**: What each token represents for matching
- **Value (V)**: The actual content to be aggregated
- **Attention Scores**: Computed via dot-product similarity
- **Attention Weights**: Normalized scores using softmax

**Multi-Head Benefits:**
- Each head learns different types of relationships
- Some heads focus on syntax, others on semantics
- Parallel processing of different attention patterns
- Increased model expressiveness

### 4. Feed-Forward Network
Position-wise fully connected layers that add non-linearity.

**Architecture:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Purpose:**
- Adds non-linear transformations
- Each position processed independently
- Typically expands then contracts dimensions (d_model → d_ff → d_model)

### 5. Encoder Layer
Combines attention and feed-forward with residual connections and layer normalization.

**Structure:**
1. **Sublayer 1**: Multi-Head Self-Attention + Residual + LayerNorm
2. **Sublayer 2**: Feed-Forward + Residual + LayerNorm

**Residual Connections:**
- Help with gradient flow during training
- Enable training of deeper networks
- Formula: `LayerNorm(x + Sublayer(x))`

### 6. Complete Transformer Encoder
Stacks multiple encoder layers to build the full model.

**Components:**
- Input embeddings
- Positional encoding
- N stacked encoder layers
- Classification head

### 7. Classification Head
Maps the encoded representations to class probabilities.

**Architecture:**
- Takes the first token embedding (CLS token approach)
- Single linear layer: `d_model → num_classes`
- Returns raw logits for CrossEntropyLoss

## 📋 Model Configuration

```python
# Model Hyperparameters
vocab_size = len(word2idx)        # Vocabulary size (dynamic)
d_model = 512                     # Embedding dimension
num_layers = 2                    # Number of encoder layers
num_heads = 8                     # Number of attention heads
d_ff = 2048                       # Feed-forward hidden dimension
dropout = 0.1                     # Dropout rate
max_seq_length = 32               # Maximum sequence length
num_classes = 42                  # Number of categories
```

## 🔄 Data Processing Pipeline

### 1. Text Preprocessing
```python
def clean_text(text):
    # Convert to lowercase
    # Remove URLs and HTML tags
    # Remove punctuation and special characters
    # Remove extra whitespace
```

### 2. Vocabulary Building
```python
def build_vocab(texts, min_freq=1):
    # Count token frequencies
    # Filter by minimum frequency
    # Add special tokens: <PAD>, <UNK>
    # Create word-to-index mapping
```

### 3. Text Encoding
```python
def encode_text(text, word2idx, max_len=32):
    # Tokenize text
    # Convert to indices
    # Apply padding/truncation
```

### 4. Dataset Class
```python
class NewsDataset(Dataset):
    # Custom PyTorch Dataset
    # Handles text encoding and label conversion
    # Returns tensors for model training
```

## 🎯 Training Process

### 1. Loss Function
- **CrossEntropyLoss** with class weights to handle imbalanced data
- Class weights computed inversely proportional to frequency

### 2. Optimizer
- **Adam optimizer** with learning rate: 1e-4
- Adaptive learning rate for stable training

### 3. Training Loop
- **Batch size**: 64
- **Epochs**: 20 (with early stopping)
- **Early stopping**: Patience of 3 epochs
- **Validation split**: 80/20 train/validation

### 4. Evaluation Metrics
- **Loss**: CrossEntropyLoss
- **Accuracy**: Classification accuracy
- **Early Stopping**: Based on validation loss

## 📊 Model Performance

The model demonstrates the effectiveness of the Transformer architecture for text classification:

- **Multi-class classification**: 42 categories
- **Class imbalance handling**: Weighted loss function
- **Regularization**: Dropout and early stopping
- **Validation**: Separate validation set for monitoring

## 🚀 Usage Examples

### Making Predictions
```python
# Example predictions
texts_to_predict = [
    "The economy is showing signs of recovery after a challenging year.",
    "New breakthrough in cancer research offers hope for many patients.",
    "The government has announced new policies to tackle climate change."
]

for text in texts_to_predict:
    predicted_category, probabilities = predict_category(
        text, model, word2idx, idx2label, max_len=32, device=device
    )
    print(f"Text: {text}")
    print(f"Predicted category: {predicted_category}")
```

### Top-K Predictions
```python
def top_k_categories(probs, idx2label, k=3):
    topk_idx = np.argsort(probs)[::-1][:k]
    return [(idx2label[i], probs[i]) for i in topk_idx]
```

## 🧠 Key Learning Concepts

### 1. Attention Mechanism
- **Self-attention**: Tokens attend to other tokens in the same sequence
- **Scaled dot-product**: Prevents attention scores from becoming too large
- **Multi-head**: Multiple attention patterns learned in parallel

### 2. Positional Information
- **Absolute positions**: Sinusoidal encoding provides position information
- **Relative positions**: Model can learn relationships between positions
- **No recurrence**: Parallel processing of all positions

### 3. Residual Learning
- **Skip connections**: Direct paths for gradient flow
- **Layer normalization**: Stabilizes training
- **Deep networks**: Enables training of deeper architectures

### 4. Transfer Learning Ready
- **Modular design**: Components can be reused
- **Encoder-only**: Suitable for classification tasks
- **Scalable**: Can adjust model size via hyperparameters

## 📁 File Structure

```
├── transformer-from-scratch-news-category.ipynb  # From-scratch Transformer implementation
├── category-classification-bert.ipynb            # Pre-trained model fine-tuning comparison
├── README.md                                     # This comprehensive documentation
├── best_model.pth                               # Saved from-scratch model weights
└── results/                                     # Fine-tuning experiment outputs
    ├── checkpoint-*/                            # Training checkpoints
    └── logs/                                   # Training logs and metrics
```

## 🔬 Technical Details

### Memory and Computational Complexity
- **Attention complexity**: O(n²d) where n = sequence length, d = model dimension
- **Memory usage**: Quadratic in sequence length due to attention matrix
- **Parallelization**: All positions processed simultaneously

### Hyperparameter Considerations
- **d_model**: Balance between capacity and computational cost
- **num_heads**: Should divide d_model evenly
- **d_ff**: Typically 4x d_model for sufficient capacity
- **num_layers**: Deeper networks may need careful initialization

### Training Considerations
- **Gradient clipping**: May be needed for stable training
- **Learning rate scheduling**: Can improve convergence
- **Warmup**: Gradual learning rate increase often beneficial

## 🎓 Educational Value

This implementation serves as an excellent learning resource for understanding:

1. **Transformer Architecture**: From basic components to full model
2. **Attention Mechanisms**: Self-attention and multi-head attention
3. **Deep Learning Best Practices**: Residual connections, normalization
4. **Text Classification**: End-to-end NLP pipeline
5. **PyTorch Implementation**: Professional deep learning code structure
6. **Fine-tuning Strategies**: Comprehensive comparison of different approaches
7. **Performance Trade-offs**: Understanding accuracy vs efficiency considerations
8. **Transfer Learning**: Leveraging pre-trained models effectively
9. **Parameter Efficiency**: Modern techniques like LoRA and BitFit
10. **Experimental Design**: Systematic comparison methodology

## � Fine-Tuning Techniques Comparison

This project also includes a comprehensive comparison of different fine-tuning approaches using pre-trained models (DeBERTa-v3-base) to establish baselines and compare with the from-scratch implementation.

### 🎯 Fine-Tuning Approaches Implemented

#### 1. **Full Fine-Tuning**
- **Description**: Updates all model parameters during training
- **Use Case**: When you have sufficient data and computational resources
- **Trade-offs**:
  - ✅ **Accuracy**: Highest potential accuracy
  - ❌ **Training Time**: Longest training time
  - ❌ **Memory Usage**: Highest memory consumption
  - ❌ **Storage**: Requires storing full model weights

#### 2. **Frozen Encoder**
- **Description**: Freezes pre-trained encoder weights, only trains classification head
- **Use Case**: Limited data or computational resources
- **Trade-offs**:
  - ⚠️ **Accuracy**: Lower accuracy, relies on pre-trained features
  - ✅ **Training Time**: Fastest training
  - ✅ **Memory Usage**: Lowest memory consumption
  - ✅ **Storage**: Minimal storage requirements

#### 3. **LoRA (Low-Rank Adaptation)**
- **Description**: Adds trainable low-rank matrices to attention layers
- **Use Case**: Efficient fine-tuning with good performance
- **Trade-offs**:
  - ✅ **Accuracy**: Near full fine-tuning performance
  - ✅ **Training Time**: Moderate training time
  - ✅ **Memory Usage**: Significantly reduced memory usage
  - ✅ **Storage**: Only stores small adapter weights

#### 4. **BitFit**
- **Description**: Only fine-tunes bias parameters throughout the model
- **Use Case**: Extremely parameter-efficient fine-tuning
- **Trade-offs**:
  - ⚠️ **Accuracy**: Good accuracy with minimal parameters
  - ✅ **Training Time**: Fast training
  - ✅ **Memory Usage**: Low memory consumption
  - ✅ **Storage**: Minimal storage (only bias terms)

### 📊 Performance Analysis Framework

```python
def train_and_evaluate(model, name):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    print(f"\n🟢 Starting: {name}")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    metrics = trainer.evaluate()
    accuracy = metrics['eval_accuracy']
    
    print(f"🔍 {name} Results")
    print(f"- Accuracy: {accuracy}")
    print(f"- Training Time: {(end_time - start_time):.2f} sec")
    print(f"- Max Memory Used: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
```

### 🏆 Expected Performance Trade-offs

| Method | Accuracy | Training Time | Memory Usage | Parameters Updated | Storage |
|--------|----------|---------------|--------------|-------------------|---------|
| **Full Fine-tuning** | 🟢 Highest | 🔴 Slowest | 🔴 Highest | 100% | 🔴 Full Model |
| **LoRA** | 🟢 High | 🟡 Moderate | 🟢 Low | <1% | 🟢 Adapters Only |
| **BitFit** | 🟡 Good | 🟢 Fast | 🟢 Low | <0.1% | 🟢 Bias Only |
| **Frozen Encoder** | 🟡 Lower | 🟢 Fastest | 🟢 Lowest | <1% | 🟢 Head Only |

### 🔬 Technical Implementation Details

#### LoRA Configuration
```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # Rank of adaptation
    lora_alpha=32,          # LoRA scaling parameter
    lora_dropout=0.1,       # Dropout for LoRA layers
    bias="none"             # Bias handling strategy
)
```

#### Model Selection Strategy
```python
# Choose based on your constraints:
inference_model = lora_model        # Best balance
# inference_model = full_model      # Maximum accuracy
# inference_model = bitfit_model    # Minimal resources
# inference_model = frozen_model    # Fastest inference
```

### 🎯 When to Use Each Approach

#### **Full Fine-tuning**
- Large datasets (>100k samples)
- High-resource environments
- Maximum accuracy requirements
- Domain-specific tasks requiring significant adaptation

#### **LoRA (Recommended)**
- Balanced accuracy and efficiency needs
- Multiple task adaptation
- Limited computational resources
- Production deployments

#### **BitFit**
- Extremely limited resources
- Quick prototyping
- Similar domain tasks
- When storage is critical

#### **Frozen Encoder**
- Very limited data (<1k samples)
- Rapid experimentation
- Pre-trained features are sufficient
- Minimal computational budget

### 🔄 Comparison with From-Scratch Implementation

| Aspect | From-Scratch Transformer | Pre-trained + Fine-tuning |
|--------|--------------------------|---------------------------|
| **Learning** | Complete architecture understanding | Transfer learning concepts |
| **Performance** | Depends on architecture & data | Leverages pre-trained knowledge |
| **Training Time** | Longer (training from random weights) | Shorter (starting from good weights) |
| **Data Requirements** | Large datasets needed | Works with smaller datasets |
| **Customization** | Full control over architecture | Limited to model modifications |
| **Educational Value** | Deep transformer understanding | Fine-tuning best practices |

### 🎓 Key Insights from Fine-tuning Experiments

1. **LoRA typically provides 95%+ of full fine-tuning performance with 10% of the memory**
2. **BitFit can achieve surprising accuracy by only updating 0.08% of parameters**
3. **Frozen encoders work well when pre-trained features align with target task**
4. **Training time scales roughly with number of trainable parameters**
5. **Memory usage is dominated by gradient computation and optimizer states**

## �🔮 Future Enhancements

Potential improvements and extensions:

1. **Model Architecture**:
   - Implement encoder-decoder architecture
   - Add cross-attention mechanisms
   - Experiment with different positional encodings
   - Compare with other pre-trained models (RoBERTa, ALBERT, etc.)

2. **Fine-tuning Improvements**:
   - Implement AdaLoRA (adaptive LoRA)
   - Experiment with different LoRA ranks
   - Try prompt tuning and prefix tuning
   - Implement QLoRA for quantized fine-tuning

3. **Training Improvements**:
   - Learning rate scheduling
   - Gradient clipping
   - Mixed precision training
   - Data augmentation techniques
   - Curriculum learning strategies

4. **Evaluation**:
   - Detailed metrics (F1, precision, recall)
   - Confusion matrix analysis
   - Error analysis and model interpretation
   - Cross-validation for robust evaluation
   - Statistical significance testing

5. **Optimization**:
   - Model pruning and quantization
   - Knowledge distillation
   - Efficient attention variants
   - Dynamic batching and gradient accumulation

6. **Comparative Analysis**:
   - Benchmark against other architectures
   - Ablation studies on components
   - Hyperparameter sensitivity analysis
   - Scalability studies across different data sizes

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

This educational project welcomes contributions for:
- Code improvements and optimizations
- Additional documentation and examples
- Bug fixes and error handling
- Performance benchmarking
- Alternative architectural implementations
- New fine-tuning techniques implementation
- Comparative analysis with other models
- Visualization and interpretation tools

---

*This implementation provides a comprehensive understanding of Transformer architecture while solving a practical text classification problem. The combination of from-scratch implementation and fine-tuning comparisons offers both theoretical understanding and practical insights into modern NLP approaches. The modular design makes it easy to understand, modify, and extend for various NLP tasks.*
