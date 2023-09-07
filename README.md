

# Custom BERT Model for Text Classification

## Model Description

This is a custom BERT model fine-tuned for text classification. The model was trained using a subset of a publicly available dataset and is capable of classifying text into 3 classes.

## Training Details

- **Architecture**: BERT Base Multilingual Cased
- **Training data**: Custom dataset
- **Preprocessing**: Tokenized using BERT's tokenizer, with a max sequence length of 80.
- **Fine-tuning**: The model was trained for 1 epoch with a learning rate of 2e-5, using AdamW optimizer and Cross-Entropy Loss.
- **Evaluation Metrics**: Accuracy on a held-out validation set.
  
## How to Use

### Dependencies
- Transformers 4.x
- Torch 1.x

### Code Snippet

For classification:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("billfass/my_bert_model")
model = AutoModelForSequenceClassification.from_pretrained("billfass/my_bert_model")

text = "Your example text here."

inputs = tokenizer(text, padding=True, truncation=True, max_length=80, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

# To get probabilities:
probs = torch.softmax(logits, dim=-1)
```

## Limitations and Bias

- Trained on a specific dataset, so may not generalize well to other kinds of text.
- Uses multilingual cased BERT, so it's not optimized for any specific language.

## Authors

- **Fassinou Bile**
- **billfass2010@gmail.com**
  
## Acknowledgments

Special thanks to Hugging Face for providing the Transformers library that made this project possible.

---
