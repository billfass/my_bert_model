

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
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained("billfass/multilingual_bert_model_classiffication")
model = BertModel.from_pretrained("billfass/multilingual_bert_model_classiffication")

def prediction(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    logits = output.last_hidden_state
    linear_layer = nn.Linear(768, 3)
    predict = torch.max(linear_layer(logits.view(-1, 768)), dim=1)
    label_pred = predict.indices[torch.argmax(predict.values).item()].item()

    return {
        "label" : label_pred
    }

print(prediction("Hello world", "I'm here"))

# or you can simply lunch demo.ipynb file to test
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
