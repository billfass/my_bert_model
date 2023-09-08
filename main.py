from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("billfass/multilingual_bert_model_classiffication")
model = BertModel.from_pretrained("billfass/multilingual_bert_model_classiffication")

class InputData(BaseModel):
    premise: str
    hypothesis: str


def prediction(data: InputData):
    try:
        inputs = tokenizer(data.premise, data.hypothesis, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)

        logits = output.last_hidden_state
        linear_layer = nn.Linear(768, 3)
        predict = torch.max(linear_layer(logits.view(-1, 768)), dim=1)
        label_pred = predict.indices[torch.argmax(predict.values).item()].item()

        return {"label" : label_pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/prediction")
def predict_get(data: InputData):
    return prediction(data)

@app.post("/prediction")
def predict_post(data: InputData):
    return prediction(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


