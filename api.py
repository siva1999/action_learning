from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

class TextGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate-text/")
async def generate_text(request: TextGenerationRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    result_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return {"generated_text": result_text}
