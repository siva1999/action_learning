from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = FastAPI()

# Initialize the text generation model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-t5")
model = AutoModelForSeq2SeqLM.from_pretrained("fine-tuned-t5")
text2text_generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=-1)

class StoryRequest(BaseModel):
    keywords: List[str]
    emotion: str
    userpref: str

class EnhancedStoryRequest(BaseModel):
    keywords: List[str]
    emotion: str
    userpref: str
    num_characters: int
    characters: List[dict]
    desc_caption: str

class CustomPromptRequest(BaseModel):
    prompt: str

class StoryImageRequest(BaseModel):
    story: str
    max_segments: int = 5

class ImageResponse(BaseModel):
    image: str
    text: str


def break_story_into_segments(story, max_segments=5):
    sentences = story.split('. ')
    total_sentences = len(sentences)
    sentences_per_segment = total_sentences // max_segments + (total_sentences % max_segments > 0)
    segments = []
    current_segment = []
    for i, sentence in enumerate(sentences):
        current_segment.append(sentence)
        if (i + 1) % sentences_per_segment == 0 or (i + 1) == total_sentences:
            segments.append('. '.join(current_segment).strip())
            current_segment = []
    return segments

@app.post("/generate_story")
def generate_story(request: StoryRequest):
    prompt = (
        f"Generate a complete story that evokes a {request.emotion} emotion. The story should feature a "
        f"{request.keywords[0]}, a {request.keywords[1]}, and a {request.keywords[2]}. "
        f"Additionally, incorporate elements of {request.userpref} to enhance the narrative. "
        f"Ensure the {request.userpref} aspects are seamlessly integrated and contribute to the overall {request.emotion} tone of the story."
    )
    try:
        generated_story = text2text_generator(prompt, max_length=512, do_sample=True, top_p=0.95, num_return_sequences=1)
        return {"story": generated_story[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_enhanced_story")
def generate_enhanced_story(request: EnhancedStoryRequest):
    char_details = "; ".join([f"{char['name']}, a {char['profession']}" for char in request.characters])
    prompt = (
        f"Generate a complete story that evokes a {request.emotion} emotion. The story should feature a {request.keywords[0]}, a {request.keywords[1]}, and {request.keywords[2]}. "
        f"Additionally, incorporate elements of {request.userpref} to enhance the narrative. "
        f"Ensure the {request.userpref} aspects are seamlessly integrated and contribute to the overall {request.emotion} tone of the story. "
        f"The story should include {request.num_characters} characters: {char_details}. "
        f"Description: {request.desc_caption}"
    )
    try:
        generated_story = text2text_generator(prompt, max_length=512, do_sample=True, top_p=0.95, num_return_sequences=1)
        return {"story": generated_story[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_custom_story")
def generate_custom_story(request: CustomPromptRequest):
    try:
        generated_story = text2text_generator(request.prompt, max_length=512, do_sample=True, top_p=0.95, num_return_sequences=1)
        return {"story": generated_story[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
