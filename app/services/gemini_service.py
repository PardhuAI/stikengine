import os
import json
import base64
import mimetypes
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# NEW: Helper function to get correct MIME type
def get_mime_type(image_path: str):
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"

def moderate_image_gemini(image_path: str, nudenet_out: dict, clip_out: dict, yolo_out: dict) -> dict:
    base64_image = encode_image(image_path)
    mime_type = get_mime_type(image_path) # Dynamically grab png/webp/jpeg
    
    context = json.dumps({
        "nudenet": nudenet_out,
        "clip": clip_out,
        "yolo": yolo_out
    }, indent=2)
    
    prompt = f"""
    You are the final human-level moderator for Stikbook, an Indian social media platform.
    The primary automated models have flagged this image as UNCERTAIN.
    Here are their outputs:
    
    {context}
    
    Your job is to look at the image and the model outputs, and make the FINAL decision.
    Is this SAFE or FLAG?
    
    Rules:
    - Shirtless men at the beach or sports are SAFE.
    - Bikinis and swimwear are SAFE unless highly suggestive/micro.
    - Explicit nudity, genitals, exposed buttocks are FLAG.
    - Sheer/transparent clothing revealing nipples/genitals is FLAG.
    - Violence or weapons are FLAG.
    
    You must output strictly JSON in this exact format:
    {{
        "decision": "SAFE" | "FLAG",
        "confidence": 0.0 - 1.0,
        "reason": "short explanation",
        "category": "SAFE" | "NUDITY" | "NEAR_NUDITY" | "SUGGESTIVE" | "WEAPON" | "VIOLENCE" | "DRUGS"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}" # Use dynamic MIME type
                            }
                        }
                    ]
                }
            ]
        )
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        return {
            "decision": "FLAG",
            "confidence": 0.0,
            "reason": f"Gemini API Error: {str(e)}",
            "category": "REVIEW"
        }
