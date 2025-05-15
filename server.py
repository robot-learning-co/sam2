from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import time
import json
from pydantic import BaseModel

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

timestamp = str(int(time.time() * 1000))
readable_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(int(timestamp) // 1000))
output_dir = os.path.join(f"{os.getcwd()}/data/", f"{readable_time}")


@app.post("/segment")
async def segment(
    request: Request, 
    image: UploadFile = File(...),
    prompt: str = Form(...),
):    
    client_host = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    referer = request.headers.get("referer", "none")

    img = await image.read()
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{image.filename}"), "wb") as f:
        f.write(img)
    
    print(f"Saved image to {output_dir}/{image.filename}")
    
    image = Image.open(f"{output_dir}/{image.filename}")    

    prompt = json.loads(prompt)
    point_coords = prompt["point_coords"] if "point_coords" in prompt else None
    point_labels = prompt["point_labels"] if "point_labels" in prompt else None
    box = prompt["box"] if "box" in prompt else None
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(    
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )    
        
    predictor.reset_predictor()
    
    return {"mask": masks[0].tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app",
                host="0.0.0.0",
                port=8000,
                reload=True)