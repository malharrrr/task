import io
import base64
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import rembg
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    RMBGModel.get_instance()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RMBGModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(orig_size, Image.Resampling.LANCZOS)

        image.putalpha(mask)
        output_io = io.BytesIO()
        image.save(output_io, format="PNG")
        return output_io.getvalue()

async def get_image_description(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "model": "moondream",
        "prompt": "Provide a 1-sentence caption of the image subject.",
        "images": [base64_image],
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("http://ollama:11434/api/generate", json=payload, timeout=120.0)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        raise HTTPException(status_code=500, detail="Ollama vision model request failed")

async def remove_bg_rembg(image_bytes: bytes) -> str:
    loop = asyncio.get_event_loop()
    output_bytes = await loop.run_in_executor(None, rembg.remove, image_bytes)
    return base64.b64encode(output_bytes).decode('utf-8')

@app.post("/remove-bg")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        model = RMBGModel.get_instance()
        processed_bytes = model.process(image_bytes)
        return Response(content=processed_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        description_task = get_image_description(image_bytes)
        bg_removal_task = remove_bg_rembg(image_bytes)
        
        description, processed_image_b64 = await asyncio.gather(description_task, bg_removal_task)
        
        return {
            "description": description,
            "image_base64": processed_image_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")