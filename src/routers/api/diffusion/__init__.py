from fastapi import APIRouter
import random
import io
import base64
from src.config import settings
from src.schemas.api.diffusion import ApiDiffusionGetResponse
from src.utils.diffusion import generate_image, stable_diffusion, waifu_diffusion
from PIL.PngImagePlugin import PngInfo

router = APIRouter()

if settings.diffusion_mode == "wd":
    pipe = waifu_diffusion()
else:
    pipe = stable_diffusion()


@router.get("/api/diffusion", response_model=ApiDiffusionGetResponse)
async def main(prompt: str = "1 girl", negative_prompt: str = ""):
    seed = random.randrange(0, 4294967295, 1)
    img = generate_image(pipe, prompt, negative_prompt, seed)
    metadata = PngInfo()
    metadata.add_text("prompt", prompt)
    metadata.add_text("negative_prompt", negative_prompt)
    metadata.add_text("seed", str(seed))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG", pnginfo=metadata)

    img_base64 = base64.b64encode(img_bytes.getvalue())
    return ApiDiffusionGetResponse(url=img_base64)
