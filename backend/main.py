from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from aiogram.types import InputFile
import io
from io import BytesIO
from PIL import Image
from u2net_utils import load_model, remove_background
from aiogram import Bot
import os

app = FastAPI()

# Ваш токен бота
BOT_TOKEN = "7932051624:AAFdRF9vgQ044ps1GgpBAf5mSZ1xNu9B4Zg"
bot = Bot(BOT_TOKEN)

# Если используете ngrok или cloudflare
origins = [
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Лучше указать конкретные URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()


@app.post("/remove_bg")
async def remove_bg(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents))
    result = remove_background(model, pil_image)

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/send_to_chat")
async def send_to_chat(
    chat_id: str = Form(...),
    image: UploadFile = File(...)
):
    # получаем имя файла
    filename = image.filename  
    print(f"Получен запрос send_to_chat:")
    print(f"chat_id = {chat_id}")
    print(f"image filename = {image.filename}")

    """
    Этот endpoint принимает chat_id и картинку, которую отправит в чат.
    """
    image_bytes = await image.read()
    print(f"Размер файла = {len(image_bytes)} байт")
    from aiogram.types import BufferedInputFile

    buffer = io.BytesIO(image_bytes)
    input_file = BufferedInputFile(file=buffer.getvalue(), filename=filename) # Имя файла, можно любое


    await bot.send_document(chat_id=chat_id, document=input_file, caption="Ваше изображение без фона 🌟"
)

    return JSONResponse({"status": "ok"})


# Запуск при ручном старте
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
