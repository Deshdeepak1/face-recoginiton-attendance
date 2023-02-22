import asyncio
import pickle
import uuid

import aiofiles
import aiofiles.os
import face_recognition
from fastapi import Depends, FastAPI, File, UploadFile
from sqlalchemy.orm import Session

from .db import User, get_db

app = FastAPI(title="Face Recognition based attendance system")


@app.get("/")
async def read_root():
    return "Face Recognition based attendance system"


@app.post("/register")
async def register(
    name: str,
    email: str,
    user_image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    filename = str(uuid.uuid4())
    user_image_path = f"images/{filename}.jpg"
    async with aiofiles.open(user_image_path, "wb+") as file:
        await file.write(user_image_file.file.read())
    try:
        await User.objects.create(name=name, email=email, filename=filename)
    except Exception:
        success = False
    else:
        success = True
        user_image = face_recognition.load_image_file(user_image_path)
        user_face_encoding = face_recognition.face_encodings(user_image)[0]
        user_encoding_path = f"encodings/{filename}.pkl"
        async with aiofiles.open(user_encoding_path, "wb") as file:
            await file.write(pickle.dumps(user_face_encoding))

    return {"success": success}


async def read_encoding(sem: asyncio.Semaphore, filename: str):
    async with sem:
        user_encoding_path = f"encodings/{filename}.pkl"
        async with aiofiles.open(user_encoding_path, "rb") as file:
            user_face_encoding = pickle.loads(await file.read())
        return user_face_encoding


@app.get("/recognition")
async def recognition(
    new_image_file: UploadFile = File(...),
):
    users = await User.objects.all()
    new_filename = str(uuid.uuid4())
    new_image_path = f"images/{new_filename}.jpg"
    async with aiofiles.open(new_image_path, "wb+") as file:
        await file.write(new_image_file.file.read())
    new_image = face_recognition.load_image_file(new_image_path)
    sem = asyncio.Semaphore(4)
    users_face_encodings = await asyncio.gather(
        *[read_encoding(sem, user.filename) for user in users]
    )
    new_face_encoding = face_recognition.face_encodings(new_image)[0]
    results = face_recognition.compare_faces(users_face_encodings, new_face_encoding)

    try:
        index = results.index(True)
    except ValueError:
        return {"success": False}
    else:
        user = users[index]
        return {"success": True, "user": user}


@app.on_event("startup")
async def startup():
    await aiofiles.os.makedirs("images/", exist_ok=True)
    await aiofiles.os.makedirs("encodings/", exist_ok=True)