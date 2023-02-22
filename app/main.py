import asyncio
import pickle
import uuid

import aiofiles
import aiofiles.os
import face_recognition
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import User, get_db

app = FastAPI(title="Face Recognition based attendance system")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    users = await User.objects.all()
    return templates.TemplateResponse(
        "index.html", {"request": request, "users": users}
    )


@app.get("/delete/{id:int}", response_class=RedirectResponse)
async def delete(id: int):
    user = await User.objects.delete(id=id)
    return "/"


@app.get("/update/{id:int}", response_class=HTMLResponse)
async def update_get(request: Request, id: int):
    user = await User.objects.get(id=id)
    return templates.TemplateResponse("update.html", {"request": request, "user": user})


@app.post("/update/{id:int}", response_class=RedirectResponse)
async def update_post(
    request: Request,
    id: int,
    name: str = Form(...),
    email: str = Form(...),
):
    user = await User.objects.get(id=id)
    user.name = name
    user.email = email
    return "/"


@app.get("/reg", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/reg")
async def register_post(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    image: UploadFile = Form(...),
):
    res = await register(name, email, image)
    res = ["User Already Exists", "Registeration Successfull"][res["success"]]
    return templates.TemplateResponse("register.html", {"request": request, "res": res})


@app.get("/recog", response_class=HTMLResponse)
async def recognize_get(request: Request):
    return templates.TemplateResponse("recognition.html", {"request": request})


@app.post("/recog")
async def recog_post(
    request: Request,
    image: UploadFile = Form(...),
):
    res = await recognition(image)
    return templates.TemplateResponse(
        "recognition.html", {"request": request, "res": res}
    )


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
