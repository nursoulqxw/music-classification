#Python modules
import tempfile
import os
import sys
from pathlib import Path
import uvicorn

#FastAPI modules
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


sys.path.append(str(Path(__file__).parent / "src"))

from predict import predict_song, le

app = FastAPI(title="Music Genre Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav", ".flac", ".aac", ".ogg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload MP3, WAV, FLAC, AAC, or OGG.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        genre, probabilities = predict_song(tmp_path)
        prob_dict = {
            le.classes_[i]: float(probabilities[i]) for i in range(len(le.classes_))
        }
        return {
            "filename": file.filename,
            "predicted_genre": genre,
            "confidence": float(max(probabilities)),
            "all_probabilities": prob_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/genres")
async def get_available_genres():
    return {"genres": list(le.classes_)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6767)
