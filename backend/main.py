#Python modules
import tempfile
import os
import sys
from pathlib import Path
import uvicorn

#FastAPI modules
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from prediction.predict import predict_song, le

CNN_MODEL_PATH = ROOT_DIR / "models" / "cnn_model.pth"
CNN_AVAILABLE = CNN_MODEL_PATH.exists()
_predict_song_cnn = None


def get_predict_song_cnn():
    global _predict_song_cnn
    if _predict_song_cnn is None:
        from prediction.predict_cnn import predict_song_cnn

        _predict_song_cnn = predict_song_cnn
    return _predict_song_cnn

app = FastAPI(title="Music Genre Classifier API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Tells the frontend which models are ready to use
@app.get("/models")
async def available_models():
    return {"ml": True, "cnn": CNN_AVAILABLE}


@app.post("/predict")
async def predict_genre(
    file: UploadFile = File(...),
    model_type: str = Form("ml"),
):
    if not file.filename.lower().endswith((".mp3", ".wav", ".flac", ".aac", ".ogg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload MP3, WAV, FLAC, AAC, or OGG.",
        )

    if model_type == "cnn" and not CNN_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CNN model is unavailable. Install torch/torchaudio and place cnn_model.pth in models/.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if model_type == "cnn":
            predict_song_cnn = get_predict_song_cnn()
            genre, prob_dict = predict_song_cnn(tmp_path)
            confidence = max(prob_dict.values())
        else:
            genre, probabilities = predict_song(tmp_path)
            prob_dict = {
                le.classes_[i]: float(probabilities[i]) for i in range(len(le.classes_))
            }
            confidence = float(max(probabilities))

        return {
            "filename": file.filename,
            "predicted_genre": genre,
            "confidence": confidence,
            "all_probabilities": prob_dict,
            "model_used": model_type,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
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
