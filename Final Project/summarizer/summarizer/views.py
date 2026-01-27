import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .pipeline import run_pipeline


def upload_audio(request):
    summaries = []
    error = None

    if request.method == "POST":
        audio_file = request.FILES.get("audio")

        if not audio_file:
            error = "No audio file uploaded."
        else:
            fs = FileSystemStorage(location="media/")
            filename = fs.save(audio_file.name, audio_file)
            file_path = fs.path(filename)

            try:
                summaries = run_pipeline(file_path)
                if not summaries:
                    summaries = [("INFO", "No sufficient speech detected to summarize.")]
            except Exception as e:
                error = str(e)
            finally:
                # ✅ auto-delete uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)

    return render(
        request,
        "summarizer/upload.html",
        {
            "summaries": summaries,
            "error": error,
        }
    )