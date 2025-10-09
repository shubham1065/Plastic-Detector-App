---
title: Plastic Waste Detector
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
---

# ♻️ Plastic Waste Classification API

This application hosts a PyTorch model for classifying different types of plastic waste (PET, PE, PP, etc.) and serves the prediction results via an API, running on the Hugging Face Docker SDK.

The deployment uses a robust Docker environment to handle the large PyTorch model and dependencies.

### Application Details

* **Model:** RexNet-150 (Trained for 6 classes)
* **Backend:** FastAPI / Uvicorn (Running inside Docker)
* **Frontend:** Static HTML/JS (Deployed separately on Netlify)
* **Deployment:** Hugging Face Spaces (Docker SDK)

### Configuration

The application is configured to start via the `app.py` entry point, which initiates the Uvicorn server on the necessary port (`7860`).

The primary goal of this setup is to provide a memory-optimized environment capable of running the large PyTorch model.