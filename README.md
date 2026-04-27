# AI/ML Image Processing API (Vision & Background Removal)

A full-stack, Dockerized application that leverages open-source AI models to process images. It features a FastAPI backend, a clean web interface, and an automated multi-container environment that handles hardware acceleration and model initialization.

##  Features

* **Advanced Background Removal:** Utilizes the `briaai/RMBG-2.0` model to accurately segment subjects and return transparent PNGs.

* **Vision-Language Captioning:** Integrates with a local Ollama instance running the `moondream` model to generate 1-sentence descriptions of the uploaded image.

* **Interactive Web UI:** A built-in, responsive HTML/CSS/JS frontend to test the endpoints without needing Postman or cURL.

* **Fully Containerized:** Uses Docker Compose to orchestrate the API, the Ollama server, and an initialization script that automatically downloads necessary models on startup.

* **Hardware Optimized:** The Dockerfile is built on an NVIDIA PyTorch base image (`2.4.0-cuda11.8-cudnn9-runtime`) to support GPU inference where available, while gracefully falling back to CPU.

## Prerequisites

Before running this project, ensure you have the following:

1. **Docker Desktop** installed and running.

2. A **Hugging Face** account.

3. Minimum **10-15 GB** of free disk space for model weights.

## 🚀 Setup & Installation

### Step 1: Accept the Model License

The `RMBG-2.0` model is gated. You must accept its terms before downloading:

1. Go to [briaai/RMBG-2.0 on Hugging Face](https://huggingface.co/briaai/RMBG-2.0).

2. Log in and click **Agree / Accept** on the model card.

3. Go to your Hugging Face **Settings -> Access Tokens** and generate a new token (with **Read** permissions).

### Step 2: Configure Environment Variables

Create a file named `.env` in the root directory of this project (next to `docker-compose.yml`) and add your Hugging Face token:

HF_TOKEN=hf_your_generated_token_here


### Step 3: Build and Run

Open your terminal in the project directory and start the Docker containers:

docker-compose up --build


**Startup Sequence:**

1. Docker builds the API and installs dependencies.

2. The `ollama` server boots up.

3. The `ollama-init` container automatically pulls the `moondream` model and then exits.

4. The `api` container downloads the `RMBG-2.0` model during the FastAPI lifespan initialization.
   *(Note: Initial startup may take a few minutes as the ~1.7GB model downloads).*

## 💻 Usage

Once the terminal logs indicate `Uvicorn running on http://0.0.0.0:8000`, open your web browser and navigate to:

**http://localhost:8000**

From the UI, you can:

* Upload an image.

* Click **Remove Background** to test the standalone masking model.

* Click **Describe & Remove BG** to test the concurrent multi-model pipeline.

## 🔌 API Endpoints

* **`POST /remove-bg`**

  * Accepts: `multipart/form-data` (image file)

  * Returns: `image/png` (Raw image bytes of the transparent result)

* **`POST /process`**

  * Accepts: `multipart/form-data` (image file)

  * Returns: `application/json`

  * Example Response:

    ```
    {
      "description": "A brown dog sitting on a grassy field.",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
    ```
