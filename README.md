# Transformer Architecture & Explainer 🚀

An interactive, visual learning platform and exploration tool for understanding **Transformer-based Large Language Models (LLMs)** like GPT-2. 

This project combines a robust **Python (FastAPI + PyTorch)** backend with a rich, interactive web frontend to inspect internal attention mechanisms, token embeddings, feed-forward layers, and token generation in real time.

---

## 🌟 Key Features

* **Interactive Multi-Head Attention:** Visualize attention weights across all 12 layers and 12 attention heads with dynamic Sankey flows and attention heatmaps.
* **Live Tokenization & Embedding Inspection:** Break down any input prompt into tokens, token embeddings, and positional encodings.
* **QKV (Query-Key-Value) Projections:** Step through dot-product self-attention, causal masking, and softmax calculations.
* **MLP & Layer Normalization:** Explore intermediate feed-forward network computations and residual stream additions.
* **Probability & Sampling Controls:** Adjust temperature, top-$k$, and top-$p$ (nucleus) sampling dynamically with live mathematical breakdowns.
* **Python Backend Integration:** Full FastAPI backend providing local inference, GPT-2 model checkpoints, and real-time computation APIs.

---

## 🏗️ Architecture

```
Transformer-Architecture/
├── backend/
│   └── main.py              # FastAPI backend serving inference APIs & static UI
├── frontend/
│   ├── index.html           # Interactive Transformer Explainer UI
│   ├── _app/                # Compiled interactive components & visualizers
│   ├── model-v2/            # GPT-2 ONNX model chunks for client-side acceleration
│   └── article_assets/      # Visual diagrams & mathematical explanations
├── official/                # SvelteKit/Vite source code & development pipeline
├── requirements.txt         # Python backend dependencies
├── run.bat                  # One-click Windows startup script
└── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites

* **Python 3.9+** installed on your system
* **Git** installed

### Quick Start (Windows)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GauravHandge/Transformer-Architecture-.git
   cd Transformer-Architecture-
   ```

2. **Launch with one click:**
   ```cmd
   run.bat
   ```
   *This automatically installs dependencies from `requirements.txt` and starts the FastAPI server.*

3. **Open in your browser:**
   ```
   http://127.0.0.1:8080
   ```

---

### Manual Setup (Cross-Platform)

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend server:**
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080
   ```

4. **Access the Application:**
   * **Web UI:** [http://127.0.0.1:8080](http://127.0.0.1:8080)
   * **API Docs (Swagger UI):** [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)
   * **API Health Check:** [http://127.0.0.1:8080/api/health](http://127.0.0.1:8080/api/health)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check endpoint returning backend and model status |
| `GET` | `/api/meta` | Returns model hyperparameters (layers, heads, dimensions, vocab size) |
| `POST` | `/api/infer` | Runs inference on an input prompt and returns tokens, attentions, hidden states, and next-token probabilities |

---

## 👤 Author & Maintainer

* **Gaurav Handge**
  * GitHub: [@GauravHandge](https://github.com/GauravHandge)
  * Repository: [Transformer-Architecture-](https://github.com/GauravHandge/Transformer-Architecture-)
  * Email: [gauravhandge770903@gmail.com](mailto:gauravhandge770903@gmail.com)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
