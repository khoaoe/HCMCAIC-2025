# HCMC AIC 2025 

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## 🧑‍💻 Getting Started

### Prerequisites
- Docker
- Docker Compose
- Python 3.10
- uv

### Prepare the data



### 🔧 Local Development
1. Clone the repo and start all services:
```bash
git clone https://github.com/yourusername/aio-aic.git
cd aio-aic
```

2. Install uv and setup env
```bash
pip install uv
uv init --python=3.10
uv add aiofiles beanie dotenv fastapi[standard] httpx ipykernel motor nicegui numpy open-clip-torch pydantic-settings pymilvus streamlit torch typing-extensions usearch uvicorn
```

3. Activate .venv
```bash
source .venv/bin/activate
```


4. Run docker compose
```bash
docker compose up -d
```


5. Data Migration 
```bash
# for .npy embeddings files 
python migration/npy_embedding_migration.py --folder_path resources\embeddings
# or for .pt files
...
# for keyframes indexing
python migration/keyframe_migration.py --file_path resources\keyframes\id2index.json
```


6. Set up .env file

Create .env file

```bash
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=<db_name>
MONGO_USER=<user_name>
MONGO_PASSWORD=<pass>
MONGO_URI=<uri_from_atlas>
HF_TOKEN=<hf_token>
HUGGING_FACE_HUB_TOKEN=<same_as_above>
GEMINI_API_KEY=<google_api_key>
```

7. Run the application

Open 2 tabs

6.1. Run the FastAPI application
```bash
cd gui
streamlit run main.py
```

6.2. Run the Streamlit application
```bash
cd app
python main.py
```