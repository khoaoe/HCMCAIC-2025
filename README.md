# HCMAI2025_Baseline

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## 🧑‍💻 Getting Started

### Prerequisites
- Docker
- Docker Compose
- Python 3.10
- uv

### Download the dataset
1. [Embedding data and keys](https://www.kaggle.com/datasets/anhnguynnhtinh/embedding-data)
2. [Keyframes](https://www.kaggle.com/datasets/anhnguynnhtinh/aic-keyframe-batch-one)


Convert the global2imgpath.json to this following format(id2index.json)
```json
{
  "0": "1/1/0",
  "1": "1/1/16",
  "2": "1/1/49",
  "3": "1/1/169",
  "4": "1/1/428",
  "5": "1/1/447",
  "6": "1/1/466",
  "7": "1/1/467",
}
```
to do this:
```bash
cd migration
python id2index_converter.py
```


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

4. Data Migration 
```bash
# Migrate embeddings with temporal support
python migration/embedding_migration.py --file_path <embedding.pt file> --id2index_path <id2index.json file path>
python migration/keyframe_migration.py --file_path <id2index.json file path>
```

### 🕐 Temporal Search Features

The system now includes advanced temporal search capabilities:

#### Temporal Search Endpoints
- `/api/v1/temporal/search/time-range` - Search within specific time ranges
- `/api/v1/temporal/search/video-time-window` - Search within video time windows  
- `/api/v1/temporal/search/temporal-stats` - Get temporal search statistics

#### Key Features
- **Native timestamp filtering** using Milvus scalar fields
- **Video-specific time windows** for precise moment retrieval
- **Cross-corpus temporal search** across multiple videos
- **Efficient temporal indexing** for fast time-based queries
- **Combined semantic and temporal relevance** scoring

#### Example Usage
```bash
# Search for "person walking" between 30-90 seconds in video L01/V001
curl -X POST "http://localhost:8000/api/v1/temporal/search/time-range" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person walking",
    "start_time": 30.0,
    "end_time": 90.0,
    "video_id": "L01/V001",
    "top_k": 10,
    "score_threshold": 0.3
  }'

# Search within a specific time window using GET
curl "http://localhost:8000/api/v1/temporal/search/video-time-window?query=sunset&video_id=L02/V003&start_time=120.5&end_time=185.2"
```

5. Set API keys/tokens
```bash
setx HF_TOKEN=<huggingface_token>
setx HUGGING_FACE_HUB_TOKEN=<same_token>
setx GEMINI_API_KEY=<google_gemini_api_key>
```

6. Run the application

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