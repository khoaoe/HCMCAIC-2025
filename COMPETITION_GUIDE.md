# HCMC AI Challenge 2025 - Competition System Guide

## Overview

This enhanced system implements a comprehensive solution for the HCMC AI Challenge 2025, supporting all three competition tracks with advanced multi-modal capabilities.

## 🏆 Competition Tasks Supported

### 1. Video Corpus Moment Retrieval (VCMR)

**Automatic Track**: Find ranked temporal segments across video corpus
```bash
POST /api/v1/competition/vcmr/automatic
```

**Input Example**:
```json
{
  "task": "vcMr_automatic",
  "query": "A woman places a framed picture on the wall",
  "corpus_index": "v1",
  "top_k": 100
}
```

**Output**: Ranked list of temporal moments with `start_time`, `end_time`, and relevance scores.

**Interactive Track**: Human-in-the-loop refinement
```bash
POST /api/v1/competition/vcmr/interactive
```

Supports binary, graded, and free-text feedback for iterative improvement.

### 2. Video Question Answering (VQA)

Answer questions about video content with evidence
```bash
POST /api/v1/competition/vqa
```

**Input Example**:
```json
{
  "task": "video_qa",
  "video_id": "L01/V001",
  "video_uri": "path/to/video.mp4",
  "question": "How many people are walking?",
  "clip": {"start_time": 10.0, "end_time": 30.0}
}
```

**Features**:
- Visual evidence from keyframes
- ASR context integration
- Confidence scoring
- Supporting timestamp evidence

### 3. Known-Item Search (KIS)

Locate exact target segments from descriptions or visual examples

**Textual KIS**:
```bash
POST /api/v1/competition/kis/textual
```

**Visual KIS**:
```bash
POST /api/v1/competition/kis/visual
```

**Progressive KIS** (with iterative hints):
```bash
POST /api/v1/competition/kis/progressive
```

## 🚀 Key Improvements

### 1. Temporal Localization
- **Keyframe-to-Time Conversion**: Accurate temporal mapping
- **Intelligent Clustering**: Groups keyframes into meaningful moments
- **Adaptive Windowing**: Context-aware temporal boundaries
- **FPS-Aware Processing**: Handles variable frame rates

### 2. Enhanced Multi-Modal Integration
- **ASR Alignment**: Speech-to-text temporal synchronization
- **Object Detection**: COCO-based visual filtering
- **Cross-Modal Reranking**: LLM-powered relevance scoring
- **Query Expansion**: Semantic variations for robust retrieval

### 3. Competition-Specific Optimizations
- **Task-Aware Prompting**: Specialized prompts for each competition task
- **Performance Optimization**: Speed/precision trade-offs for different tracks
- **Interactive Feedback**: Advanced feedback integration mechanisms
- **Evidence Tracking**: Comprehensive provenance for evaluation

### 4. Advanced Agent Capabilities
- **Query Understanding**: Deep analysis of user intent and requirements
- **Multi-Query Fusion**: Combines multiple search strategies
- **Intelligent Caching**: Performance optimization for repeated queries
- **Dynamic Parameter Tuning**: Adapts to query complexity and mode

## 📊 Architecture Improvements

### Core Components

1. **CompetitionTaskDispatcher**: Routes tasks to specialized handlers
2. **TemporalLocalizer**: Converts keyframes to temporal moments
3. **ASRTemporalAligner**: Synchronizes speech with visual content
4. **MultiModalRetriever**: Advanced cross-modal search and ranking
5. **PerformanceOptimizer**: Real-time optimization for competition constraints

### Enhanced Schemas

- Competition-compliant input/output formats
- Temporal mapping structures
- Interactive feedback models
- Evidence tracking schemas

## 🔧 Configuration

### Competition Mode Settings

**Automatic Track** (Precision Focus):
```python
{
    "top_k": 300,
    "score_threshold": 0.05,
    "enable_reranking": True,
    "temporal_clustering_gap": 5.0,
    "asr_weight": 0.3,
    "visual_weight": 0.7
}
```

**Interactive Track** (Speed Focus):
```python
{
    "top_k": 100,
    "score_threshold": 0.1,
    "enable_reranking": False,
    "temporal_clustering_gap": 3.0,
    "asr_weight": 0.4,
    "visual_weight": 0.6
}
```

### Data Requirements

1. **Video Metadata**: FPS, duration, frame counts
2. **ASR Data**: Temporal speech-to-text transcripts
3. **Object Detection**: COCO-labeled keyframes
4. **Embeddings**: Pre-computed visual and text embeddings

## 🎯 Performance Optimizations

### Speed Optimizations
- **Parallel Processing**: Concurrent keyframe analysis
- **Smart Caching**: Embedding and result caching
- **Dynamic Top-K**: Adaptive result limits based on complexity
- **Batch Operations**: Efficient database queries

### Precision Optimizations
- **Multi-Query Expansion**: Robust search coverage
- **Cross-Modal Fusion**: Combines visual, audio, and text signals
- **LLM Reranking**: Contextual relevance scoring
- **Temporal Clustering**: Meaningful moment boundaries

## 📈 Monitoring and Metrics

### Performance Tracking
- Response time monitoring
- Cache hit rates
- Confidence score distributions
- Success rate by task type

### Competition Compliance
- Output format validation
- Temporal boundary verification
- Score range enforcement
- Evidence completeness checks

## 🛠 Usage Examples

### VCMR Automatic
```python
# Find moments of people walking in park
request = {
    "task": "vcMr_automatic",
    "query": "people walking in a park during sunset",
    "corpus_index": "v1",
    "top_k": 50
}

response = await competition_controller.process_vcmr_automatic(request)
# Returns ranked temporal moments across entire corpus
```

### Video QA
```python
# Answer question about specific video clip
request = {
    "task": "video_qa",
    "video_id": "L01/V001",
    "question": "What color is the car?",
    "clip": {"start_time": 15.0, "end_time": 25.0}
}

response = await competition_controller.process_video_qa(request)
# Returns answer with visual evidence and confidence
```

### Interactive VCMR with Feedback
```python
# Initial search
candidate = await vcmr_agent.process_interactive_vcmr("woman driving car")

# User provides feedback
feedback = {"refine": "focus on red car in urban setting"}
refined_candidate = await vcmr_agent.process_interactive_vcmr(
    "woman driving car", 
    feedback=feedback
)
```

## 🔍 Advanced Features

### 1. Multi-Modal Query Understanding
- Extracts entities, actions, temporal cues
- Identifies visual attributes and context
- Optimizes search strategy per query type

### 2. Intelligent Temporal Clustering
- Groups related keyframes into coherent moments
- Handles sparse and dense keyframe distributions
- Adaptive temporal gap detection

### 3. Cross-Modal Evidence Fusion
- Combines visual, audio, and textual evidence
- Weighted scoring across modalities
- Contextual relevance assessment

### 4. Interactive Feedback Learning
- Binary, graded, and textual feedback support
- Query refinement strategies
- Session state management

## 🚦 Competition Compliance Checklist

✅ **Output Format Compliance**
- Temporal boundaries in seconds
- Proper JSON schema adherence
- Score normalization (0-1 range)

✅ **Task Implementation**
- VCMR Automatic & Interactive
- Video QA with evidence
- All KIS variants (T, V, C)

✅ **Performance Requirements**
- Real-time response for interactive track
- Top-K limits respected (≤100)
- Evidence and provenance tracking

✅ **Resource Utilization**
- Uses provided ASR and embeddings
- Leverages object detection metadata
- Supports any allowed pre-trained models

## 📝 Next Steps for Competition

1. **Data Preparation**:
   - Load competition video corpus
   - Prepare ASR transcripts with temporal alignment
   - Generate/load video metadata (FPS, duration)

2. **Model Configuration**:
   - Fine-tune embedding models if allowed
   - Optimize prompts for competition evaluation metrics
   - Calibrate confidence thresholds

3. **Performance Testing**:
   - Benchmark response times across tasks
   - Validate output format compliance
   - Test interactive feedback loops

4. **Competition Deployment**:
   - Configure for competition environment
   - Enable monitoring and logging
   - Prepare backup strategies

## 🔧 Development Commands

```bash
# Start development environment
cd app && python main.py

# Test competition endpoints
curl -X POST "http://localhost:8000/api/v1/competition/vcmr/automatic" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "corpus_index": "v1", "top_k": 10}'

# Run performance benchmarks
python -m app.utils.benchmark_competition_tasks

# Validate output formats
python -m app.utils.validate_competition_compliance
```

This enhanced system provides a robust, competition-ready solution that maximizes performance across all HCMC AI Challenge 2025 tracks while maintaining flexibility for real-time optimization and feedback integration.
