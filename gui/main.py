import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import sys

# Add app to path to import settings
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from app.core.settings import AppSettings

from app.utils.common_utils import safe_convert_video_num

# Page configuration
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .mode-selector {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .score-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

# Load app settings
@st.cache_resource
def get_app_settings():
    return AppSettings()

app_settings = get_app_settings()

# Define fullscreen image dialog
@st.dialog("Fullscreen Image Viewer", width="large")
def show_fullscreen_image(image_path, caption):
    """Display image in fullscreen dialog"""
    try:
        st.image(image_path, use_container_width=True, caption=caption)
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        st.write(f"**Path:** {image_path}")

# Header
st.markdown("""
<div class="search-container">
    <h1 style="margin: 0; font-size: 2.5rem;">🔍 Keyframe Search</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Search through video keyframes using semantic similarity
    </p>
</div>
""", unsafe_allow_html=True)

# API Configuration
with st.expander("⚙️ API Configuration", expanded=False):
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL for the keyframe search API"
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url

# Main search interface
col1, col2 = st.columns([2, 1])

with col1:
    # Search query
    query = st.text_input(
        "🔍 Search Query",
        placeholder="Enter your search query (e.g., 'person walking in the park')",
        help="Enter 1-1000 characters describing what you're looking for"
    )
    
    # Search parameters
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        # Limit top_k to 100 for API compatibility
        top_k = st.slider("📊 Max Results", min_value=1, max_value=100, value=10)
    with col_param2:
        score_threshold = st.slider("🎯 Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    # Advanced Filters
    with st.expander("🔧 Bộ lọc Nâng cao (Advanced Filters)", expanded=False):
        
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            # Author filter
            filter_author = st.text_input(
                "👤 Author",
                placeholder="e.g., 'Báo Tuổi Trẻ'",
                help="Filter results by video author/creator"
            )
            
            # Keywords filter
            filter_keywords_input = st.text_input(
                "🏷️ Keywords",
                placeholder="e.g., 'tin tức, thời sự' (comma separated)",
                help="Filter results by specific keywords (comma separated)"
            )
        
        with col_filter2:
            # Date range filter
            st.markdown("📅 **Date Range**")
            use_date_filter = st.checkbox("Enable date filtering", value=False)
            
            if use_date_filter:
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input(
                        "From Date",
                        help="Start date for filtering"
                    )
                with col_date2:
                    end_date = st.date_input(
                        "To Date", 
                        help="End date for filtering"
                    )
            else:
                start_date = None
                end_date = None
        
        # Hybrid search options
        st.markdown("**🔀 Hybrid Search Options**")
        col_hybrid1, col_hybrid2 = st.columns(2)
        
        with col_hybrid1:
            use_hybrid_search = st.checkbox(
                "Enable Hybrid Search", 
                value=False,
                help="Combine visual similarity with metadata text search"
            )
        
        with col_hybrid2:
            metadata_weight = st.slider(
                "Metadata Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Weight for metadata results in hybrid search (0.0 = visual only, 1.0 = metadata only)"
            )
        
        # Parse keywords if provided
        filter_keywords = []
        if filter_keywords_input.strip():
            filter_keywords = [kw.strip() for kw in filter_keywords_input.split(',') if kw.strip()]
        
        # Format date for API if date filtering is enabled
        filter_publish_date = None
        if use_date_filter and start_date and end_date:
            filter_publish_date = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

with col2:
    # Search mode selector
    st.markdown("### 🎛️ Search Mode")
    search_mode = st.selectbox(
        "Mode",
        options=["Default", "Exclude Groups", "Include Groups & Videos", "Temporal Search", "Video Time Window", "GRAB Search"],
        help="Choose how to filter your search results"
    )

# Mode-specific parameters
if search_mode == "Exclude Groups":
    st.markdown("### 🚫 Exclude Groups")
    exclude_groups_input = st.text_input(
        "Group IDs to exclude",
        placeholder="Enter group IDs separated by commas (e.g., 1, 3, 7)",
        help="Keyframes from these groups will be excluded from results"
    )
    
    # Parse exclude groups
    exclude_groups = []
    if exclude_groups_input.strip():
        try:
            exclude_groups = [int(x.strip()) for x in exclude_groups_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")

elif search_mode == "Include Groups & Videos":
    st.markdown("### ✅ Include Groups & Videos")
    
    col_inc1, col_inc2 = st.columns(2)
    with col_inc1:
        include_groups_input = st.text_input(
            "Group IDs to include",
            placeholder="e.g., 2, 4, 6",
            help="Only search within these groups"
        )
    
    with col_inc2:
        include_videos_input = st.text_input(
            "Video IDs to include",
            placeholder="e.g., 101, 102, 203",
            help="Only search within these videos"
        )
    
    # Parse include groups and videos
    include_groups = []
    include_videos = []
    
    if include_groups_input.strip():
        try:
            include_groups = [int(x.strip()) for x in include_groups_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")
    
    if include_videos_input.strip():
        try:
            include_videos = [int(x.strip()) for x in include_videos_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid video IDs separated by commas")

elif search_mode == "Temporal Search":
    st.markdown("### ⏰ Temporal Search")
    
    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        start_time = st.number_input(
            "Start Time (seconds)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Start time for temporal filtering"
        )
    
    with col_temp2:
        end_time = st.number_input(
            "End Time (seconds)",
            min_value=0.1,
            value=60.0,
            step=0.1,
            help="End time for temporal filtering"
        )
    
    video_id_temporal = st.text_input(
        "Video ID (optional)",
        placeholder="L01/V001",
        help="Restrict search to specific video (format: Lxx/Vxxx)"
    )
    
    if end_time <= start_time:
        st.error("End time must be greater than start time")

elif search_mode == "Video Time Window":
    st.markdown("### 🎬 Video Time Window")
    
    video_id_window = st.text_input(
        "Video ID (required)",
        placeholder="L01/V001",
        help="Video to search within (format: Lxx/Vxxx)"
    )
    
    col_window1, col_window2 = st.columns(2)
    with col_window1:
        window_start_time = st.number_input(
            "Start Time (seconds)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Window start time"
        )
    
    with col_window2:
        window_end_time = st.number_input(
            "End Time (seconds)",
            min_value=0.1,
            value=30.0,
            step=0.1,
            help="Window end time"
        )
    
    if not video_id_window.strip():
        st.error("Video ID is required for video time window search")
    
    if window_end_time <= window_start_time:
        st.error("End time must be greater than start time")

elif search_mode == "GRAB Search":
    st.markdown("### 🚀 GRAB Framework Search")
    st.markdown("**GRAB**: Global Re-ranking and Adaptive Bidirectional search")
    
    col_grab1, col_grab2 = st.columns(2)
    with col_grab1:
        grab_start_time = st.number_input(
            "Start Time (seconds)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Start time for GRAB temporal search"
        )
        
        grab_video_id = st.text_input(
            "Video ID (optional)",
            placeholder="L01/V001",
            help="Target video for GRAB search (format: Lxx/Vxxx)"
        )
    
    with col_grab2:
        grab_end_time = st.number_input(
            "End Time (seconds)",
            min_value=0.1,
            value=60.0,
            step=0.1,
            help="End time for GRAB temporal search"
        )
        
        grab_optimization = st.selectbox(
            "Optimization Mode",
            options=["fast", "balanced", "precision"],
            index=1,  # Default to balanced
            help="GRAB optimization level: fast (speed), balanced (default), precision (accuracy)"
        )
    
    # GRAB Framework Features
    st.markdown("#### 🔧 GRAB Features")
    grab_features_info = {
        "fast": "Basic temporal search (λs=0.8, λt=0.2, 1s window)",
        "balanced": "Full GRAB pipeline (λs=0.7, λt=0.3, 2s window)",
        "precision": "Maximum accuracy (λs=0.6, λt=0.4, 3s window)"
    }
    
    st.info(f"**{grab_optimization.title()} Mode**: {grab_features_info[grab_optimization]}")
    
    col_features1, col_features2 = st.columns(2)
    with col_features1:
        st.markdown("**Enabled Features:**")
        if grab_optimization != "fast":
            st.markdown("- ✅ Shot Detection & Sampling")
            st.markdown("- ✅ Perceptual Hash Deduplication") 
            st.markdown("- ✅ SuperGlobal Reranking")
        if grab_optimization == "precision":
            st.markdown("- ✅ ABTS Boundary Detection")
            st.markdown("- ✅ Temporal Stability Analysis")
    
    with col_features2:
        st.markdown("**Performance:**")
        performance_info = {
            "fast": "⚡ Fastest (~1-2s)",
            "balanced": "⚖️ Balanced (~3-5s)", 
            "precision": "🎯 Highest Accuracy (~5-8s)"
        }
        st.markdown(f"- {performance_info[grab_optimization]}")
        st.markdown("- 📊 Detailed Analysis")
        st.markdown("- 🔄 Comparison Mode")
    
    if grab_end_time <= grab_start_time:
        st.error("End time must be greater than start time")

# Search button and logic
if st.button("🚀 Search", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query")
    elif len(query) > 1000:
        st.error("Query too long. Please keep it under 1000 characters.")
    else:
        with st.spinner("🔍 Searching for keyframes..."):
            try:
                if search_mode == "Default":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold
                    }
                    
                    # Add advanced filters if any are provided
                    if filter_author or filter_keywords or filter_publish_date or use_hybrid_search:
                        if filter_author:
                            payload["filter_author"] = filter_author
                        if filter_keywords:
                            payload["filter_keywords"] = filter_keywords
                        if filter_publish_date:
                            payload["filter_publish_date"] = filter_publish_date
                        if use_hybrid_search:
                            payload["use_hybrid_search"] = True
                            payload["metadata_weight"] = metadata_weight
                
                elif search_mode == "Exclude Groups":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "exclude_groups": exclude_groups
                    }
                    
                    # Add advanced filters if any are provided
                    if filter_author or filter_keywords or filter_publish_date or use_hybrid_search:
                        if filter_author:
                            payload["filter_author"] = filter_author
                        if filter_keywords:
                            payload["filter_keywords"] = filter_keywords
                        if filter_publish_date:
                            payload["filter_publish_date"] = filter_publish_date
                        if use_hybrid_search:
                            payload["use_hybrid_search"] = True
                            payload["metadata_weight"] = metadata_weight
                
                elif search_mode == "Temporal Search":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/temporal/search/time-range"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "start_time": start_time,
                        "end_time": end_time,
                        "video_id": video_id_temporal if video_id_temporal.strip() else None
                    }
                
                elif search_mode == "Video Time Window":
                    if not video_id_window.strip():
                        st.error("Video ID is required for video time window search")
                        st.stop()
                    
                    endpoint = f"{st.session_state.api_base_url}/api/v1/temporal/search/video-time-window"
                    # Use GET parameters for this endpoint
                    params = {
                        "query": query,
                        "video_id": video_id_window,
                        "start_time": window_start_time,
                        "end_time": window_end_time,
                        "top_k": top_k,
                        "score_threshold": score_threshold
                    }
                    
                    # For GET request, we'll use requests.get instead
                    response = requests.get(endpoint, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        # Normalize Windows-style paths
                        for item in results:
                            if isinstance(item, dict) and 'path' in item and isinstance(item['path'], str):
                                item['path'] = item['path'].replace('\\', '/')
                        st.session_state.search_results = results
                        st.success(f"✅ Found {len(st.session_state.search_results)} results in video {video_id_window} time window!")
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
                    # Skip the normal POST request processing for this mode
                    # st.stop()
                
                elif search_mode == "GRAB Search":
                    # Validate top_k for GRAB search (max 100)
                    if top_k > 100:
                        st.error("❌ GRAB Search Error: top_k cannot exceed 100. Please reduce the Max Results value.")
                        st.stop()
                    
                    endpoint = f"{st.session_state.api_base_url}/api/v1/temporal/search/enhanced-moments"
                    # Use GET parameters for GRAB search
                    params = {
                        "query": query,
                        "start_time": grab_start_time if grab_start_time > 0 else None,
                        "end_time": grab_end_time if grab_end_time > grab_start_time else None,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "optimization_level": grab_optimization
                    }
                    
                    # Only add video_id if it's provided
                    if grab_video_id and grab_video_id.strip():
                        params["video_id"] = grab_video_id.strip()
                        
                    # Remove None values
                    params = {k: v for k, v in params.items() if v is not None}
                    
                    # For GRAB search, we'll use requests.post with JSON for detailed response
                    response = requests.post(endpoint, params=params, timeout=60)  # Longer timeout for GRAB
                    
                    if response.status_code == 200:
                        data = response.json()
                        moments = data.get("moments", [])
                        
                        # Convert moments to results format for display
                        results = []
                        for moment in moments:
                            try:
                                
                                # Check if moment has the expected structure
                                if "video_id" not in moment:
                                    continue
                                
                                video_id = moment["video_id"]
                                
                                # Handle different video_id formats
                                if '/' in video_id:
                                    # Format: "L26/L26_V138" or similar
                                    parts = video_id.split('/')
                                    group_part = parts[0]
                                    video_part = parts[1] if len(parts) > 1 else parts[0]
                                    
                                    # Extract group number
                                    if group_part.startswith('L'):
                                        group_num = int(group_part[1:])
                                    else:
                                        group_num = int(group_part)
                                    
                                    # Extract video number
                                    if video_part.startswith('L'):
                                        video_num_raw = video_part[1:]  # Remove 'L' prefix
                                    elif video_part.startswith('V'):
                                        video_num_raw = video_part[1:]  # Remove 'V' prefix
                                    else:
                                        video_num_raw = video_part
                                    
                                    video_num = safe_convert_video_num(video_num_raw)
                                    
                                else:
                                    # Fallback: assume it's just a group number or path
                                    print(f"WARNING: Unexpected video_id format: {video_id}")
                                    continue
                                
                                
                                # Use first evidence keyframe for path, fallback to keyframe_range
                                evidence_keyframes = moment.get("evidence_keyframes", [])
                                if evidence_keyframes:
                                    # Extract keyframe number from file path
                                    # Format: 'C:\\...\\L21\\L21_V024\\011.jpg' -> 011
                                    kf_path = evidence_keyframes[0]
                                    if isinstance(kf_path, str) and '\\' in kf_path:
                                        # Extract filename without extension
                                        filename = os.path.basename(kf_path)
                                        kf_num = int(filename.split('.')[0])  # Remove .jpg extension
                                    else:
                                        kf_num = int(kf_path)  # If it's already a number
                                else:
                                    # Fallback: use keyframe_start from keyframe_range
                                    keyframe_range = moment.get("keyframe_range", {})
                                    kf_num = keyframe_range.get("start", 0)  # Use 0 as fallback
                                
                                # Ensure all values are integers
                                group_num = int(group_num)
                                video_num = int(video_num)
                                kf_num = int(kf_num)
                                
                                # Fix the path construction to avoid f-string backslash issue
                                keyframes_path = app_settings.KEYFRAMES_FOLDER.replace('\\', '/')
                                path = f"{keyframes_path}/L{group_num:02d}/L{group_num:02d}_V{video_num:03d}/{kf_num:03d}.jpg"
                                
                            except Exception as e:
                                print(f"ERROR processing moment: {e}")
                                print(f"Moment data: {moment}")
                                continue
                            
                            results.append({
                                "path": path,
                                "score": moment["confidence_score"],
                                "moment_info": {
                                    "start_time": moment["start_time"],
                                    "end_time": moment["end_time"],
                                    "duration": moment["duration"]
                                }
                            })
                        
                        st.session_state.search_results = results
                        st.success(f"🚀 GRAB Framework found {len(results)} temporal moments!")
                        
                        # Show GRAB-specific metrics
                        if data.get("optimization_level"):
                            st.info(f"**Optimization**: {data['optimization_level'].title()} mode applied")
                        
                    else:
                        st.error(f"❌ GRAB Search Error: {response.status_code} - {response.text}")
                    
                    # Skip the normal POST request processing for GRAB search
                    # st.stop()
                
                # Include Groups & Videos
                elif search_mode == "Include Groups & Videos":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "include_groups": include_groups,
                        "include_videos": include_videos
                    }
                    
                    # Add advanced filters if any are provided
                    if filter_author or filter_keywords or filter_publish_date or use_hybrid_search:
                        if filter_author:
                            payload["filter_author"] = filter_author
                        if filter_keywords:
                            payload["filter_keywords"] = filter_keywords
                        if filter_publish_date:
                            payload["filter_publish_date"] = filter_publish_date
                        if use_hybrid_search:
                            payload["use_hybrid_search"] = True
                            payload["metadata_weight"] = metadata_weight
                
                
                # POST request
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    # Normalize Windows-style paths to use forward slashes so Streamlit can load images
                    for item in results:
                        if isinstance(item, dict) and 'path' in item and isinstance(item['path'], str):
                            item['path'] = item['path'].replace('\\', '/')
                    st.session_state.search_results = results
                    st.success(f"✅ Found {len(st.session_state.search_results)} results!")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")
    
    # Show active filters if any were used
    active_filters = []
    if filter_author:
        active_filters.append(f"👤 Author: {filter_author}")
    if filter_keywords:
        active_filters.append(f"🏷️ Keywords: {', '.join(filter_keywords)}")
    if filter_publish_date:
        active_filters.append(f"📅 Date: {filter_publish_date}")
    if use_hybrid_search:
        active_filters.append(f"🔀 Hybrid Search (weight: {metadata_weight})")
    
    if active_filters:
        st.markdown("### 🔧 Active Filters")
        for filter_info in active_filters:
            st.markdown(f"- {filter_info}")
        st.markdown("---")
        
    # Results summary
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("Total Results", len(st.session_state.search_results))
    
    with col_metric2:
        avg_score = sum(result['score'] for result in st.session_state.search_results) / len(st.session_state.search_results)
        st.metric("Average Score", f"{avg_score:.3f}")
    
    with col_metric3:
        max_score = max(result['score'] for result in st.session_state.search_results)
        st.metric("Best Score", f"{max_score:.3f}")
    
    # Sort by score (highest first)
    sorted_results = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)
    
    # Display results in a grid
    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                # Try to display image with fullscreen functionality
                try:
                    # Display image with click to fullscreen
                    if st.button(f"📷", key=f"img_{i}", help="Click to view fullscreen"):
                        show_fullscreen_image(result['path'], f"Keyframe {i+1} - Score: {result['score']:.3f}")
                    
                    # Show thumbnail image
                    st.image(result['path'], width=200, caption=f"Keyframe {i+1}")
                    
                except Exception as e:
                    st.markdown(f"""
                    <div style="
                        background: #f0f0f0; 
                        height: 150px; 
                        width: 200px;
                        border-radius: 10px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        border: 2px dashed #ccc;
                        margin: 0 auto;
                    ">
                        <div style="text-align: center; color: #666;">
                            🖼️<br>Image Preview<br>Not Available<br>
                            <small style="font-size: 10px;">Path: {result['path']}</small>
                        </div>
                    </div>
                    <div style="margin-top: 5px; font-size: 12px; color: #666; text-align: center;">
                        Keyframe {i+1}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_info:
                # display for GRAB framework results
                moment_info = result.get('moment_info', {})
                
                if moment_info:
                    # GRAB framework result with temporal moment info
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; color: #333;">🎯 Temporal Moment #{i+1}</h4>
                            <span class="score-badge">Score: {result['score']:.3f}</span>
                        </div>
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; text-align: center;">
                                <div>
                                    <div style="font-size: 0.8rem; opacity: 0.8;">Start Time</div>
                                    <div style="font-weight: bold;">{moment_info['start_time']:.1f}s</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; opacity: 0.8;">Duration</div>
                                    <div style="font-weight: bold;">{moment_info['duration']:.1f}s</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; opacity: 0.8;">End Time</div>
                                    <div style="font-weight: bold;">{moment_info['end_time']:.1f}s</div>
                                </div>
                            </div>
                        </div>
                        <p style="margin: 0.5rem 0; color: #666;"><strong>Representative Frame:</strong> {result['path']}</p>
                        <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; font-family: monospace; font-size: 0.8rem;">
                            {result['path']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Traditional keyframe result
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; color: #333;">Result #{i+1}</h4>
                            <span class="score-badge">Score: {result['score']:.3f}</span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #666;"><strong>Path:</strong> {result['path']}</p>
                        <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; font-family: monospace; font-size: 0.9rem;">
                            {result['path']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)