import streamlit as st
import requests
import json
from typing import List, Optional
import pandas as pd
import os

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
    
    /* Clickable image styles */
    .clickable-image {
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-radius: 10px;
        border: 2px solid transparent;
    }
    
    .clickable-image:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border-color: #667eea;
    }
    
    /* Modal styles */
    .image-modal {
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.9);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .image-modal.active {
        display: block;
        opacity: 1;
    }
    
    .modal-content {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    .modal-image {
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        transition: transform 0.2s ease;
        cursor: grab;
    }
    
    .modal-image:active {
        cursor: grabbing;
    }
    
    .close-modal {
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10000;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 50%;
        transition: background 0.2s ease;
    }
    
    .close-modal:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .zoom-controls {
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 25px;
        color: white;
        font-size: 14px;
        z-index: 10000;
    }
    
    .zoom-info {
        position: absolute;
        top: 20px;
        left: 30px;
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 15px;
        border-radius: 15px;
        color: white;
        font-size: 14px;
        z-index: 10000;
    }
</style>

<script>
let currentZoom = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

function openImageModal(imageSrc, caption) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    
    modalImage.src = imageSrc;
    modalCaption.textContent = caption || '';
    modal.classList.add('active');
    
    // Reset zoom and position
    currentZoom = 1;
    translateX = 0;
    translateY = 0;
    updateImageTransform();
    updateZoomInfo();
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('active');
    
    // Restore body scroll
    document.body.style.overflow = 'auto';
}

function updateImageTransform() {
    const modalImage = document.getElementById('modalImage');
    modalImage.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentZoom})`;
}

function updateZoomInfo() {
    const zoomInfo = document.getElementById('zoomInfo');
    zoomInfo.textContent = `Zoom: ${Math.round(currentZoom * 100)}%`;
}

// Zoom with mouse wheel
function handleWheel(e) {
    e.preventDefault();
    
    const zoomSpeed = 0.1;
    const rect = e.target.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    const mouseX = e.clientX - centerX;
    const mouseY = e.clientY - centerY;
    
    const oldZoom = currentZoom;
    
    if (e.deltaY < 0) {
        // Zoom in
        currentZoom = Math.min(currentZoom + zoomSpeed, 5);
    } else {
        // Zoom out
        currentZoom = Math.max(currentZoom - zoomSpeed, 0.1);
    }
    
    // Adjust position to zoom towards mouse cursor
    const zoomRatio = currentZoom / oldZoom;
    translateX = translateX * zoomRatio + mouseX * (1 - zoomRatio);
    translateY = translateY * zoomRatio + mouseY * (1 - zoomRatio);
    
    updateImageTransform();
    updateZoomInfo();
}

// Pan with mouse drag
function handleMouseDown(e) {
    if (currentZoom > 1) {
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        e.target.style.cursor = 'grabbing';
    }
}

function handleMouseMove(e) {
    if (isDragging && currentZoom > 1) {
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateImageTransform();
    }
}

function handleMouseUp(e) {
    if (isDragging) {
        isDragging = false;
        e.target.style.cursor = 'grab';
    }
}

// Reset zoom on double click
function handleDoubleClick(e) {
    currentZoom = 1;
    translateX = 0;
    translateY = 0;
    updateImageTransform();
    updateZoomInfo();
}

// Keyboard shortcuts
function handleKeyDown(e) {
    if (document.getElementById('imageModal').classList.contains('active')) {
        switch(e.key) {
            case 'Escape':
                closeImageModal();
                break;
            case '+':
            case '=':
                e.preventDefault();
                currentZoom = Math.min(currentZoom + 0.2, 5);
                updateImageTransform();
                updateZoomInfo();
                break;
            case '-':
                e.preventDefault();
                currentZoom = Math.max(currentZoom - 0.2, 0.1);
                updateImageTransform();
                updateZoomInfo();
                break;
            case '0':
                e.preventDefault();
                currentZoom = 1;
                translateX = 0;
                translateY = 0;
                updateImageTransform();
                updateZoomInfo();
                break;
        }
    }
}

// Event listeners
document.addEventListener('keydown', handleKeyDown);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Create modal if it doesn't exist
    if (!document.getElementById('imageModal')) {
        const modalHTML = `
            <div id="imageModal" class="image-modal">
                <div class="modal-content">
                    <span class="close-modal" onclick="closeImageModal()">&times;</span>
                    <div class="zoom-info" id="zoomInfo">Zoom: 100%</div>
                    <img id="modalImage" class="modal-image" 
                         onwheel="handleWheel(event)"
                         onmousedown="handleMouseDown(event)"
                         onmousemove="handleMouseMove(event)"
                         onmouseup="handleMouseUp(event)"
                         ondblclick="handleDoubleClick(event)">
                    <div class="zoom-controls">
                        <div id="modalCaption"></div>
                        <div style="margin-top: 5px; font-size: 12px; opacity: 0.8;">
                            🔍 Scroll to zoom • 🖱️ Drag to pan • ⌨️ +/- to zoom • 0 to reset • Esc to close
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

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
        top_k = st.slider("📊 Max Results", min_value=1, max_value=200, value=10)
    with col_param2:
        score_threshold = st.slider("🎯 Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

with col2:
    # Search mode selector
    st.markdown("### 🎛️ Search Mode")
    search_mode = st.selectbox(
        "Mode",
        options=["Default", "Exclude Groups", "Include Groups & Videos"],
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
                
                elif search_mode == "Exclude Groups":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "exclude_groups": exclude_groups
                    }
                
                else:  # Include Groups & Videos
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "include_groups": include_groups,
                        "include_videos": include_videos
                    }
                

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

# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")
        
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
                # Try to display clickable image with zoom functionality
                try:
                    # Use HTML to create clickable image
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <img src="{result['path']}" 
                             class="clickable-image" 
                             style="width: 200px; max-height: 150px; object-fit: cover;"
                             onclick="openImageModal('{result['path']}', 'Keyframe {i+1} - Score: {result['score']:.3f}')"
                             alt="Keyframe {i+1}"
                             title="Click to view fullscreen with zoom">
                        <div style="margin-top: 5px; font-size: 12px; color: #666; text-align: center;">
                            Keyframe {i+1}<br>
                            <span style="font-size: 10px; opacity: 0.7;">Click to zoom</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except:
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
                            🖼️<br>Image Preview<br>Not Available
                        </div>
                    </div>
                    <div style="margin-top: 5px; font-size: 12px; color: #666; text-align: center;">
                        Keyframe {i+1}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_info:
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

<!-- Ensure modal is available -->
<div id="imageModal" class="image-modal">
    <div class="modal-content">
        <span class="close-modal" onclick="closeImageModal()">&times;</span>
        <div class="zoom-info" id="zoomInfo">Zoom: 100%</div>
        <img id="modalImage" class="modal-image" 
             onwheel="handleWheel(event)"
             onmousedown="handleMouseDown(event)"
             onmousemove="handleMouseMove(event)"
             onmouseup="handleMouseUp(event)"
             ondblclick="handleDoubleClick(event)">
        <div class="zoom-controls">
            <div id="modalCaption"></div>
            <div style="margin-top: 5px; font-size: 12px; opacity: 0.8;">
                🔍 Scroll to zoom • 🖱️ Drag to pan • ⌨️ +/- to zoom • 0 to reset • Esc to close
            </div>
        </div>
    </div>
</div>

<script>
// Ensure modal functionality is available
if (typeof currentZoom === 'undefined') {
    window.currentZoom = 1;
    window.isDragging = false;
    window.translateX = 0;
    window.translateY = 0;
}
</script>
""", unsafe_allow_html=True)