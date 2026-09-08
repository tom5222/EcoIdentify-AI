// EcoIdentify AI - Client-Side Controller Engine

let videoStream = null;
let autoScanInterval = null;
let isAutoScanning = false;
let isClassifying = false;
let currentMode = 'webcam';
let uploadedFile = null;

// Initialize WebRTC Camera Stream on Page Load
document.addEventListener("DOMContentLoaded", () => {
    initWebcam();
    checkHealth();
});

// Health check to verify AI backend
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        const statusText = document.getElementById('statusText');
        if (data.model_loaded) {
            statusText.innerText = "AI Model Online";
            statusText.parentElement.classList.remove('offline');
        } else {
            statusText.innerText = "AI Model Offline";
            statusText.parentElement.classList.add('offline');
        }
    } catch (err) {
        console.warn("Backend health check failed:", err);
    }
}

// Initialize Webcam Feed
async function initWebcam() {
    const video = document.getElementById('videoFeed');
    const fallback = document.getElementById('cameraFallback');
    const scanner = document.getElementById('scannerOverlay');

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("WebRTC getUserMedia is not supported in this browser.");
        }

        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = videoStream;
        await video.play();

        fallback.style.display = 'none';
        scanner.style.display = 'flex';
    } catch (err) {
        console.error("Camera access error:", err);
        fallback.style.display = 'flex';
        scanner.style.display = 'none';
    }
}

// Stop Webcam Stream
function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    const video = document.getElementById('videoFeed');
    if (video) video.srcObject = null;
}

// Switch between Webcam and File Upload Modes
function switchMode(mode) {
    currentMode = mode;
    
    // Update Tab UI
    document.getElementById('tabWebcam').classList.toggle('active', mode === 'webcam');
    document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');

    // Update Views
    document.getElementById('webcamView').classList.toggle('active', mode === 'webcam');
    document.getElementById('uploadView').classList.toggle('active', mode === 'upload');

    if (mode === 'webcam') {
        if (!videoStream) initWebcam();
    } else {
        // Stop auto-scan if switching away from webcam
        if (isAutoScanning) {
            document.getElementById('autoScanToggle').checked = false;
            toggleAutoScan(false);
        }
    }
}

// Capture frame from video element to hidden canvas
function captureFrameBase64() {
    const video = document.getElementById('videoFeed');
    const canvas = document.getElementById('hiddenCanvas');
    
    if (!video || video.readyState !== 4) return null;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.85);
}

// Capture and Classify Button Event
async function captureAndClassify() {
    if (isClassifying) return;
    
    const base64Image = captureFrameBase64();
    if (!base64Image) {
        alert("Camera feed not ready. Please allow camera access.");
        return;
    }

    await sendPredictionRequest({ image: base64Image });
}

// Continuous Auto-Scan Toggle
function toggleAutoScan(enabled) {
    isAutoScanning = enabled;
    const pulse = document.getElementById('autoScanPulse');
    pulse.classList.toggle('active', enabled);

    if (enabled) {
        // Immediate scan then interval
        captureAndClassify();
        autoScanInterval = setInterval(() => {
            if (currentMode === 'webcam' && isAutoScanning) {
                captureAndClassify();
            }
        }, 1500); // Scan every 1.5 seconds
    } else {
        if (autoScanInterval) {
            clearInterval(autoScanInterval);
            autoScanInterval = null;
        }
    }
}

// Send prediction payload to backend API
async function sendPredictionRequest(payload, isFormData = false) {
    isClassifying = true;

    try {
        let response;
        if (isFormData) {
            response = await fetch('/api/predict-file', {
                method: 'POST',
                body: payload
            });
        } else {
            response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        }

        const data = await response.json();

        if (response.ok && data.success) {
            renderResults(data);
        } else {
            console.error("Prediction API returned error:", data);
        }
    } catch (err) {
        console.error("Network or API error:", err);
    } finally {
        isClassifying = false;
    }
}

// Render Prediction Results into Dashboard UI
function renderResults(data) {
    // 1. Update Top-1 Banner Card
    document.getElementById('predictionEmoji').innerText = data.emoji;
    document.getElementById('predictionTitle').innerText = data.predicted_class;
    document.getElementById('confidencePill').innerText = data.confidence_percentage;
    document.getElementById('predictionSubtext').innerText = `Classified as ${data.predicted_class} with high AI confidence.`;

    // 2. Update Class Probabilities Bars
    const probs = data.probabilities;
    for (const [clsName, probVal] of Object.entries(probs)) {
        const item = document.querySelector(`.progress-item[data-class="${clsName}"]`);
        if (item) {
            const percentStr = `${(probVal * 100).toFixed(1)}%`;
            item.querySelector('.val').innerText = percentStr;
            item.querySelector('.progress-fill').style.width = percentStr;
        }
    }

    // 3. Update Recycling Guidance Card
    const guidance = data.guidance;
    if (guidance) {
        const binBadge = document.getElementById('binBadge');
        binBadge.innerText = guidance.bin_color;
        binBadge.style.backgroundColor = `${guidance.bin_badge_color}25`; // translucent
        binBadge.style.borderColor = guidance.bin_badge_color;
        binBadge.style.color = guidance.bin_badge_color;

        const stepsContainer = document.getElementById('guidanceSteps');
        stepsContainer.innerHTML = guidance.steps.map((step, idx) => `
            <div class="step-item">
                <span class="step-num">${idx + 1}</span>
                <span>${step}</span>
            </div>
        `).join('');

        if (guidance.eco_tip) {
            const ecoBox = document.getElementById('ecoTipBox');
            document.getElementById('ecoTipText').innerText = guidance.eco_tip;
            ecoBox.style.display = 'flex';
        }
    }
}

/* --- Drag & Drop File Upload Logic --- */

function triggerFileInput() {
    document.getElementById('fileInput').click();
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropzone').classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
        processUploadedFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
        processUploadedFile(files[0]);
    }
}

function processUploadedFile(file) {
    if (!file.type.match('image.*')) {
        alert("Please upload a valid image file (JPG, PNG, WEBP).");
        return;
    }

    uploadedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('dropzonePrompt').style.display = 'none';
        document.getElementById('previewContainer').style.display = 'block';
        document.getElementById('btnUploadClassify').disabled = false;
        
        // Instant classification upon drop/upload for slick UX
        classifyUploadedImage();
    };
    reader.readAsDataURL(file);
}

function clearImagePreview(e) {
    if (e) e.stopPropagation();
    uploadedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('imagePreview').src = '';
    document.getElementById('dropzonePrompt').style.display = 'flex';
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('btnUploadClassify').disabled = true;
}

async function classifyUploadedImage() {
    if (!uploadedFile || isClassifying) return;

    const formData = new FormData();
    formData.append('file', uploadedFile);

    await sendPredictionRequest(formData, true);
}
