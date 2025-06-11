// Global variables
let stream = null;
let isDetecting = false;
let webcamElement,
  canvas,
  displayCanvas,
  behaviorsList,
  startButton,
  stopButton,
  realtimeContainer,
  cameraSelect;

let isRecording = false;
let videoStream = null;
let mediaRecorder = null;
let recordedChunks = [];

// Add tracking variables for behaviors
let sessionBehaviors = {
    detectedObjects: {},
    detectedBehaviors: {},
    timestamp: null,
    frames: 0
};

let currentBehaviors = {};

// Initialize DOM elements when the page loads
function initializeElements() {
  webcamElement = document.getElementById("webcam");
  canvas = document.getElementById("output-canvas");
  displayCanvas = document.getElementById("display-canvas");
  behaviorsList = document.getElementById("realtime-behaviors");
  startButton = document.getElementById("startButton");
  stopButton = document.getElementById("stopButton");
  realtimeContainer = document.getElementById("realtime-container");
  cameraSelect = document.getElementById("cameraSelect");

  // Add event listeners
  if (startButton) {
    startButton.addEventListener('click', startRealtime);
    // Ensure the button is visible initially
    startButton.style.display = 'inline-flex';
  }
  if (stopButton) {
    stopButton.addEventListener('click', stopRealtime);
    // Ensure the button is hidden initially
    stopButton.style.display = 'none';
  }

  // Initialize camera devices
  initializeCameraDevices();

  // Reset session behaviors
  resetSessionBehaviors();
}

function resetSessionBehaviors() {
    sessionBehaviors = {
        detectedObjects: {
            'laptop': 0,
            'mobile phone': 0
        },
        detectedBehaviors: {},
        timestamp: new Date().toISOString(),
        frames: 0
    };
    currentBehaviors = {};
}

// Get available camera devices and populate the select dropdown
async function initializeCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === "videoinput");

    // Clear and populate the camera select dropdown
    cameraSelect.innerHTML = "";

    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = '<option value="">No cameras found</option>';
      startButton.disabled = true;
      return;
    }

    videoDevices.forEach((device) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${cameraSelect.length + 1}`;
      cameraSelect.appendChild(option);
    });

    // Enable the start button if we have cameras
    startButton.disabled = false;

    // If no labels are available, request camera permission to get labels
    if (!videoDevices[0].label) {
      await navigator.mediaDevices.getUserMedia({ video: true });
      // After getting permission, re-enumerate devices to get labels
      initializeCameraDevices();
    }
  } catch (error) {
    console.error("Error getting camera devices:", error);
    cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
    startButton.disabled = true;
  }
}

// Initialize elements immediately if document is ready
initializeElements();

// Also initialize when DOM is fully loaded to ensure all elements are available
document.addEventListener("DOMContentLoaded", initializeElements);

async function startRealtime() {
    try {
        // Reset session behaviors when starting new session
        resetSessionBehaviors();
        
        // Hide the processed image section
        const processedImageSection = document.getElementById('processed-section');
        if (processedImageSection) {
            processedImageSection.style.display = 'none';
        }
        
        // Check browser support
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Your browser doesn't support webcam access. Please use Chrome, Firefox, or Edge.");
        }

        // Get selected camera device
        const selectedDeviceId = cameraSelect.value;
        if (!selectedDeviceId) {
            throw new Error("Please select a camera device.");
        }

        // Stop any existing streams
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        // Request webcam access with selected device
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        // Set up video element
        webcamElement.srcObject = stream;
        webcamElement.style.display = "block";

        try {
            await webcamElement.play();
        } catch (playError) {
            throw new Error("Failed to start video playback: " + playError.message);
        }

        // Show container and update UI
        realtimeContainer.style.display = "block";
        startButton.style.display = "none";
        stopButton.style.display = "inline-flex";

        // Set canvas sizes
        const videoWidth = webcamElement.videoWidth || 640;
        const videoHeight = webcamElement.videoHeight || 480;
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        displayCanvas.width = videoWidth;
        displayCanvas.height = videoHeight;

        // Start detection
        isDetecting = true;
        startDetection();

    } catch (error) {
        console.error("Error in startRealtime:", error);
        alert(error.message || "Failed to start realtime session");
    }
}

window.stopRealtime = function () {
  console.log("stopRealtime called");
  isDetecting = false;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  if (webcamElement) {
    webcamElement.srcObject = null;
    webcamElement.style.display = "none";
  }

  startButton.style.display = "inline-flex";
  stopButton.style.display = "none";
  realtimeContainer.style.display = "none";
  behaviorsList.innerHTML = "";

  // Clear the canvas
  const displayCtx = displayCanvas.getContext("2d");
  displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
};

async function startDetection() {
  const displayCtx = displayCanvas.getContext("2d");

  while (isDetecting) {
    try {
      // Ensure video is playing and ready
      if (webcamElement.readyState === webcamElement.HAVE_ENOUGH_DATA) {
        // Draw current video frame to canvas
        canvas.getContext("2d").drawImage(webcamElement, 0, 0);
        
        // Get the frame data
        const imageData = canvas.toDataURL("image/jpeg", 0.8);
        
        // Send frame to server for processing
        const response = await fetch("/process_frame", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ frame: imageData }),
        });
        
        if (response.ok) {
          const result = await response.json();
          updateBehaviors(result.behaviors);
          
          // Draw the processed frame with detections
          if (result.frame_data) {
            const img = new Image();
            img.onload = () => {
              displayCtx.drawImage(img, 0, 0);
            };
            img.src = "data:image/jpeg;base64," + result.frame_data;
          }
        }
      }

      // Wait for next frame
      await new Promise((resolve) => setTimeout(resolve, 200));
    } catch (error) {
      console.error("Error processing frame:", error);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}

function updateBehaviors(behaviors) {
  behaviorsList.innerHTML = "";
  
  // Create Objects section
  const objectsHeader = document.createElement("h5");
  objectsHeader.className = "glow-text";
  objectsHeader.textContent = "Detected Objects";
  behaviorsList.appendChild(objectsHeader);
  
  const objectsList = document.createElement("ul");
  objectsList.className = "behavior-list";
  
  // Add detected objects
  ['laptop', 'mobile phone'].forEach(obj => {
    if (behaviors[obj] && behaviors[obj] > 0) {
      const li = document.createElement("li");
      li.className = "behavior-item";
      li.innerHTML = `
        <span class="behavior-name">${obj}</span>
        <span class="badge neon-badge">${behaviors[obj]}</span>
      `;
      objectsList.appendChild(li);
    }
  });
  behaviorsList.appendChild(objectsList);
  
  // Create Behaviors section
  const behaviorsHeader = document.createElement("h5");
  behaviorsHeader.className = "glow-text mt-4";
  behaviorsHeader.textContent = "Detected Behaviors";
  behaviorsList.appendChild(behaviorsHeader);
  
  const behaviorList = document.createElement("ul");
  behaviorList.className = "behavior-list";
  
  // Add detected behaviors
  Object.entries(behaviors).forEach(([behavior, count]) => {
    if (!['laptop', 'mobile phone'].includes(behavior) && count > 0) {
      const li = document.createElement("li");
      li.className = "behavior-item";
      li.innerHTML = `
        <span class="behavior-name">${behavior}</span>
        <span class="badge neon-badge">${count}</span>
      `;
      behaviorList.appendChild(li);
    }
  });
  behaviorsList.appendChild(behaviorList);
}

// Log that the script has loaded and functions are exposed
console.log("Realtime.js loaded and functions exposed to window");

async function stopRealtime() {
    try {
        isDetecting = false;
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        if (webcamElement) {
            webcamElement.srcObject = null;
            webcamElement.style.display = "none";
        }

        // Get the exact data from the real-time display
        const realtimeData = {
            detectedObjects: {},
            detectedBehaviors: {}
        };

        // Get objects from real-time display
        const objectsList = document.querySelector('#realtime-behaviors .behavior-list');
        if (objectsList) {
            objectsList.querySelectorAll('.behavior-item').forEach(item => {
                const name = item.querySelector('.behavior-name').textContent;
                const count = parseInt(item.querySelector('.badge').textContent);
                if (['laptop', 'mobile phone'].includes(name)) {
                    realtimeData.detectedObjects[name] = count;
                }
            });
        }

        // Get behaviors from real-time display
        const behaviorsList = document.querySelectorAll('#realtime-behaviors .behavior-list')[1];
        if (behaviorsList) {
            behaviorsList.querySelectorAll('.behavior-item').forEach(item => {
                const name = item.querySelector('.behavior-name').textContent;
                const count = parseInt(item.querySelector('.badge').textContent);
                if (!['laptop', 'mobile phone'].includes(name)) {
                    realtimeData.detectedBehaviors[name] = count;
                }
            });
        }

        // Send final data to server
        const response = await fetch('/stop_realtime', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sessionEnd: true,
                sessionData: {
                    ...realtimeData,
                    timestamp: new Date().toISOString(),
                    frames: sessionBehaviors.frames,
                    startTime: sessionBehaviors.timestamp
                }
            })
        });

        const data = await response.json();
        
        if (data.status === 'success') {
            // Create download section
            const behaviorList = document.getElementById('realtime-behaviors');
            if (behaviorList) {
                // Create a section for downloads
                const downloadSection = document.createElement('div');
                downloadSection.className = 'mt-4 p-3 border rounded bg-dark';
                
                // Add a header
                const header = document.createElement('h5');
                header.className = 'glow-text mb-3';
                header.innerHTML = '<i class="fas fa-download"></i> Download Detection Results';
                downloadSection.appendChild(header);

                // Add Excel download button
                const excelBtn = document.createElement('a');
                excelBtn.href = data.excel_url;
                excelBtn.className = 'btn btn-success neon-btn w-100';
                excelBtn.innerHTML = '<i class="fas fa-file-excel"></i> Download Excel Report';
                
                // Set up direct download
                excelBtn.onclick = async (e) => {
                    e.preventDefault();
                    try {
                        const downloadResponse = await fetch(data.excel_url);
                        if (!downloadResponse.ok) throw new Error('Download failed');
                        
                        const blob = await downloadResponse.blob();
                        const url = window.URL.createObjectURL(blob);
                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                        
                        const downloadLink = document.createElement('a');
                        downloadLink.href = url;
                        downloadLink.download = `realtime_detection_${timestamp}.xlsx`;
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                        window.URL.revokeObjectURL(url);

                        // Show success message
                        const successMsg = document.createElement('div');
                        successMsg.className = 'alert alert-success mt-3';
                        successMsg.innerHTML = '<i class="fas fa-check-circle"></i> Excel report downloaded successfully!';
                        downloadSection.appendChild(successMsg);
                    } catch (error) {
                        console.error('Download error:', error);
                        const errorMsg = document.createElement('div');
                        errorMsg.className = 'alert alert-danger mt-3';
                        errorMsg.innerHTML = '<i class="fas fa-exclamation-circle"></i> Failed to download Excel file. Please try again.';
                        downloadSection.appendChild(errorMsg);
                    }
                };
                
                downloadSection.appendChild(excelBtn);
                behaviorList.appendChild(downloadSection);
            }
        }
    } catch (error) {
        console.error('Error stopping realtime:', error);
        alert('Error stopping detection. Please try again.');
    } finally {
        // Reset UI
        if (startButton && stopButton) {
            startButton.style.display = 'inline-flex';
            stopButton.style.display = 'none';
            startButton.disabled = false;
        }
        
        // Keep the container visible to show final results
        if (realtimeContainer) {
            realtimeContainer.style.display = 'block';
        }
    }
}

// Add this function to track frames during detection
function updateSessionBehaviors(behaviors) {
    sessionBehaviors.frames++;
    for (const [behavior, count] of Object.entries(behaviors)) {
        if (count > 0) {
            if (['laptop', 'mobile phone'].includes(behavior)) {
                sessionBehaviors.detectedObjects[behavior] = count;
            } else {
                sessionBehaviors.detectedBehaviors[behavior] = count;
            }
        }
    }
}
