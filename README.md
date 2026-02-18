🛡️ Real-Time RTSP Object Detection
This project provides a low-latency solution for real-time object detection on RTSP video streams (like Dahua or Hikvision cameras) using YOLO and PyAV.

🚀 Key Features
Zero Latency: Uses PyAV to bypass standard OpenCV buffering for a true "live" feed.

AI Powered: Integrated with Ultralytics YOLO for state-of-the-art detection.

Optimized: Uses UDP transport and no-buffer flags to ensure the display stays at the present moment.

🛠️ Installation
Clone the repository:

```Bash
git clone <your-repo-url>
cd Object-detection
```

Install dependencies:

```Bash
pip install av opencv-python ultralytics python-dotenv
```
Set up your environment:
Create a .env file in the root folder and add your camera URL:

Extrait de code
```python
RTSP_CAM=rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1
```
⚙️ Configuration
You can easily customize the script by modifying these two lines in test.py:

1. Change the Video Stream
Update the .env file or modify the connection string. Using subtype=1 (Sub-stream) is highly recommended for better performance.

2. Change the AI Model
In the code, locate the model initialization:

```Python
# Change "yolo26m.pt" to "yolo11n.pt" for maximum speed
model = YOLO("yolo26m.pt")
```
- yolo26n.pt (Nano): Fastest, ideal for older CPUs or high FPS.

- yolo26m.pt (Medium): Good balance between accuracy and speed.

- yolo26l.pt (Large): High accuracy, requires a strong GPU.

💻 Usage
Run the detection script:

```Bash
python test.py
```
Press 'q' to quit the video window.
