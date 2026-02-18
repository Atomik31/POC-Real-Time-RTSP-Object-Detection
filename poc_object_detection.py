import av
import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
RTSP_URL = os.getenv("RTSP_CAM")
# 1. Load the model
# Note: yolo26m is a "Medium" model. If it stutters too much, switch to "yolo26n.pt"
model = YOLO("yolo11m.pt") 
# 2. Open the stream via PyAV (Fast engine)
container = av.open(RTSP_URL, options={
    'rtsp_transport': 'udp', 
    'fflags': 'nobuffer', 
    'flags': 'low_delay'
})
print("🚀 Real-time AI detection enabled...")
try:
    for frame in container.decode(video=0):
        # Convert PyAV -> NumPy (OpenCV compatible)
        img = frame.to_ndarray(format='bgr24')
        # 3. Run detection on the current image (NOT on the URL)
        # stream=True prevents RAM saturation
        # We disable show=True because we handle display ourselves
        results = model.predict(img, conf=0.3, verbose=False)
        # 4. Draw results on the image
        # plot() returns the image with detection boxes drawn
        annotated_frame = results[0].plot()
        # 5. Resize for visual comfort (16:9)
        display_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow('YOLO + PyAV Direct', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()
