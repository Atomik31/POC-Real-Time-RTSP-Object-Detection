import av
import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
RTSP_URL = os.getenv("RTSP_CAM")

# 1. Chargement du modèle
# Note : yolo26m est un modèle "Medium". Si ça saccade trop, passe en "yolo11n.pt"
model = YOLO("yolo26m.pt") 

# 2. Ouverture du flux via PyAV (Moteur rapide)
container = av.open(RTSP_URL, options={
    'rtsp_transport': 'udp', 
    'fflags': 'nobuffer', 
    'flags': 'low_delay'
})

print("🚀 Détection IA en direct activée...")

try:
    for frame in container.decode(video=0):
        # Conversion PyAV -> NumPy (OpenCV compatible)
        img = frame.to_ndarray(format='bgr24')

        # 3. Lancement de la détection sur l'image actuelle (PAS sur l'URL)
        # stream=True permet de ne pas saturer la RAM
        # On désactive show=True car on gère l'affichage nous-mêmes
        results = model.predict(img, conf=0.3, verbose=False)

        # 4. Dessiner les résultats sur l'image
        # plot() renvoie l'image avec les boîtes de détection dessinées
        annotated_frame = results[0].plot()

        # 5. Redimensionnement pour le confort visuel (16:9)
        display_frame = cv2.resize(annotated_frame, (1280, 720))

        cv2.imshow('YOLO + PyAV Direct', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Erreur : {e}")
finally:
    cv2.destroyAllWindows()