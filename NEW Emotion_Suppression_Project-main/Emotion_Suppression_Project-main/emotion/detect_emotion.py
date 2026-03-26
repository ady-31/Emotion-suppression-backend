import cv2

def detect_emotions_from_video(video_path):
    try:
        from deepface import DeepFace
    except Exception:
        # If DeepFace/TensorFlow is not available, skip visible emotion extraction.
        return []

    cap = cv2.VideoCapture(video_path)

    emotions_over_time = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion_probs = result[0]["emotion"]
            emotions_over_time.append(emotion_probs)

        except Exception:
            continue

    cap.release()

    return emotions_over_time