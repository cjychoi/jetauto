import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"

RTSP_URL = "rtsp://smi:samsung123@172.28.161.132:88/videoMain"
YOLO_MODEL = "yolo11x.pt"
FACE_DB_PATH = "static/face_db.json"
FACE_RECOGNITION_THRESHOLD = 0.3
FACE_MIN_SIZE_RT = (50, 50)
FACE_DEVICE_ID = 2  # -1 for CPU

FACE_MIN_SIZE_RT = (30, 30)

# VLLM_URL = "http://localhost:8000/v1/chat/completions"
# VLLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct-FP8"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "/models/Qwen3-VL-4B-Instruct-FP8"
VLLM_DEBUG_SAVE = False
LLM_DEBUG_SAVE = False

# VLLM_URL = "http://105.165.31.103:8000/v1/chat/completions"
# VLLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# LLM_URL = "http://105.165.31.69:8000/v1/chat/completions"
# LLM_MODEL = "MiniMaxAI/MiniMax-M2.5"

LLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "/models/Qwen3-VL-4B-Instruct-FP8"

EMBEDDING_HOST = '0.0.0.0'
EMBEDDING_PORT = 7200
EMBEDDING_URL = f"http://localhost:{EMBEDDING_PORT}"

FLASK_SECRET_KEY = 'sentry-secret-key'
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 6001

GATE_LOG_ENABLED = True
GATE_LOG_REMOTE_HOST = 'y3.zhang@105.165.109.65'
GATE_LOG_REMOTE_PATH = '/home/y3.zhang/.openclaw/workspace/smi-gate-log/gate.log'
