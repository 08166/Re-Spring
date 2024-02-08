from fastapi import FastAPI, File, UploadFile, Form  # FastAPI 관련 필수 모듈 임포트
from fastapi.staticfiles import StaticFiles  # 정적 파일을 호스팅하기 위한 모듈
from fastapi.responses import JSONResponse, HTMLResponse  # 응답 유형 모듈
import cv2  # OpenCV 라이브러리
import os  # 운영체제와 상호작용하기 위한 모듈
import numpy as np  # 수치 계산을 위한 라이브러리
from insightface.app import FaceAnalysis  # 얼굴 분석을 위한 InsightFace 라이브러리
import uvicorn  # ASGI 서버
import glob
import traceback  # 오류 추적을 위한 모듈


app = FastAPI()  # FastAPI 애플리케이션 인스턴스 생성
faceapp = FaceAnalysis  # InsightFace 얼굴 분석 모델 인스턴스 생성

# 이미지를 저장할 디렉토리 설정
UPLOAD_DIRECTORY = "static/images"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)  # 디렉토리가 없으면 생성

# 정적 파일을 호스팅하기 위한 디렉토리 설정
app.mount("/static", StaticFiles(directory=f"{UPLOAD_DIRECTORY}"), name="static")

# 이미지 파일을 저장하는 함수
def save_image(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)  # 파일 경로 조합
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())  # 파일 쓰기
    return file_path  # 저장된 파일의 경로 반환

# 등록된 얼굴 정보를 저장하는 딕셔너리
registered_faces = {}

# InsightFace 얼굴 모델 초기화 클래스
class FaceModel:
    def __init__(self):
        self.faceapp = FaceAnalysis( det_name='retinaface_mnet025_v2', rec_name='arcface_r100_v1', ga_name=None, det_size=(640, 640), ctx_id=-1)
        self.faceapp.prepare(ctx_id=-1, det_size=(640, 480))

        # self.faceapp = FaceAnalysis(det_name="retinaface_r50_v1", rec_name="arcface_r100_v1")
        print("self.faceapp",self.faceapp)
        
    def get_embedding(self, face_image):
        # 얼굴 검출 및 임베딩 추출 함수
        try:
            # 이미지가 NumPy 배열인지 확인
            if isinstance(face_image, np.ndarray):
            # 이미지의 형태와 데이터 타입 출력
                print("이미지 형태:", face_image.shape)
                print("데이터 타입:", face_image.dtype)
            else:
                print("이미지는 NumPy 배열이 아닙니다.")
            faces = self.faceapp.get(face_image)
            if not faces:
                return None
            face_embedding = faces[0].normed_embedding  # 첫 번째 얼굴의 임베딩 추출
            return face_embedding
        except Exception as e:
            traceback.print_exc()  # 오류 추적
            return None

model = FaceModel()  # 모델 인스턴스 생성

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # 루트 경로로 접근했을 때 HTML 응답을 반환하는 엔드포인트
    with open("templates/main.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# 얼굴 등록을 위한 엔드포인트
@app.post("/register_face")
async def register_face(image: UploadFile = File(...), user_id: str = Form(...)):
        print(f"Received user_id: {user_id}")  # 사용자 ID 출력

        file_path = save_image(image)  # 이미지 저장
        print(f"Saved image to: {file_path}")  # 저장된 이미지 경로 출력
        
        face_image = cv2.imread(file_path)  # 이미지 읽기
        print(cv2.imshow("Image : ", face_image))
        face_embedding = model.get_embedding(face_image)  # 얼굴 임베딩 추출
        print("face_embedding : ", face_embedding)
        
        if face_embedding is not None:
            print("face_embedding : ", face_embedding)
            registered_faces[user_id] = face_embedding  # 얼굴 정보 등록
            return {"message": f"환영합니다. {user_id}"}
        else:
            return {"message": "얼굴 등록에 실패했습니다. 얼굴이 감지되지 않았습니다."}
    
# @app.post("/recognize_face")
# async def recognize_face(image: UploadFile = File(...)):
#     # 클라이언트로부터 받은 이미지 파일을 서버에 저장합니다.
#     file_path = save_image(image)

#     # 저장된 이미지 파일을 읽어서 OpenCV 형식의 이미지 데이터로 변환합니다.
#     unknown_face_image = cv2.imread(file_path)

#     # 변환된 이미지 데이터로부터 얼굴의 임베딩을 추출합니다.
#     unknown_face_embedding = model.get_embedding(unknown_face_image)

#     # 얼굴 임베딩 추출에 성공했다면, 등록된 사용자의 임베딩과 비교합니다.
#     if unknown_face_embedding is not None:
#         min_distance = float('inf')  # 최소 거리를 무한대로 초기화합니다.
#         recognized_user_id = None  # 인식된 사용자 ID를 None으로 초기화합니다.

#     # 등록된 모든 사용자의 얼굴 임베딩과 비교
#         for user_id, saved_embedding in registered_faces.items():
#             distance = np.linalg.norm(saved_embedding - unknown_face_embedding)
#             if distance < min_distance:
#                 min_distance = distance
#                 recognized_user_id = user_id

#         threshold = 0.6  # 거리의 임계값을 설정합니다.

#         # 최소 거리가 임계값보다 작다면, 사용자를 인식한 것으로 간주하고 환영 메시지를 반환합니다.
#         if min_distance < threshold:
#             return JSONResponse(content={"message": f"{recognized_user_id}님 환영합니다."})
#         else:
#             # 임계값보다 크다면, 등록되지 않은 사용자로 간주합니다.
#             return JSONResponse(content={"message": "등록되지 않은 회원입니다."})
#     else:
#         # 얼굴 임베딩 추출에 실패한 경우, 오류 메시지를 반환합니다.
#         return JSONResponse(content={"message": "얼굴 인식에 실패했습니다. 얼굴이 감지되지 않았습니다."})

@app.post("/recognize_face")
async def recognize_face(image: UploadFile = File(...)):
    # 파일을 메모리에 바로 로드
    image_data = await image.read()
    numpy_array = np.frombuffer(image_data, np.uint8)
    print("numpy_array", numpy_array)
    unknown_face_image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    print("unknown_face_image", unknown_face_image)

    # 변환된 이미지 데이터로부터 얼굴의 임베딩을 추출
    unknown_face_embedding = model.get_embedding(unknown_face_image)
    print("unknown_face_embedding", unknown_face_embedding)

    # 얼굴 임베딩 추출에 성공했다면, 등록된 사용자의 임베딩과 비교
    if unknown_face_embedding is not None:
        print("unknown_face_embedding", unknown_face_embedding)
        min_distance = float('inf')
        print("min_distance", min_distance)
        recognized_user_id = None
        print("recognized_user_id", recognized_user_id)

        # 서버에 저장된 모든 임베딩 파일(.npy)을 순회하며, 알 수 없는 얼굴 임베딩과의 거리를 계산
        for embedding_file in glob.glob(os.path.join(UPLOAD_DIRECTORY, "*.npy")):
            print("embedding_file", embedding_file)
            saved_face_embedding = np.load(embedding_file)
            print("saved_face_embedding", saved_face_embedding)
            distance = np.linalg.norm(saved_face_embedding - unknown_face_embedding)
            print("distance", distance)

            if distance < min_distance:
                min_distance = distance
                recognized_user_id = os.path.splitext(os.path.basename(embedding_file))[0]

        threshold = 0.6  # 임계값 설정

        if min_distance < threshold:
            return JSONResponse(content={"message": f"인식 성공: {recognized_user_id}"})
        else:
            return JSONResponse(content={"message": "등록되지 않은 사용자입니다."})
    else:
        return JSONResponse(content={"message": "얼굴 인식에 실패했습니다. 얼굴이 감지되지 않았습니다."})