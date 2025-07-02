# EasyOCR와 OpenCV를 활용한 OCR 실습 코드

from easyocr import Reader  # EasyOCR에서 Reader 클래스 import. OCR의 핵심 엔진
import argparse  # 명령줄 인자를 처리
import cv2  # 이미지 불러오기/처리/출력을 위한 OpenCV 모듈
import torch

print(f"[DEBUG] CUDA 사용 가능 여부: {torch.cuda.is_available()}")


# 텍스트를 깨끗하게 정리하는 함수 (ASCII 문자만 남기고 나머지는 제거)
def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


ap = argparse.ArgumentParser()  # 명령줄 인자 설정
ap.add_argument(
    "-i", "--image", required=True, help="path to input image to be OCR'd"
)  # --image: 입력 이미지 경로 (입력 필수)
ap.add_argument(
    "-l",
    "--langs",
    type=str,
    default="en",
    help="comma separated list of languages to OCR",
)  # --langs: 사용할 언어 설정 (예: 'en', 'ko', 'en,ko') - 기본값은 영어
ap.add_argument(
    "-g", "--gpu", type=int, default=-1, help="whether or not GPU should be used"
)  # --gpu: GPU 사용 여부 (1이면 사용, -1이면 사용 안 함)
args = vars(ap.parse_args())  # 인자 파싱 결과를 딕셔너리로 저장

# break the input languages into a comma separated list
langs = args["langs"].split(
    ","
)  # 언어 코드를 리스트로 변환 (예: 'en,ko' → ['en', 'ko'])
print(
    "[INFO] OCR'ing with the following languages: {}".format(langs)
)  # 어떤 언어로 OCR을 수행할지 출력


# load the input image from disk
image = cv2.imread(args["image"])  # 이미지 파일을 OpenCV로 읽기
# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)  # gpu=True이면 GPU 사용, False이면 CPU 사용
results = reader.readtext(image)  # 이미지에서 텍스트 읽기 (bounding box, text, 신뢰도)

# 결과 반복 출력 및 이미지에 시각화
# loop over the results
for bbox, text, prob in results:
    print("[INFO] {:.4F}: {}".format(prob, text))  # 인식된 텍스트와 신뢰도 출력

    # bbox: 사각형 꼭짓점 좌표 4개 (top-left, top-right, bottom-right, bottom-left)
    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))  # 좌측 상단
    tr = (int(tr[0]), int(tr[1]))  # 우측 상단
    br = (int(br[0]), int(br[1]))  # 우측 하단
    bl = (int(bl[0]), int(bl[1]))  # 좌측 하단

    text = cleanup_text(text)  # 텍스트 정리 (특수문자 제거 등)
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)  # 박스 그리기: 초록색 사각형
    cv2.putText(
        image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )  # 텍스트 쓰기: 박스 위쪽에 글자 출력

# 결과 이미지 창에 띄우기
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)  # 아무 키나 누를 때까지 대기
