# EasyOCR와 OpenCV를 활용한 OCR 실습 코드

from easyocr import Reader  # EasyOCR에서 Reader 클래스 import. OCR의 핵심 엔진
import argparse  # 명령줄 인자를 처리
import cv2  # 이미지 불러오기/처리/출력을 위한 OpenCV 모듈
import torch

from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 맑은 고딕 폰트 불러오기
font_path = "C:/WINDOWS/FONTS/MALGUNSL.TTF"
font = ImageFont.truetype(font_path, 20)

# OCR 결과 후처리 교정 딕셔너리
corrections = {
    "테스트t": "테스트1",
    
}

print(f"[DEBUG] CUDA 사용 가능 여부: {torch.cuda.is_available()}")


# # 텍스트를 깨끗하게 정리하는 함수 (ASCII 문자만 남기고 나머지는 제거)
# def cleanup_text(text):
#     # strip out non-ASCII text so we can draw the text on the image
#     # using OpenCV
#     return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# 텍스트를 깨끗하게 정리하는 함수 (ASCII + 한글 허용)
def cleanup_text(text):
    cleaned = []
    for c in text:
        code = ord(c)
        if (
            code < 128  # ASCII 문자
            or 0xAC00 <= code <= 0xD7A3  # 한글 완성형 (가~힣)
            or 0x1100 <= code <= 0x11FF  # 한글 자모 (ㄱ~ㆎ)
            or 0x3130 <= code <= 0x318F  # 호환 자모 (ㄱ~ㅣ)
        ):
            cleaned.append(c)
    return "".join(cleaned).strip()


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

# 1. 흑백 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 2. 이진화 (Thresholding)
_, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # thresh = 150
# 3. EasyOCR 입력을 위한 컬러 복원 (3채널로 변환)
image = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

reader = Reader(langs, gpu=args["gpu"] > 0)  # gpu=True이면 GPU 사용, False이면 CPU 사용
results = reader.readtext(image)  # 이미지에서 텍스트 읽기 (bounding box, text, 신뢰도)

# OpenCV 이미지를 PIL로 변환
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)


corrected_results = []

# 결과 반복 출력 및 이미지에 시각화
# loop over the results
for bbox, text, prob in results:

    original_text = text
    text = cleanup_text(text)  # 텍스트 정리 (특수문자 제거 등)
    # cv2.rectangle(image, tl, br, (0, 255, 0), 2)  # 박스 그리기: 초록색 사각형
    # cv2.putText(
    #     image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    # )  # 텍스트 쓰기: 박스 위쪽에 글자 출력

    # 교정 딕셔너리에 해당하면 수정
    for wrong, right in corrections.items():
        if wrong in text:
            text = text.replace(wrong, right)

    print("[INFO] {:.4F}: {}".format(prob, original_text))  # 항상 출력
    if text != original_text:
        print("[INFO][교정] : {}".format(text))  # 다를 때만 출력

    # bbox: 사각형 꼭짓점 좌표 4개 (top-left, top-right, bottom-right, bottom-left)
    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))  # 좌측 상단
    tr = (int(tr[0]), int(tr[1]))  # 우측 상단
    br = (int(br[0]), int(br[1]))  # 우측 하단
    bl = (int(bl[0]), int(bl[1]))  # 좌측 하단

    draw.rectangle(
        [tl, br], outline=(0, 255, 0), width=2
    )  # 박스 그리기: 초록색 사각형(PIL 방식)
    # PIL 방식으로 텍스트 출력 (한글 지원)
    draw.text((tl[0], tl[1] - 25), text, font=font, fill=(0, 255, 0))

    # ✅ 교정된 결과 저장
    corrected_results.append((bbox, text, prob))

# 전체 정확도 평균 계산
if corrected_results:
    avg_confidence = sum([prob for _, _, prob in corrected_results]) / len(
        corrected_results
    )
    print(f"[INFO] 전체 OCR 정확도 평균: {avg_confidence:.4f}")
else:
    print("[INFO] 인식된 텍스트가 없습니다.")

# OCR 결과를 텍스트 파일로 저장
output_path = "threshold_150+후처리 버전.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for bbox, text, prob in corrected_results:
        original_text = cleanup_text(
            results[corrected_results.index((bbox, text, prob))][1]
        )  # 원래 텍스트 재추출
        if text != original_text:
            f.write(f"[교정] [{prob:.4f}] {text}\n")
        else:
            f.write(f"[{prob:.4f}] {text}\n")

    f.write(f"\n[평균 정확도] {avg_confidence:.4f}\n")


print(f"[INFO] 결과를 '{output_path}'에 저장했습니다.")

# PIL 이미지를 다시 OpenCV 이미지로 변환
image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 결과 이미지 창에 띄우기
# 창 크기 줄이기 (예: 최대 너비 n px로 제한)
MAX_WIDTH = 1200
if image.shape[1] > MAX_WIDTH:
    scale = MAX_WIDTH / image.shape[1]
    image = cv2.resize(
        image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
    )

cv2.imshow("Image", image)
cv2.waitKey(0)  # 아무 키나 누를 때까지 대기
