# EasyOCR 기반 텍스트 인식

EasyOCR와 OpenCV, Pillow(PIL)를 활용하여 이미지 속 텍스트를 인식하고 시각화하는 OCR 실습.

---

## 📌 주요 기능

- ✅ 이미지 전처리 (Grayscale + Thresholding)
- ✅ EasyOCR을 통한 텍스트 인식
- ✅ 교정 사전을 활용한 오탈자 후처리
- ✅ PIL을 활용한 한글 텍스트 및 사각형 시각화
- ✅ 결과를 이미지로 출력 및 `.txt` 파일로 저장

---

## 🛠️ 주요 라이브러리

- Python 3.8+
- EasyOCR
- OpenCV
- Pillow (PIL)

설치:
```bash
pip install easyocr opencv-python pillow
```

---

## 🖼️ 예시 흐름

1. `images/` 폴더에 이미지 파일을 넣고,
2. 아래 명령어로 실행합니다:

```bash
python easy_ocr.py -i images/sample.jpg -l ko,en -g -1
```

3. 결과는 화면 출력 + `output.txt`로 저장됩니다.

---

## 🧪 전처리 / 후처리

- **전처리**: 흐린 배경 제거 및 텍스트 대비 향상을 위해 thresholding 적용
- **후처리**: `BRCAI` → `BRCA1`, `보험고드` → `보험코드` 등 교정 사전 기반 자동 수정

---

## 📁 디렉토리 구조

```
.
├── easy_ocr.py           # 메인 실행 파일
├── images/               # 테스트 이미지 폴더
├── output.txt            # OCR 결과 저장 파일
└── README.md
```

---

## ✍️ 참고 사항

- PIL을 사용하는 이유는 OpenCV에서 한글 텍스트 출력이 불가능하기 때문입니다.
- 정확도 향상을 위해 후처리 교정 사전을 자유롭게 확장할 수 있습니다.

---

## 📬 개발자

- 전우진 (Joycong)  
- [GitHub 프로필](https://github.com/Joycong)
