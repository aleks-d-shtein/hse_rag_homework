"""PDF -> data/pages.jsonl: одна строка на страницу {doc_id, page, text}.

Из корня репозитория: python -m rag.parse

Текст собирается по строкам из координат слов, так таблица остаётся таблицей:
подпись строки и её числа попадают в одну строку через « | ».
Страницы без текстового слоя (сканы) распознаются через macOS Vision.
"""

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "pages.jsonl"


def lines_from_boxes(boxes, page_height):
    """boxes: (x0, y0, x1, y1, text) в координатах страницы, y вниз."""
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows, tol = [], page_height * 0.006
    for box in boxes:
        cy = (box[1] + box[3]) / 2
        if rows and abs(rows[-1][0] - cy) <= tol:
            rows[-1][1].append(box)
        else:
            rows.append([cy, [box]])
    out = []
    for _, row in rows:
        row.sort(key=lambda b: b[0])
        line = row[0][4]
        for prev, box in zip(row, row[1:]):
            gap = box[0] - prev[2]
            line += (" | " if gap > page_height * 0.006 else " ") + box[4]
        out.append(line.strip())
    return "\n".join(out)


def text_layer(page):
    boxes = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    boxes.append((*span["bbox"], span["text"].strip()))
    return lines_from_boxes(boxes, page.rect.height)


def ocr(page):
    import Quartz
    import Vision
    from Foundation import NSData

    png = page.get_pixmap(dpi=200).tobytes("png")
    src = Quartz.CGImageSourceCreateWithData(NSData.dataWithBytes_length_(png, len(png)), None)
    image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLanguages_(["ru-RU", "en-US"])
    request.setRecognitionLevel_(0)  # accurate
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    handler.performRequests_error_([request], None)
    boxes = []
    for obs in request.results():
        b = obs.boundingBox()  # нормированные координаты, y вверх
        y1 = 1 - b.origin.y
        boxes.append((b.origin.x, y1 - b.size.height, b.origin.x + b.size.width, y1,
                      obs.topCandidates_(1)[0].string()))
    return lines_from_boxes(boxes, 1.0)


def main():
    docs = [json.loads(line) for line in (DATA / "documents.jsonl").open(encoding="utf-8")]
    with OUT.open("w", encoding="utf-8") as out:
        for doc in docs:
            pdf = pymupdf.open(DATA / "reports" / doc["path"])
            scanned = 0
            for number, page in enumerate(pdf, start=1):
                text = text_layer(page)
                if len(text) < 100:
                    text, scanned = ocr(page), scanned + 1
                out.write(json.dumps({"doc_id": doc["doc_id"], "page": number, "text": text},
                                     ensure_ascii=False) + "\n")
            print(doc["doc_id"], len(pdf), "страниц, OCR:", scanned, flush=True)


if __name__ == "__main__":
    main()
