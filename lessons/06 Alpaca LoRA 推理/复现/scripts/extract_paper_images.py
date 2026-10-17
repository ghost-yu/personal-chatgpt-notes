"""Direct crops from the original LoRA v2 PDF; no redraw."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    document = pdfium.PdfDocument(str(args.pdf))
    assert len(document) == 26
    scale = 4
    images = []
    (root / '图片').mkdir(exist_ok=True)
    for name, page_number, rect in [
        ('LoRA原论文_Figure1_双分支.png', 1, (386, 519, 507, 635)),
    ]:
        image = document[page_number-1].render(scale=scale).to_pil()
        crop = image.crop(tuple(round(x * scale) for x in rect))
        path = root / '图片' / name
        crop.save(path)
        images.append({'file': path.relative_to(root).as_posix(), 'page_1based': page_number,
                       'crop_points_top_left': list(rect), 'render_scale': scale,
                       'pixel_size': list(crop.size),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'method': 'Original PDF page rendered then cropped; no redraw or data edits'})
    record = {'paper': 'LoRA: Low-Rank Adaptation of Large Language Models',
              'version': 'arXiv:2106.09685v2', 'pdf_url': 'https://arxiv.org/pdf/2106.09685v2',
              'pdf_sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
              'local_pdf': str(args.pdf.resolve()), 'images': images,
              'pypdfium2': importlib.metadata.version('pypdfium2'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / '复现/outputs/论文截图来源.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved original Figure 1; inspect the crop visually.')


if __name__ == '__main__':
    main()
