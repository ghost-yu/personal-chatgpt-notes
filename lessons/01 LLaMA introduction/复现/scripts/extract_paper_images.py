"""Render and crop original LLaMA v1 PDF content; never redraw the figure."""
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
    pdf_bytes = args.pdf.read_bytes()
    assert pdf_bytes.startswith(b'%PDF')
    pdf = pdfium.PdfDocument(str(args.pdf))
    assert len(pdf) == 27, 'Expected the 27-page arXiv v1 PDF'
    scale = 3
    page_image = pdf[2].render(scale=scale).to_pil()
    out = root / '图片'
    out.mkdir(exist_ok=True)
    images = []
    for name, rect in [
        ('LLaMA原论文_Table2_模型规格.png', (112, 67, 483, 154)),
        ('LLaMA原论文_Figure1_训练曲线.png', (302, 191, 526, 345)),
    ]:
        dest = out / name
        crop = page_image.crop(tuple(round(v * scale) for v in rect))
        crop.save(dest)
        images.append({'file': dest.relative_to(root).as_posix(), 'page_1based': 3,
                       'crop_points_top_left': list(rect), 'render_scale': scale,
                       'pixel_size': list(crop.size),
                       'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                       'method': 'Original PDF rendered and cropped; no redraw or data edits'})
    report = {'paper': 'LLaMA: Open and Efficient Foundation Language Models',
              'version': 'arXiv:2302.13971v1',
              'pdf_url': 'https://arxiv.org/pdf/2302.13971v1',
              'pdf_sha256': hashlib.sha256(pdf_bytes).hexdigest(),
              'local_pdf': str(args.pdf.resolve()), 'images': images,
              'pypdfium2': importlib.metadata.version('pypdfium2'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / '复现/outputs/论文截图来源.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved two original PDF crops; visual inspection is still required.')


if __name__ == '__main__':
    main()
