"""Crop original Chen et al. arXiv:1604.06174v2 PDF; no redraw."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, type=Path)
    parser.add_argument('--confirmed-visual-review', action='store_true',
                        help='Only set after these exact crop images have been visually inspected.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    (root/'图片').mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(args.pdf))
    crops = [
        ('原论文_Figure1_计算图与存储复用.png', 3, (108, 75, 505, 203)),
        ('原论文_Algorithm1_分段重算.png', 5, (106, 74, 506, 319)),
        ('原论文_公式1_内存权衡.png', 5, (165, 631, 507, 660)),
        ('原论文_Figure3_重算计算图.png', 6, (145, 75, 458, 220)),
    ]
    images = []
    for name, page, rect in crops:
        scale = 4
        full = doc[page-1].render(scale=scale).to_pil()
        image = full.crop(tuple(round(value*scale) for value in rect))
        path = root/'图片'/name
        image.save(path)
        images.append(dict(file=path.relative_to(root).as_posix(), page_1based=page,
                           crop_points_top_left=list(rect), render_scale=scale, pixel_size=list(image.size),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           visually_inspected=args.confirmed_visual_review))
    record = dict(paper='Training Deep Nets with Sublinear Memory Cost', version='arXiv:1604.06174v2',
                  pdf_url='https://arxiv.org/pdf/1604.06174v2', local_pdf=str(args.pdf.resolve()),
                  pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(), pdf_pages=len(doc),
                  command=[sys.executable]+sys.argv, cwd=str(Path.cwd()),
                  method='Original PDF pages rendered and cropped; no redraw, translation overlay, or content edits.',
                  pypdfium2=importlib.metadata.version('pypdfium2'), images=images,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'复现/outputs/论文截图来源.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Saved four original PDF crops; visually inspect each before delivery.')


if __name__ == '__main__':
    main()
