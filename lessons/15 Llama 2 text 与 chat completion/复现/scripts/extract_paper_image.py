"""Crop the original Llama 2 Figure 9; keep source and visual-review evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--confirmed-visual-review', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = root/'图片/Llama2原论文_Figure9_多轮要求.png'
    target.parent.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(args.pdf))
    rect = (96, 68, 516, 304)
    scale = 4
    crop = doc[15].render(scale=scale).to_pil().crop(tuple(int(x*scale) for x in rect))
    crop.save(target)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(method='Original PDF rendered and cropped, no redraw or relabel.',
        paper='Llama 2: Open Foundation and Fine-Tuned Chat Models', version='arXiv:2307.09288v2',
        pdf_url='https://arxiv.org/pdf/2307.09288v2', pdf_path=str(args.pdf.resolve()),
        pdf_sha256=sha(args.pdf), pdf_pages=len(doc), page_1based=16, figure='Figure 9',
        image=target.relative_to(root).as_posix(), image_sha256=sha(target), pixels=list(crop.size),
        crop_points_top_left=list(rect), scale=scale, visually_inspected=args.confirmed_visual_review,
        course_images=[dict(file=p.relative_to(root).as_posix(), sha256=sha(p))
                       for p in sorted((root/'代码/imgs').glob('*.png'))],
        pypdfium2=importlib.metadata.version('pypdfium2'), command=[sys.executable]+sys.argv,
        script_sha256=sha(Path(__file__)), cwd=str(Path.cwd()))
    (root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Figure 9 cropped. Inspect the PNG before marking visual review complete.')


if __name__ == '__main__':
    main()
