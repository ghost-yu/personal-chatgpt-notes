"""Render and crop original PDF figures, without redraw or content edits."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructgpt-pdf', required=True, type=Path)
    parser.add_argument('--lora-pdf', required=True, type=Path)
    parser.add_argument('--confirmed-visual-review', action='store_true',
                        help='Set only after a human/agent has viewed these exact crops; never automatic.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    (root/'图片').mkdir(exist_ok=True)
    papers = []
    specs = [
        (args.instructgpt_pdf, 'Training language models to follow instructions with human feedback',
         '2203.02155v1', [('InstructGPT原论文_Figure2_三阶段.png', 3, (108, 70, 508, 300))]),
        (args.lora_pdf, 'LoRA: Low-Rank Adaptation of Large Language Models',
         '2106.09685v2', [('LoRA原论文_Figure1_双分支.png', 1, (386, 519, 507, 635))]),
    ]
    for pdf_path, title, version, crops in specs:
        doc = pdfium.PdfDocument(str(pdf_path))
        images = []
        for name, page, rect in crops:
            scale = 4
            rendered = doc[page-1].render(scale=scale).to_pil()
            cropped = rendered.crop(tuple(round(value*scale) for value in rect))
            path = root/'图片'/name
            cropped.save(path)
            images.append(dict(file=path.relative_to(root).as_posix(), page_1based=page,
                               crop_points_top_left=list(rect), render_scale=scale,
                               pixel_size=list(cropped.size), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               visually_inspected=args.confirmed_visual_review))
        papers.append(dict(paper=title, version='arXiv:'+version, pdf_url='https://arxiv.org/pdf/'+version,
                           local_pdf=str(pdf_path.resolve()), pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                           pdf_pages=len(doc), images=images))
    record = dict(method='Original PDF page rendered then cropped. No redraw, relabel, or content edits.',
                  command=[sys.executable]+sys.argv, cwd=str(Path.cwd()),
                  scope='Supplementary concept figures; course notebook is not an InstructGPT experiment.',
                  papers=papers, pypdfium2=importlib.metadata.version('pypdfium2'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'复现/outputs/论文截图来源.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Saved two original PDF crops; inspect both before delivery.')


if __name__ == '__main__':
    main()
