"""Direct crops of original Llama 2 v2 and LoRA v2 PDF pages; no redraw."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama2-pdf', type=Path, required=True)
    parser.add_argument('--lora-pdf', type=Path, required=True)
    parser.add_argument('--confirmed-visual-review', action='store_true',
                        help='Set only after visually inspecting these exact crop images.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    (root/'图片').mkdir(exist_ok=True)
    specs = [
        (args.llama2_pdf, 'Llama 2: Open Foundation and Fine-Tuned Chat Models', '2307.09288v2', [
            ('Llama2原论文_Figure4_训练阶段.png', 5, (82, 70, 531, 277)),
            ('Llama2原论文_Table1_模型配置.png', 6, (82, 71, 530, 202)),
        ]),
        (args.lora_pdf, 'LoRA: Low-Rank Adaptation of Large Language Models', '2106.09685v2', [
            ('LoRA原论文_Figure1_双分支.png', 1, (386, 519, 507, 635)),
        ]),
    ]
    papers = []
    for pdf_path, title, version, crops in specs:
        doc = pdfium.PdfDocument(str(pdf_path))
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
        papers.append(dict(paper=title, version='arXiv:'+version, pdf_url='https://arxiv.org/pdf/'+version,
                           local_pdf=str(pdf_path.resolve()), pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                           pdf_pages=len(doc), images=images))
    record = dict(method='Original PDF pages rendered and cropped. No redraw, relabel, or content edits.',
                  papers=papers, command=[sys.executable]+sys.argv, cwd=str(Path.cwd()),
                  pypdfium2=importlib.metadata.version('pypdfium2'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved three original PDF crops; inspect each visually before delivery.')


if __name__ == '__main__':
    main()
