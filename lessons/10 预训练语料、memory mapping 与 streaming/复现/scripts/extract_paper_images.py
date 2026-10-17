"""Direct crops of original C4 documentation and fixed Pile v1 tables."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pypdfium2 as pdfium

parser = argparse.ArgumentParser()
parser.add_argument('--c4-pdf', required=True, type=Path)
parser.add_argument('--pile-pdf', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
entries = []
for path, title, version, url, pages, page, rect, name in [
    (args.c4_pdf, 'Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus',
     'EMNLP 2021, 2021.emnlp-main.98 proceedings PDF',
     'https://aclanthology.org/2021.emnlp-main.98.pdf',20,2,(304,69,535,182),
     'C4原论文_Table1_三个版本.png'),
    (args.pile_pdf, 'The Pile: An 800GB Dataset of Diverse Text for Language Modeling',
     'arXiv:2101.00027v1','https://arxiv.org/pdf/2101.00027v1',39,3,(71,61,530,475),
     'Pile原论文_Table1_语料组成.png')]:
    doc = pdfium.PdfDocument(str(path))
    assert len(doc) == pages
    destination = root/'图片'/name
    destination.parent.mkdir(exist_ok=True)
    crop = doc[page-1].render(scale=4).to_pil().crop(tuple(round(v*4) for v in rect))
    crop.save(destination)
    entries.append({'paper':title,'version':version,'pdf_url':url,
        'local_pdf':str(path.resolve()),'pdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'page_1based':page,'table':'Table 1','image':destination.relative_to(root).as_posix(),
        'image_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'crop_points_top_left':list(rect),'render_scale':4,'pixel_size':list(crop.size)})
record = {'method':'Original PDFs rendered and directly cropped; no redraw','images':entries,
          'pypdfium2':importlib.metadata.version('pypdfium2'),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved C4 Table 1 and Pile v1 Table 1; inspect visually.')
