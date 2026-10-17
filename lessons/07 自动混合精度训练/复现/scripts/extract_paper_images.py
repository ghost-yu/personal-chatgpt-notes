"""Direct crop of Mixed Precision Training v3 Figure 1; no redraw."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pypdfium2 as pdfium

parser = argparse.ArgumentParser()
parser.add_argument('--pdf', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
doc = pdfium.PdfDocument(str(args.pdf))
assert len(doc) == 12
scale = 4
rect = (168,79,446,224)
path = root/'图片/MixedPrecision原论文_Figure1_单层训练.png'
path.parent.mkdir(exist_ok=True)
crop = doc[2].render(scale=scale).to_pil().crop(tuple(round(v*scale) for v in rect))
crop.save(path)
record = {'paper':'Mixed Precision Training','version':'arXiv:1710.03740v3',
          'pdf_url':'https://arxiv.org/pdf/1710.03740v3','page_1based':3,'figure':'Figure 1',
          'local_pdf':str(args.pdf.resolve()),'pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
          'image':path.relative_to(root).as_posix(),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'crop_points_top_left':list(rect),'render_scale':scale,'pixel_size':list(crop.size),
          'method':'Original PDF rendered and directly cropped; no redraw',
          'role':'Supplementary method figure; original notebook is a framework tutorial',
          'pypdfium2':importlib.metadata.version('pypdfium2'),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved original Figure 1 from PDF page 3; inspect visually.')
