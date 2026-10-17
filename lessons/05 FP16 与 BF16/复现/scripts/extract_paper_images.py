"""Direct crop of BFLOAT16 paper v3 Table 1, without redrawing."""
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
assert len(doc) == 10
scale = 4
rect = (120, 154, 497, 237)
image = doc[3].render(scale=scale).to_pil().crop(tuple(round(v*scale) for v in rect))
(root/'图片').mkdir(exist_ok=True)
path = root/'图片/BFLOAT16原论文_Table1_格式对照.png'
image.save(path)
record = {
    'paper':'A Study of BFLOAT16 for Deep Learning Training',
    'version':'arXiv:1905.12322v3','pdf_url':'https://arxiv.org/pdf/1905.12322v3',
    'local_pdf':str(args.pdf.resolve()),
    'pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
    'page_1based':4,'table':'Table 1','crop_points_top_left':list(rect),
    'render_scale':scale,'image':path.relative_to(root).as_posix(),
    'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixel_size':list(image.size),
    'method':'Original PDF render and direct crop; no redraw or edits to table data',
    'role':'Supplementary format comparison; original course has no unique main paper',
    'pypdfium2':importlib.metadata.version('pypdfium2'),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
(root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved original v3 Table 1 from PDF page 4; inspect visually.')
