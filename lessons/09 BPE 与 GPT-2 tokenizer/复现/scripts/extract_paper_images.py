"""Direct screenshots of ACL 2016 BPE paper Figure 1 and Algorithm 1."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pypdfium2 as pdfium
parser=argparse.ArgumentParser()
parser.add_argument('--pdf',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
doc=pdfium.PdfDocument(str(args.pdf))
assert len(doc)==11
images=[]
for label,filename,rect in [
 ('Figure 1','BPE原论文_Figure1_合并规则.png',(72,319,291,397)),
 ('Algorithm 1','BPE原论文_Algorithm1_学习合并.png',(72,62,291,311))]:
 path=root/'图片'/filename
 path.parent.mkdir(exist_ok=True)
 crop=doc[3].render(scale=4).to_pil().crop(tuple(round(v*4) for v in rect))
 crop.save(path)
 images.append({'figure_or_algorithm':label,'page_1based':4,'printed_page':1718,
  'image':path.relative_to(root).as_posix(),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'crop_points_top_left':list(rect),'render_scale':4,'pixel_size':list(crop.size)})
record={'paper':'Neural Machine Translation of Rare Words with Subword Units',
 'version':'ACL 2016, P16-1162 proceedings PDF',
 'pdf_url':'https://aclanthology.org/P16-1162.pdf','local_pdf':str(args.pdf.resolve()),
 'pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
 'method':'Original fixed PDF rendered and directly cropped; no redraw',
 'role':'Supplementary character-BPE algorithm source; not a GPT-2 architecture figure',
 'images':images,'pypdfium2':importlib.metadata.version('pypdfium2'),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved two original PDF crops; inspect visually.')
