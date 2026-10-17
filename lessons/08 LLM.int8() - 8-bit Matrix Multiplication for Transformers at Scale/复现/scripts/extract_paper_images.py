"""Direct crops of fixed LLM.int8() v2 PDF; no redraw."""
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
assert len(doc)==20
records=[]
for page,label,filename,rect in [
 (3,'Figure 2','LLMint8原论文_Figure2_计算路径.png',(106,69,507,225)),
 (6,'Table 1','LLMint8原论文_Table1_C4结果.png',(119,145,493,281))]:
 image=root/'图片'/filename
 image.parent.mkdir(exist_ok=True)
 crop=doc[page-1].render(scale=4).to_pil().crop(tuple(round(v*4) for v in rect))
 crop.save(image)
 records.append({'page_1based':page,'figure_or_table':label,'image':image.relative_to(root).as_posix(),
  'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
  'crop_points_top_left':list(rect),'render_scale':4,'pixel_size':list(crop.size)})
record={'paper':'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
 'version':'arXiv:2208.07339v2','pdf_url':'https://arxiv.org/pdf/2208.07339v2',
 'local_pdf':str(args.pdf.resolve()),'pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
 'method':'Fixed original PDF rendered and directly cropped; no redraw',
 'pypdfium2':importlib.metadata.version('pypdfium2'),'images':records,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'复现/outputs/论文截图来源.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved Figure 2 page 3 and Table 1 page 6; inspect visually.')
