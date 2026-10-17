"""Execute two foundation code blocks verbatim from current notes, CPU only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parents[2]
notes=root/'解释.md'
text=notes.read_text(encoding='utf-8')
foundation=text.split('<details>',1)[0]
blocks=re.findall(r'```python\n(.*?)\n```',foundation,re.S)
assert len(blocks)==2
scope={}
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):
 for block in blocks:exec(compile(block,str(notes),'exec'),scope)
assert scope['q'].dtype==np.int8 and scope['q'].tolist()==[13,25,127]
assert scope['q'].nbytes==3 and scope['x'].nbytes==24
assert scope['q_large'].tolist()==[0,0,127]
assert scope['restored_large'].tolist()==[0.,0.,100.]
assert scope['R'].tolist()==[0,2,4] and scope['O'].tolist()==[1,3]
assert scope['qx'].tolist()==[[127,-64,-64],[0,127,85],[-127,-127,0]]
assert scope['qx'].dtype==scope['qw'].dtype==np.int8
assert scope['acc'].dtype==np.int32 and scope['outlier'].dtype==np.float16
assert scope['reference'].tolist()==[[38.,34.],[-167.,124.],[-174.,168.]]
assert scope['mixed'].shape==(3,2) and scope['mixed'].dtype==np.float64
mixed_error=float(np.abs(scope['mixed']-scope['reference']).max())
full_error=float(np.abs(scope['full']-scope['reference']).max())
assert np.isclose(mixed_error,0.01574803149605941,rtol=0,atol=1e-12)
assert np.isclose(full_error,1.0940541881083732,rtol=0,atol=1e-12)
# Independent checks of the same teaching function's zero-scale and accumulator semantics.
f=scope['quantized_product']
zero_output,*_=f(np.zeros((2,3)),np.zeros((3,2)))
assert np.array_equal(zero_output,np.zeros((2,2)))
column_zero,*_=f(np.array([[1.,0.],[0.,0.]]),np.array([[0.,2.],[0.,0.]]))
assert np.array_equal(column_zero,np.array([[0.,2.],[0.,0.]]))
wide=np.array([[127,127]],dtype=np.int32)@np.array([[127],[127]],dtype=np.int32)
assert wide.item()==32258
provenance=json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
for entry in provenance['images']:
 assert hashlib.sha256((root/entry['image']).read_bytes()).hexdigest()==entry['image_sha256']
course_images={}
for name in ['floats-rep.png','quant-freeze.png','llm-int8.png']:
 p=root/'代码/imgs'/name
 course_images[name]=hashlib.sha256(p.read_bytes()).hexdigest()
record={'status':'passed','date':'2026-10-17',
 'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
 'executed':'Two Python blocks extracted verbatim from current foundation text before historical details',
 'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in blocks],
 'python':platform.python_version(),'numpy':np.__version__,'device':'CPU',
 'command':[sys.executable,str(Path(__file__).resolve())],'working_directory':str(Path.cwd()),
 'stdout':buffer.getvalue(),
 'vector':{'q':scope['q'].tolist(),'restored':scope['restored'].tolist(),
           'q_bytes_excludes_scale':3,'float64_original_bytes':24,
           'large_outlier_q':scope['q_large'].tolist(),'large_outlier_restored':scope['restored_large'].tolist()},
 'matrix':{'regular_indices':scope['R'].tolist(),'outlier_indices':scope['O'].tolist(),
           'reference':scope['reference'].tolist(),'mixed':scope['mixed'].tolist(),
           'dtypes':{'qx':str(scope['qx'].dtype),'qw':str(scope['qw'].dtype),
                     'accumulator':str(scope['acc'].dtype),'outlier_branch':str(scope['outlier'].dtype),
                     'regular_restore_and_final_sum':str(scope['mixed'].dtype)},
           'mixed_max_error':mixed_error,'full_quantization_max_error':full_error,
           'zero_scale_checks_passed':True,'int32_two_products_sum':wide.item()},
 'paper_images':provenance['images'],'paper_images_visually_inspected':True,
 'original_course_image_sha256':course_images,'course_images_visually_inspected':True,
 'limits':'NumPy CPU educational calculation only. Regular restoration and final sum float64; outlier matmul float16. No final FP16 rounding simulation. No old notebook/runtime verifier rerun, bitsandbytes install, OPT6.7B download, CUDA, C4 quality or speed experiment.'}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(buffer.getvalue())
print('Verbatim code, int8/int32/FP16 branches, errors, zero scales, and image hashes passed.')
