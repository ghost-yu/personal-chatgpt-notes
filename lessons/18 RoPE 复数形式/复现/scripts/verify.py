"""Execute the pinned notebook and Meta precompute using real CPU PyTorch."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
import urllib.request
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]


def read_source(path, url, expected):
    raw=path.read_bytes() if path else urllib.request.urlopen(url,timeout=30).read()
    allowed=expected if isinstance(expected,list) else [expected]
    assert hashlib.sha256(raw).hexdigest() in allowed, 'Pinned source hash mismatch'
    return raw.decode('utf-8')


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',type=Path)
    parser.add_argument('--official-model',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    base='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+manifest['tutorial_commit']+'/'
    record=manifest['notebooks'][0]
    expected=[record['sha256'],record['upstream_sha256']] if args.notebook else record['upstream_sha256']
    nb=json.loads(read_source(args.notebook,base+record['notebook'],expected))
    src=read_source(args.official_model,'https://raw.githubusercontent.com/meta-llama/llama/'+manifest['meta_commit']+'/llama/model.py',manifest['meta_model_sha256'])
    state={}; executed=[]; real_pe=None; real_class=None; captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']!='code':continue
            code=''.join(cell['source'])
            exec(compile(code,'notebook cell '+str(i),'exec'),state)
            executed.append(i)
            if i==15:
                real_pe=state['spe']().detach().clone()
                real_class=state['SinPositionEncoding']
    plt.close('all')
    complex_pe=state['spe']()
    assert tuple(real_pe.shape)==(10,6)
    torch.testing.assert_close(real_pe,complex_pe,rtol=1e-6,atol=1e-7)
    tree=ast.parse(src)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='precompute_freqs_cis')
    ns={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'pinned Meta precompute','exec'),ns)
    precompute=ns['precompute_freqs_cis']
    c=precompute(6,10)
    assert c.dtype==torch.complex64 and c.shape==(10,3)
    torch.testing.assert_close(c.abs(),torch.ones(10,3),rtol=1e-6,atol=1e-7)
    ordered=torch.view_as_real(c)
    restored=ordered.flip((2,)).flatten(1,2)
    torch.testing.assert_close(restored,real_pe,rtol=1e-6,atol=1e-7)
    assert not torch.allclose(ordered.flatten(1,2),real_pe)
    wrong_angles=torch.outer(torch.arange(10).float(),1/(10000**torch.arange(0,6,2).float()))
    wrong_c=torch.polar(torch.ones_like(wrong_angles),wrong_angles)
    assert not torch.allclose(wrong_c,c)
    q=torch.tensor([.3,-1.2,2.,.5,-.7,.8],dtype=torch.float64)
    k=torch.tensor([1.1,.4,-.6,1.5,2.,-.3],dtype=torch.float64)
    omega=10000**(-torch.arange(0,6,2,dtype=torch.float64)/6)
    def rotate(v,pos):
        z=torch.view_as_complex(v.reshape(3,2))
        phase=torch.polar(torch.ones_like(omega),pos*omega)
        return torch.view_as_real(z*phase).flatten()
    m,n=3,7
    qr=rotate(q,m); kr=rotate(k,n)
    matrix=np.zeros((6,6))
    for j,w in enumerate(omega.tolist()):matrix[2*j:2*j+2,2*j:2*j+2]=rotation(m*w)
    rotation_error=float(np.max(np.abs(qr.numpy()-matrix@q.numpy())))
    torch.testing.assert_close(qr.norm(),q.norm(),rtol=1e-12,atol=1e-12)
    actual=float(qr@kr)
    zq=torch.view_as_complex(q.reshape(3,2)); zk=torch.view_as_complex(k.reshape(3,2))
    relative=float((zq*zk.conj()*torch.exp(1j*(m-n)*omega)).real.sum())
    matrix_relative=float(q@rotate(k,n-m))
    shifted=float(rotate(q,m+11)@rotate(k,n+11))
    assert max(abs(actual-relative),abs(actual-matrix_relative),abs(actual-shifted))<1e-12
    no_conjugate=float((zq*zk*torch.exp(1j*(m-n)*omega)).real.sum())
    assert abs(actual-no_conjugate)>1e-3
    z=1+2j; quarter=z*np.exp(1j*np.pi/2)
    np.testing.assert_allclose([quarter.real,quarter.imag],[-2,1],atol=1e-15)
    # Validate real torch polar's autograd through the angle input.
    angles=torch.tensor([.2,-.7],dtype=torch.float64,requires_grad=True)
    polar=torch.polar(torch.ones_like(angles),angles)
    scalar=(polar.real+2*polar.imag).sum(); scalar.backward()
    expected=-angles.detach().sin()+2*angles.detach().cos()
    torch.testing.assert_close(angles.grad,expected,rtol=1e-12,atol=1e-12)
    # No even-dimension guard in upstream; demonstrate why the caller must enforce it.
    try:
        real_class(10,5)()
        odd_error='unexpected success'
    except RuntimeError as exc:
        odd_error=str(exc)
    assert odd_error!='unexpected success'
    assert precompute(5,10).shape==(10,2)
    assert state['SinPositionEncoding'](10,5)().shape==(10,6)
    complex_bytes=c.numel()*c.element_size()
    # Own explanatory figure: equal-scale unit circle, quarter-turn, frequency schedule.
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    a=np.linspace(0,2*np.pi,500); ax=axes[0]
    ax.plot(np.cos(a),np.sin(a),color='#9ca3af',label='unit circle')
    for endpoint,label,color in [(complex(z),'z = 1 + 2i','#2563eb'),(quarter,'z exp(i*pi/2) = -2 + i','#dc2626')]:
        ax.annotate('',xy=(endpoint.real,endpoint.imag),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':2,'color':color})
        ax.text(endpoint.real+.06,endpoint.imag+.06,label,fontsize=9,color=color)
    ax.axhline(0,color='#d1d5db'); ax.axvline(0,color='#d1d5db')
    ax.set(xlabel='Real',ylabel='Imaginary',title='Complex multiplication rotates a vector',xlim=(-2.8,2.6),ylim=(-1.5,2.8)); ax.set_aspect('equal')
    ax=axes[1]; frequencies=10000**(-np.arange(0,128,2)/128)
    ax.semilogy(np.arange(64),frequencies,'-',color='#2563eb')
    ax.set(xlabel='Pair index j (0-based)',ylabel='Radians per position',title='Llama head_dim=128, base=10000');ax.grid(alpha=.3)
    fig.savefig(ROOT/'图片/complex_rotation.png',dpi=160);plt.close(fig)
    report={'status':'部分成功','execution':'real torch tensors, kernels and autograd; no fixtures','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'device':'cpu','executed_code_cells':executed,'real_sin_class_parameters':sum(p.numel() for p in real_class(10,6).parameters()),'pe_shape':list(real_pe.shape),'complex_frequency_shape':list(c.shape),'complex_dtype':str(c.dtype),'frequencies_dim6':[float(x) for x in 10000**(-torch.arange(0,6,2).float()/6)],'real_complex_pe_max_error':float((restored-real_pe).abs().max()),'missing_dim_denominator_max_error':float((wrong_c-c).abs().max()),'complex_table_bytes_dim6_end10':complex_bytes,'real_imag_pair_bytes':ordered.numel()*ordered.element_size(),'rotation_matrix_max_error':rotation_error,'norm_error':float(abs(qr.norm()-q.norm())),'rotated_dot':actual,'relative_complex_dot':relative,'relative_matrix_dot':matrix_relative,'common_shift_dot':shifted,'wrong_no_conjugate_dot':no_conjugate,'polar_angle_gradient_max_error':float((angles.grad-expected).abs().max()),'odd_dim_real_class_error':odd_error,'odd_dim_precompute_shape':[10,2],'odd_dim_complex_sin_shape':[10,6],'limitations':['No model weights, Llama generation, attention kernel or CUDA test','No complete RoFormer experiments or long-context quality evaluation','Main notebook run; supporting overview inspected statically, not fully executed']}
    output=ROOT/'复现/outputs';output.mkdir(exist_ok=True)
    (output/'教学验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'notebook_stdout.txt').write_text(captured.getvalue(),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
