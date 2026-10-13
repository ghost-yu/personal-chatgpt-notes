"""CPU NumPy teaching checks; does not execute PyTorch/FairScale or a model."""
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def rms(x, g, eps=1e-6):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)*g


def silu(x):
    return x/(1+np.exp(-x))


def finite_gradient(f, x, step=1e-6):
    out=np.zeros_like(x)
    for i in range(x.size):
        a=x.copy(); b=x.copy()
        a.flat[i]+=step; b.flat[i]-=step
        out.flat[i]=(f(a)-f(b))/(2*step)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--official-model',type=Path,help='Optional pinned Meta model.py snapshot; only AST-extract FeedForward.forward')
    args=parser.parse_args()
    rng=np.random.default_rng(17)
    x=rng.normal(size=(20,5,10))
    eps=1e-6; g=np.ones(10)
    r=rms(x,g,eps)
    mu=x.mean(-1,keepdims=True)
    ln=(x-mu)/np.sqrt(((x-mu)**2).mean(-1,keepdims=True)+1e-5)
    expected_norm=np.sqrt(10*x.var(-1,ddof=0)+10*x.mean(-1)**2)
    expected_norm/=np.sqrt((x*x).mean(-1)+eps)
    np.testing.assert_allclose(np.linalg.norm(r,axis=-1),expected_norm)
    assert r.shape==ln.shape==(20,5,10)
    np.testing.assert_allclose(ln.mean(-1),0,atol=1e-15)
    assert not np.allclose(r.mean(-1),0)
    np.testing.assert_array_equal(rms(np.zeros(10),g),np.zeros(10))
    z=np.array([3.,4.]); ones=np.ones(2)
    np.testing.assert_allclose(np.linalg.norm(rms(z,ones,0)),np.sqrt(2))
    np.testing.assert_allclose(np.linalg.norm(rms(z,2*ones,0)),2*np.sqrt(2))
    np.testing.assert_allclose(rms(2*z,ones,0),rms(z,ones,0))
    assert not np.allclose(rms(z+1,ones,0),rms(z,ones,0))
    assert not np.allclose(rms(2*z,ones,1),rms(z,ones,1))
    a=np.array([.3,-.7,1.2]); gain=np.array([1.,2.,.5]); upstream=np.array([.4,-.2,.8])
    s=np.sqrt(np.mean(a*a)+eps)
    jac=np.diag(gain)/s-np.outer(gain*a,a)/(a.size*s**3)
    analytic=upstream@jac
    numeric=finite_gradient(lambda v:float(upstream@rms(v,gain)),a)
    rms_error=float(np.max(np.abs(analytic-numeric)))
    assert rms_error<1e-8
    d=4; m=6
    W1=rng.normal(size=(m,d)); W3=rng.normal(size=(m,d)); W2=rng.normal(size=(d,m))
    t=rng.normal(size=(2,3,d))
    def ffn(v): return (silu(v@W1.T)*(v@W3.T))@W2.T
    y=ffn(t)
    assert y.shape==(2,3,d)
    v=t[0,0]; u=v@W1.T; q=v@W3.T; sig=1/(1+np.exp(-u))
    derivative=sig+u*sig*(1-sig)
    h=np.ones(d)@W2
    gradient=(h*q*derivative)@W1+(h*silu(u))@W3
    finite=finite_gradient(lambda vv:float(ffn(vv).sum()),v)
    ffn_error=float(np.max(np.abs(gradient-finite)))
    assert ffn_error<1e-7
    official={'status':'not requested'}
    if args.official_model:
        raw=args.official_model.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        expected=json.loads((Path(__file__).resolve().parents[2]/'来源清单.json').read_text(encoding='utf-8'))['meta_model_sha256']
        assert sha==expected,'Official snapshot differs from pinned manifest'
        tree=ast.parse(raw.decode('utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeedForward')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        method.returns=None
        for arg in method.args.args: arg.annotation=None
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        ns={'F':SimpleNamespace(silu=silu)}
        exec(compile(module,'official FeedForward.forward (NumPy fixture)','exec'),ns)
        fixture=SimpleNamespace(w1=lambda xx:xx@W1.T,w3=lambda xx:xx@W3.T,w2=lambda xx:xx@W2.T)
        observed=ns['forward'](fixture,t)
        np.testing.assert_allclose(observed,y)
        official={'status':'passed AST-extracted forward with NumPy fixtures','sha256':sha,'max_error':float(np.max(np.abs(observed-y)))}
    rounded=256*((int(2*(4*4096)/3)+255)//256)
    assert rounded==11008
    report={'status':'部分成功','python':platform.python_version(),'numpy':np.__version__,'seed':17,'input_shape':list(x.shape),'ln_mean_max_abs':float(np.abs(ln.mean(-1)).max()),'rms_mean_not_forced_zero':float(r.mean(-1)[0,0]),'rms_l2_min_max':[float(np.linalg.norm(r,axis=-1).min()),float(np.linalg.norm(r,axis=-1).max())],'sqrt10':float(np.sqrt(10)),'gain2_l2':float(np.linalg.norm(rms(z,2*ones,0))),'rms_gradient_max_error':rms_error,'silu_values':silu(np.array([-2.,0.,2.])).tolist(),'ffn_input_shape':list(t.shape),'ffn_hidden_shape':[2,3,m],'ffn_output_shape':list(y.shape),'ffn_gradient_max_error':ffn_error,'official_forward':official,'llama_7b_ffn_width':rounded,'swiglu_parameters_per_layer':3*4096*rounded,'standard_4d_ffn_parameters':8*4096**2,'limitations':['No PyTorch/FairScale forward or autograd','No trained weights, model generation, GPU benchmark, or paper training reproduction','NumPy float64 gradients do not verify FP16/BF16 numerical behavior']}
    dest=Path(__file__).resolve().parents[1]/'outputs/教学验证.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
