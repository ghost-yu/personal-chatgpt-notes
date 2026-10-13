# -*- coding: utf-8 -*-
"""真实执行原版/注释版全部8代码格，并与位解码对照CPU tensor数值边界。"""
import hashlib
import base64
import inspect
import json
import math
import os
import platform
import re
import struct
import sys
from pathlib import Path
import IPython
import nbformat
import torch
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(path, annotated=False):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = nb['cells'][2::2] if annotated else nb['cells']
    assert len(cells) == 27
    shell = InteractiveShell.instance()
    shell.reset(new_session=False)
    # 在CLI中把IPython实际取得的源码送到文本捕获器，避免外部pager等待键盘。
    # hook只决定展示媒介；原??指令、pinfo2和inspect路径保持真实执行。
    def text_pager(self, data, start=0, screen_lines=0):
        print(data if isinstance(data,str) else data.get('text/plain',''))
    shell.set_hook('show_in_pager', text_pager)
    records = []
    wd = Path.cwd()
    os.chdir(path.parent)
    try:
        for index, cell in enumerate(cells):
            if cell['cell_type'] != 'code':
                continue
            with capture_output() as captured:
                result = shell.run_cell(''.join(cell['source']), store_history=False)
            error = result.error_before_exec or result.error_in_exec
            rec = {'cell':index, 'success':result.success, 'error_type':type(error).__name__ if error else None}
            if index == 18:
                assert isinstance(error, AssertionError) and 'Torch not compiled with CUDA enabled' in str(error)
                rec['expected_CPU_failure'] = str(error)
                rec['outer_get_device_properties_not_reached'] = True
            else:
                assert result.success, (index, error)
            if index in (5,26):
                image = result.result
                rendered = image._repr_png_()
                data, metadata = rendered if isinstance(rendered, tuple) else (rendered,{})
                if isinstance(data,str):
                    data = base64.b64decode(data)
                assert isinstance(data, bytes)
                filename = 'float16.png' if index==5 else 'bf-float.png'
                assert hashlib.sha256(data).hexdigest() == hash_file(path.parent.parent/'imgs'/filename)
                width, height = struct.unpack('>II', data[16:24])
                assert metadata.get('width') == (400 if index==5 else None)
                rec.update({'png_bytes':len(data), 'png_sha256':hashlib.sha256(data).hexdigest(),
                            'png_size':[width,height], 'display_width':metadata.get('width')})
            if index==16:
                assert result.result is False
                rec['cuda_bf16_supported'] = result.result
            if index==17:
                assert result.result is torch.bfloat16
                rec['result'] = str(result.result)
            if index==19:
                assert result.result is None and torch.version.hip is None
                rec['HIP_version'] = None
            if index==20:
                shown = captured.stdout+'\n'+captured.stderr+'\n'+ '\n'.join(str(o.data) for o in captured.outputs)
                shown = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', shown)
                assert 'def is_bf16_supported' in shown and 'including_emulation' in shown
                rec['pinfo2_showed_actual_function_source'] = True
                rec['inspection_contains_including_emulation'] = True
            records.append(rec)
    finally:
        os.chdir(wd)
    assert [r['cell'] for r in records] == [0,5,16,17,18,19,20,26]
    return {'notebook_sha256':hash_file(path), 'executed_code_cells':8,
            'successful_cells':7, 'expected_failures':1, 'cells':records}


def decode_bits(bits, exponent_bits, fraction_bits, bias):
    """补充标量解码：整数域→正常/次正规/特殊值；不假装它是原notebook代码。"""
    sign = (bits >> (exponent_bits+fraction_bits)) & 1
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits)-1)
    fraction = bits & ((1 << fraction_bits)-1)
    if exponent == (1 << exponent_bits)-1:
        return math.nan if fraction else (-math.inf if sign else math.inf)
    significand = fraction/(1 << fraction_bits)
    power = 1-bias if exponent==0 else exponent-bias
    if exponent:
        significand += 1
    return (-1 if sign else 1) * math.ldexp(significand, power)


def verify_numbers():
    results = {}
    for name, dtype, e, m, bias in [('fp16',torch.float16,5,10,15),('bf16',torch.bfloat16,8,7,127)]:
        bits = [0, 1<<15, 1, 1<<m, ((1<<e)-2)<<m | ((1<<m)-1), ((1<<e)-1)<<m, ((1<<e)-1)<<m | 1]
        # int16以二进制补码存同一16位；view只重解释位，不是数值类型转换。
        encoded = torch.tensor([b if b < 32768 else b-65536 for b in bits],dtype=torch.int16)
        values = encoded.view(dtype)
        decoded = [decode_bits(b,e,m,bias) for b in bits]
        for theoretical, actual in zip(decoded, values.to(torch.float64).tolist()):
            assert (math.isnan(theoretical) and math.isnan(actual)) or theoretical==actual
        assert torch.signbit(values[1]).item()
        info = torch.finfo(dtype)
        assert info.max == decoded[4] and info.tiny == decoded[3] and info.eps==2**(-m)
        smallest = torch.nextafter(torch.tensor(0,dtype=dtype),torch.tensor(1,dtype=dtype)).item()
        assert smallest == 2**(1-bias-m)
        one = torch.tensor(1,dtype=dtype)
        gap = (torch.nextafter(one,torch.tensor(2,dtype=dtype))-one).item()
        assert gap == 2**(-m)
        midpoints = torch.tensor([1+2**(-m-1),1+3*2**(-m-1)],dtype=torch.float64).to(dtype).tolist()
        assert midpoints == [1.0,1+2**(1-m)]  # ties to even: even retained significand wins.
        underflow = torch.tensor([2**(-bias-m)],dtype=torch.float64).to(dtype).item()
        assert underflow == 0.0  # smallest positive subnormal的一半，最近偶数舍入到0。
        results[name] = {'max':info.max,'min_normal':info.tiny,'min_subnormal_nextafter':smallest,
                         'gap_above_1':gap,'bytes_per_element':values.element_size(),
                         'seven_patterns_verified':['+0','-0','min_subnormal','min_normal','max','+inf','nan'],
                         'ties_to_even_midpoint_results':midpoints, 'half_min_subnormal_rounds_to':underflow}
    example = torch.tensor([0x7680],dtype=torch.int16).view(torch.float16).item()
    assert example == decode_bits(0x7680,5,10,15) == 26624
    converted = torch.tensor([1.001,100000.0],dtype=torch.float64)
    half, brain = converted.to(torch.float16), converted.to(torch.bfloat16)
    assert half[0].item()==1.0009765625 and brain[0].item()==1.0
    assert torch.isinf(half[1]).item() and torch.isfinite(brain[1]).item()
    results['example_bits_0x7680'] = example
    results['rounding_input_1_001'] = {'fp16':half[0].item(),'bf16':brain[0].item()}
    results['input_100000'] = {'fp16':'inf','bf16':brain[1].item()}
    results['tensor_path'] = 'int16[7]→view(dtype)[7]→float64[7]；float64[2]→float16/bfloat16[2]'
    results['same_shape_storage_bytes'] = {str(d):torch.zeros(7,dtype=d).numel()*torch.zeros(7,dtype=d).element_size() for d in (torch.float16,torch.bfloat16,torch.float32)}
    assert list(results['same_shape_storage_bytes'].values()) == [14,14,28]
    return results


def main():
    assert torch.__version__ == '2.5.1+cpu', '本脚本的预期失败路径固定在CPU构建，其他环境应单独记录'
    assert not torch.cuda.is_available()
    lesson = Path(__file__).resolve().parents[2]
    directory = lesson/'代码/tutorials'
    original = execute(directory/'05_float16_bf16.ipynb')
    annotated = execute(directory/'代码注释.ipynb',annotated=True)
    assert original['cells'] == annotated['cells']
    source = inspect.getsource(torch.cuda.is_bf16_supported)
    record = {
        'status':'部分成功', 'environment':{'python':platform.python_version(),'OS':platform.platform(),
          'torch':torch.__version__,'IPython':IPython.__version__,'nbformat':nbformat.__version__,
          'cuda_available':torch.cuda.is_available(),'torch_version_cuda':torch.version.cuda,'torch_version_hip':torch.version.hip},
        'command':'python lessons/05 FP16 与 BF16/复现/scripts/verify_runtime.py',
        'working_directory':'repository root; image execution temporarily uses each notebook parent',
        'original':original,'annotated':annotated,'original_and_annotated_behavior_equal':True,
        'supplemental_CPU_tensors':verify_numbers(),
        'inspected_source':{'function':'torch.cuda.is_bf16_supported','signature':str(inspect.signature(torch.cuda.is_bf16_supported)),
                            'source_sha256':hashlib.sha256(source.encode('utf-8')).hexdigest(),
                            'official_source':'https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/__init__.py'},
        'limits':['CPU-only; CUDA cell18 fails by actual original expression',
                  'No CUDA/ROCm/TPU arithmetic, AMP, model training, throughput or memory benchmark',
                  'CPU bit-storage and conversion results do not establish other hardware subnormal behavior',
                  'Original historical GPU outputs are preserved, never treated as current observations'],
    }
    (lesson/'复现/outputs/runtime_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Both notebooks: 8 code cells attempted, 7 passed, cell18 expected CPU failure; behavior equal.')
    print('Actual CPU bit patterns, finfo, nextafter, signed zero, NaN/Inf, round-to-even, overflow and storage checks passed.')


if __name__ == '__main__':
    main()
