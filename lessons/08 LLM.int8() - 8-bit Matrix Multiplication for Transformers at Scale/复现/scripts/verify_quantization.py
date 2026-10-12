# -*- coding: utf-8 -*-
"""Deterministic CPU teaching experiment. No torch, models, or CUDA kernels."""
import json
from pathlib import Path

X = [[2, 45, -1, -17, -1], [0, 12, 3, -63, 2], [-1, 37, -1, -83, 0]]
W = [[-1, 0], [2, 0], [0, -2], [3, -2], [-1, 2]]

def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def add(a, b):
    return [[x+y for x, y in zip(ar, br)] for ar, br in zip(a, b)]

def select(a, b, ids):
    return [[row[i] for i in ids] for row in a], [b[i] for i in ids]

def quant_mm(a, b, rounded=True):
    # Python round uses nearest-even on ties. Python ints do not overflow.
    cx = [max(map(abs, row)) for row in a]
    cw = [max(map(abs, col)) for col in zip(*b)]
    op = round if rounded else (lambda v: v)
    qa = [[op(127*x/s) if s else 0 for x in row] for row, s in zip(a, cx)]
    qb = [[op(127*x/s) if s else 0 for x, s in zip(row, cw)] for row in b]
    acc = mm(qa, qb)
    out = [[v*cx[i]*cw[j]/127**2 for j,v in enumerate(row)] for i,row in enumerate(acc)]
    return out, {"row_absmax":cx,"column_absmax":cw,"quantized_x":qa,"quantized_w":qb,"accumulator":acc}

def error(a, b):
    return max(abs(x-y) for ar, br in zip(a,b) for x,y in zip(ar,br))

def main():
    exact = mm(X,W)
    out_ids = [j for j,col in enumerate(zip(*X)) if max(map(abs,col)) > 6]
    reg_ids = [j for j in range(len(W)) if j not in out_ids]
    xr, wr = select(X,W,reg_ids)
    xo, wo = select(X,W,out_ids)
    regular, detail = quant_mm(xr,wr)
    mixed = add(regular,mm(xo,wo))
    naive, _ = quant_mm(X,W)
    unrounded, _ = quant_mm(xr,wr,False)
    unrounded_mixed = add(unrounded,mm(xo,wo))
    zero, _ = quant_mm([[0,0]],[[0],[0]])
    assert exact == [[38,34],[-167,124],[-174,168]]
    assert out_ids == [1,3]
    assert error(unrounded_mixed,exact) < 1e-10
    assert 0 < error(mixed,exact) < error(naive,exact)
    assert zero == [[0]]
    # All selected feature contributions appear once; no double-counting.
    assert add(mm(xr,wr),mm(xo,wo)) == exact
    result = {"status":"passed","scope":"Python integer arithmetic; not an int8 CUDA/FP16 run", "outlier_indices":out_ids,"ground_truth":exact,"original_unrounded_example":unrounded_mixed,"mixed_quantized":mixed,"naive_quantized":naive,"mixed_max_abs_error":error(mixed,exact),"naive_max_abs_error":error(naive,exact),"regular_detail":detail,"zero_scale_guard":zero}
    path = Path(__file__).resolve().parents[1]/'outputs'/'教学验证.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__ == '__main__':
    main()
