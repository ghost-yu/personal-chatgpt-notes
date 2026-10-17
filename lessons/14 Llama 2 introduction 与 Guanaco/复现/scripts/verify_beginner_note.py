"""Execute the four Python examples verbatim from the current reading note."""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--reference-script', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.tokenizer) == '9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'
    assert sha(args.reference_script) == 'e75c36b75daa03445ef9e763847b8e1450b561fbc222f345e1daa65f46687aec'
    root = Path(__file__).resolve().parents[2]
    note = root/'解释.md'
    front = note.read_text(encoding='utf-8').split('<details>')[0]
    blocks = re.findall(r'^```python\n(.*?)^```', front, re.M | re.S)
    assert len(blocks) == 4
    os.environ['LESSON14_TOKENIZER'] = str(args.tokenizer.resolve())
    namespace = {'__name__': '__main__'}
    captures = []
    for index, block in enumerate(blocks, 1):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(block, f'{note.name}:block{index}', 'exec'), namespace)
        captures.append(dict(block=index, sha256=hashlib.sha256(block.encode()).hexdigest(), output=stream.getvalue()))
        print(f'Block {index} passed:\n{stream.getvalue()}')
    # Parse only the official dataclass; do not execute its remote model/data calls.
    from dataclasses import dataclass, field, fields
    from typing import Optional
    from transformers import HfArgumentParser
    tree = ast.parse(args.reference_script.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ScriptArguments')
    scope = dict(dataclass=dataclass, field=field, Optional=Optional)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'official-dataclass-only', 'exec'), scope)
    parsed = HfArgumentParser(scope['ScriptArguments']).parse_args_into_dataclasses([
        '--model_name', 'meta-llama/Llama-2-7b-hf', '--dataset_name', 'timdettmers/openassistant-guanaco',
        '--load_in_4bit', '--use_peft', '--batch_size', '16', '--gradient_accumulation_steps', '2'])[0]
    parsed = {item.name: getattr(parsed, item.name) for item in fields(parsed)}
    assert (parsed['seq_length'], parsed['peft_lora_r'], parsed['peft_lora_alpha'], parsed['num_train_epochs']) == (512, 64, 16, 3)
    record = dict(status='passed', scope='Four verbatim note blocks, CPU random tiny Llama; reference dataclass/CLI parsing only.',
                  command=[sys.executable]+sys.argv, cwd=str(Path.cwd()),
                  versions={name: importlib.metadata.version(name) for name in ['torch','transformers','trl','peft','datasets','accelerate']},
                  python=sys.version, note_sha256=sha(note), script_sha256=sha(Path(__file__)),
                  tokenizer_sha256=sha(args.tokenizer), reference_script_sha256=sha(args.reference_script),
                  original_notebook_sha256=sha(root/'代码/tutorials/llama2_introduction.ipynb'),
                  blocks=captures, results=namespace['results'], reference_arguments=parsed,
                  image_record_sha256=sha(root/'复现/outputs/论文截图来源.json'),
                  limits=['No pretrained 7B weights, GPU, bitsandbytes, original 3-epoch CLI run, full Guanaco, or quality evaluation.',
                          'Historical verification files unchanged and not rerun.'])
    (root/'复现/outputs/基础讲解验证.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('All four examples and reference argument parsing passed.')


if __name__ == '__main__':
    main()
