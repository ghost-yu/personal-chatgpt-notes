"""Run the current note's four examples against fixed official CPU device variants."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types


SOURCE_HASHES = {
    'llama-generation.py': 'acc739aeda4aa222305ca5e7ae020df38d0fa841530d966f041c13e065db05c4',
    'llama-model.py': '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754',
    'llama-tokenizer.py': 'a675762526cd5f6e9d742a642e33a3560ea825522b8a9efb531d36b7c7a69e6c',
}
TOKEN_SHA = '9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CPUDevices(ast.NodeTransformer):
    def __init__(self):
        self.cuda_calls = 0
        self.cuda_devices = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            assert not node.args and not node.keywords
            node.func.attr = 'to'
            node.args = [ast.Constant('cpu')]
            self.cuda_calls += 1
        for keyword in node.keywords:
            if keyword.arg == 'device' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 'cuda':
                keyword.value.value = 'cpu'
                self.cuda_devices += 1
        return node


def module(name, path, transform=False):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    devices = CPUDevices()
    if transform:
        tree = devices.visit(tree)
    obj = types.ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), obj.__dict__)
    return obj, dict(cuda_to_cpu=devices.cuda_calls, device_cuda_to_cpu=devices.cuda_devices)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--fairscale-path', type=Path)
    args = parser.parse_args()
    for name, expected in SOURCE_HASHES.items():
        assert sha(args.sources/name) == expected, name
    assert sha(args.tokenizer) == TOKEN_SHA
    if args.fairscale_path:
        sys.path.insert(0, str(args.fairscale_path.resolve()))
    import torch
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel, destroy_model_parallel
    root = Path(__file__).resolve().parents[2]
    note = root/'解释.md'
    front = note.read_text(encoding='utf-8').split('<details>',1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', front, re.M|re.S)
    assert len(blocks) == 4
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='lesson15-note-') as temporary:
        work = Path(temporary)
        (work/'tokenizer.model').write_bytes(args.tokenizer.read_bytes())
        torch.distributed.init_process_group('gloo', init_method=(work/'rendezvous').as_uri(), rank=0, world_size=1)
        try:
            initialize_model_parallel(1)
            package = types.ModuleType('llama')
            package.__path__ = []
            sys.modules['llama'] = package
            token_module, _ = module('llama.tokenizer', args.sources/'llama-tokenizer.py')
            model_module, model_changes = module('llama.model', args.sources/'llama-model.py', True)
            gen_module, generation_changes = module('llama.generation', args.sources/'llama-generation.py', True)
            assert model_changes == dict(cuda_to_cpu=2, device_cuda_to_cpu=0)
            assert generation_changes == dict(cuda_to_cpu=0, device_cuda_to_cpu=3)
            tokenizer = token_module.Tokenizer(str(work/'tokenizer.model'))
            params = model_module.ModelArgs(dim=8,n_layers=1,n_heads=2,vocab_size=32000,
                multiple_of=8,max_seq_len=128,max_batch_size=4)
            torch.manual_seed(15)
            model = model_module.Transformer(params)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.ndim == 1:
                        parameter.fill_(1)
                    else:
                        parameter.normal_(mean=0, std=0.02)
            before = {name: value.clone() for name,value in model.state_dict().items()}
            llama = gen_module.Llama(model.eval(), tokenizer)
            generation_calls, forward_calls = [], []
            real_forward = model.forward
            def observed_forward(tokens, start_pos):
                logits = real_forward(tokens, start_pos)
                forward_calls.append(dict(start_pos=start_pos, ids=tokens.tolist(),
                    input_shape=list(tokens.shape), logits_shape=list(logits.shape)))
                return logits
            model.forward = observed_forward
            real_generate = llama.generate
            signature = inspect.signature(real_generate)
            def observed_generate(*call_args, **call_kwargs):
                bound = signature.bind(*call_args, **call_kwargs)
                bound.apply_defaults()
                row = copy.deepcopy(dict(bound.arguments))
                start = len(forward_calls)
                answer = real_generate(*call_args, **call_kwargs)
                row['result'] = copy.deepcopy(answer)
                row['forward_calls'] = copy.deepcopy(forward_calls[start:])
                generation_calls.append(row)
                return answer
            llama.generate = observed_generate
            namespace = dict(llama=llama,tokenizer=tokenizer,generation_calls=generation_calls,
                forward_calls=forward_calls,UNSAFE_ERROR=gen_module.UNSAFE_ERROR,torch=torch)
            captures = []
            for index, block in enumerate(blocks,1):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(compile(block,f'{note.name}:block{index}','exec'),namespace)
                captures.append(dict(block=index,sha256=hashlib.sha256(block.encode()).hexdigest(),output=output.getvalue()))
                print(f'Block {index} passed:\n{output.getvalue()}',flush=True)
            assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
            assert sum(p.numel() for p in model.parameters()) == 512856
            result = dict(status='passed',scope='Four verbatim note blocks, real official CPU Transformer/generate, no pretrained weights.',
                python=sys.version,torch=torch.__version__,fairscale=importlib.metadata.version('fairscale'),
                sentencepiece=importlib.metadata.version('sentencepiece'),backend='gloo',world_size=1,model_parallel_size=1,
                official_revision='7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e',model_config=vars(params),
                dtype='float32',parameters=512856,weights_unchanged=True,model_changes=model_changes,generation_changes=generation_changes,
                observation='Wrappers log arguments and call the full real methods; no fake answers or substituted model calculation.',
                results=namespace['results'],generation_calls=generation_calls,blocks=captures,
                note_sha256=sha(note),script_sha256=sha(Path(__file__)),tokenizer_sha256=sha(args.tokenizer),
                sources={name:sha(args.sources/name) for name in SOURCE_HASHES},
                original_notebooks={p.name:sha(p) for p in (root/'代码/tutorials').glob('*.ipynb') if '代码注释' not in p.name and '带说明' not in p.name},
                image_record_sha256=sha(root/'复现/outputs/论文截图来源.json'),
                command=[sys.executable]+sys.argv,cwd=str(Path.cwd()),
                not_run=['Official build/NCCL/torchrun/checkpoint loading','original7B/13B weights','CUDA/FP16/MP2+',
                         'training/GAtt','new sampling/logprob/EOS-controlled experiments','quality/performance benchmarks','historical verifiers'])
        finally:
            destroy_model_parallel()
            torch.distributed.destroy_process_group()
    (root/'复现/outputs/基础讲解验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Four current-note blocks passed; all model state unchanged.')


if __name__ == '__main__':
    main()
