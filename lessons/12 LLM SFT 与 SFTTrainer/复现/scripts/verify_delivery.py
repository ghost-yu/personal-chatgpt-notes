# -*- coding: utf-8 -*-
"""核对27格原件/派生件及22个AST，再真实执行旧版CPU SFT教学流程。"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import platform
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import accelerate
import datasets
import nbformat
import peft
import tokenizers
import torch
import transformers
import trl
from peft import PeftModel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, TrainingArguments, set_seed
from trl import SFTTrainer
from trl.trainer.utils import ConstantLengthDataset, DataCollatorForCompletionOnlyLM

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "代码/tutorials"
SHA = "bcbb4c7c4e7e0310a07e8dccf0437eb3e0597031288e2a6332b3c677384dfa51"
CODE = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25, 26]
EXECUTED = [0, 1, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 22, 23, 26]


def source(cell):
    return "".join(cell["source"])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))


def structural_audit():
    paths = [FOLDER / n for n in ("llm_training_sft.ipynb", "原代码_带说明.ipynb", "代码注释.ipynb")]
    raw, explained, annotated = [json.loads(p.read_bytes()) for p in paths]
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == SHA
    assert len(raw["cells"]) == 27 and [i for i,c in enumerate(raw["cells"]) if c["cell_type"]=="code"] == CODE
    for nb in (raw, explained, annotated):
        # validate may warn about historical missing ids; do not rewrite the original document.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nbformat.validate(nbformat.from_dict(copy.deepcopy(nb)))
    for derived, changed in ((explained, False), (annotated, True)):
        assert len(derived["cells"]) == 55
        assert {k:v for k,v in derived.items() if k!="cells"} == {k:v for k,v in raw.items() if k!="cells"}
        for i,c in enumerate(raw["cells"]):
            note, saved = derived["cells"][1+2*i:3+2*i]
            assert note["cell_type"] == "markdown" and "单元 {}（".format(i) in source(note)
            assert {k:v for k,v in c.items() if k!="source"} == {k:v for k,v in saved.items() if k!="source"}
            if not changed or c["cell_type"] != "code":
                assert saved == c
            else:
                assert ast.dump(ast.parse(source(c))) == ast.dump(ast.parse(source(saved)))
                # All original lines are retained in order; only comment lines are inserted.
                old = source(c).splitlines(keepends=True)
                j = 0
                for line in source(saved).splitlines(keepends=True):
                    if j < len(old) and line == old[j]:
                        j += 1
                    else:
                        assert line.lstrip().startswith("#")
                assert j == len(old)
    original_code = "\n\n".join(source(raw["cells"][i]) for i in CODE)
    reading = FOLDER / "llm_training_sft_代码注释.py"
    assert ast.dump(ast.parse(reading.read_text(encoding="utf-8"))) == ast.dump(ast.parse(original_code))
    doc = (ROOT/"代码单元解释.md").read_text(encoding="utf-8")
    assert all("## 单元 {}（".format(i) in doc for i in range(27))
    result = dict(original_sha256=SHA, cells=27, code_indices=CODE, markdown_indices=[2,10,16,17,21],
                  schemas_passed=3, derived_cells=55, code_ast_passed=22, reading_ast_passed=True,
                  original_lines_preserved=True, metadata_and_historical_outputs_preserved=True,
                  all_27_notes_present=True)
    write_json(ROOT/"复现/outputs/annotation_verification.json", result)
    return raw, annotated, result


class CPUSettings(ast.NodeTransformer):
    """Only literal arguments of original cells24/25 change; imported APIs remain authentic."""
    def __init__(self, output):
        self.output = str(output)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments":
            for kw in node.keywords:
                if kw.arg == "fp16": kw.value = ast.Constant(False)
                if kw.arg == "output_dir": kw.value = ast.Constant(self.output)
            node.keywords += [ast.keyword(arg="max_steps", value=ast.Constant(2)),
                              ast.keyword(arg="report_to", value=ast.Constant("none"))]
        elif isinstance(node.func, ast.Name) and node.func.id == "SFTTrainer":
            for kw in node.keywords:
                if kw.arg == "max_seq_length": kw.value = ast.Constant(32)
        return node


def module_from_file(path):
    spec = importlib.util.spec_from_file_location("sft_label_teaching", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_state(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_cells(cells, tokenizer_path, output, teaching):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = teaching.make_teaching_dataset()
    config = LlamaConfig(vocab_size=50320, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=128,
                         bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    model = LlamaForCausalLM(config).half()
    assert not getattr(model,"is_loaded_in_4bit",False) and not getattr(model,"is_loaded_in_8bit",False)
    context = dict(tokenizer=tokenizer, train_dataset=dataset, model=model)
    stream = io.StringIO()
    sizes = {}
    base_state = None
    with contextlib.redirect_stdout(stream):
        for i in EXECUTED:
            exec(compile(source(cells[i]),"original_cell_"+str(i),"exec"), context)
            model = context["model"]
            if i == 19:
                sizes["before_embedding"] = list(model.model.embed_tokens.weight.shape)
            if i == 20:
                sizes["after_embedding"] = list(model.model.embed_tokens.weight.shape)
                sizes["after_head"] = list(model.lm_head.weight.shape)
                assert model.config.vocab_size == len(tokenizer) == 50257
            if i == 22:
                assert all(not p.requires_grad for p in model.parameters())
                assert all(p.dtype==torch.float32 for p in model.parameters())
                assert not model.is_gradient_checkpointing
                base_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model = context["model"]
    assert isinstance(model, PeftModel)
    targets = [name for name,m in model.named_modules() if hasattr(m,"lora_A")]
    assert len(targets)==2 and all(name.endswith(("q_proj","v_proj")) for name in targets)
    trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in trainable.values()) == 1024
    assert all("lora_" in name for name in trainable)

    labels, block = teaching.audit_labels(tokenizer, dataset)
    batch = {key:value.unsqueeze(0) for key,value in block.items()}
    model.eval()
    with torch.no_grad():
        initial = model(**batch)
        with model.disable_adapter():
            base_logits = model(**batch).logits
        torch.testing.assert_close(initial.logits, base_logits, rtol=0, atol=0)
        manual = teaching.shifted_cross_entropy(initial.logits, batch["labels"])
        torch.testing.assert_close(initial.loss, manual, rtol=0, atol=0)
    model.train()
    model(**batch).loss.backward()
    grads = {name:float(p.grad.norm()) for name,p in trainable.items()}
    assert all(p.grad is None for name,p in model.named_parameters() if name not in trainable)
    assert any(v>0 for name,v in grads.items() if "lora_B" in name)
    assert all(v==0 for name,v in grads.items() if "lora_A" in name)  # B initially zero.
    model.zero_grad(set_to_none=True)

    # Original fp16=True really fails on CPU; record it before the explicit teaching variant.
    failure_context = dict(context)
    with tempfile.TemporaryDirectory(prefix="sft-original-fp16-") as attempt:
        failure_context["TrainingArguments"] = TrainingArguments
        tree = ast.parse(source(cells[24]))
        # Only redirect output path so the failed original attempt cannot create a reading-folder checkpoint.
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg=="output_dir": node.value=ast.Constant(attempt)
        try:
            exec(compile(ast.fix_missing_locations(tree),"cell24_original_fp16","exec"), failure_context)
        except ValueError as error:
            fp16_failure = str(error)
            assert "FP16" in fp16_failure or "fp16" in fp16_failure
        else:
            raise AssertionError("Expected original fp16=True to reject CPU")

    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    trace = []
    def observe(module, args, kwargs, result):
        if module.training:
            trace.append(dict(input_shape=list(kwargs["input_ids"].shape),
                              label_shape=list(kwargs["labels"].shape),
                              logits_shape=list(result.logits.shape), loss=float(result.loss.detach()),
                              learning_rate=context["trainer"].optimizer.param_groups[0]["lr"],
                              logits_dtype=str(result.logits.dtype), loss_dtype=str(result.loss.dtype)))
    handle = model.register_forward_hook(observe, with_kwargs=True)
    with contextlib.redirect_stdout(stream):
        for i in (24,25):
            tree = CPUSettings(output).visit(ast.parse(source(cells[i])))
            exec(compile(ast.fix_missing_locations(tree),"cell_"+str(i)+"_CPU_variant","exec"), context)
    handle.remove()
    trainer = context["trainer"]
    assert trainer.model is model and trainer.state.global_step == 2
    assert type(trainer.data_collator).__name__ == "DataCollatorWithPadding"
    assert isinstance(trainer.train_dataset, ConstantLengthDataset) and trainer.train_dataset.infinite
    assert len([m for m in model.modules() if hasattr(m,"lora_A")]) == 2
    assert isinstance(trainer.optimizer, accelerate.optimizer.AcceleratedOptimizer)
    assert isinstance(trainer.optimizer.optimizer, torch.optim.AdamW)
    opt_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert opt_ids == {id(p) for p in trainable.values()}
    assert len(trace)==2 and all(t["input_shape"]==[4,32] and t["logits_shape"]==[4,32,50257] for t in trace)
    assert [t["learning_rate"] for t in trace] == [0.0, 2e-4]
    deltas = {name:float((p.detach()-before[name]).abs().max()) for name,p in model.named_parameters()}
    assert all(value==0 for name,value in deltas.items() if name not in trainable)
    assert any(value>0 for name,value in deltas.items() if name in trainable)
    adapter_state = {name:p.detach().clone() for name,p in model.named_parameters() if name in trainable}

    completion = DataCollatorForCompletionOnlyLM("### Response:", tokenizer=tokenizer)
    try:
        SFTTrainer(model=model,args=context["training_args"],train_dataset=dataset,tokenizer=tokenizer,
                   dataset_text_field="text",packing=True,max_seq_length=32,data_collator=completion)
    except ValueError as error:
        completion_failure = str(error)
        assert "not compatible" in completion_failure
    else: raise AssertionError("Expected completion-only/packing incompatibility")

    # 另建真实非packed训练器，仅检查数据准备与默认collator，不运行第三次训练。
    nonpacked_trainer = SFTTrainer(
        model=model, args=context["training_args"], train_dataset=dataset, tokenizer=tokenizer,
        dataset_text_field="text", packing=False, max_seq_length=32,
    )
    assert type(nonpacked_trainer.data_collator).__name__ == "DataCollatorForLanguageModeling"
    assert isinstance(nonpacked_trainer.train_dataset, datasets.Dataset)
    nonpacked_batch = next(iter(nonpacked_trainer.get_train_dataloader()))
    assert nonpacked_batch["input_ids"].shape == nonpacked_batch["labels"].shape == (4,32)

    # Actually save/reload the adapter against exactly the same resized, frozen random base.
    adapter_dir = output / "adapter-reload"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    saved = torch.load(adapter_dir/"adapter_model.bin",map_location="cpu",weights_only=True)
    assert len(saved)==4 and all("lora_" in name for name in saved)
    resized_config = copy.deepcopy(model.config)
    fresh = LlamaForCausalLM(resized_config)
    fresh.load_state_dict(base_state, strict=True)
    reloaded = PeftModel.from_pretrained(fresh, adapter_dir)
    model.eval(); reloaded.eval()
    with torch.no_grad():
        trained_logits = model(**batch).logits
        loaded_logits = reloaded(**batch).logits
    original_state = model.state_dict()
    reload_state = reloaded.state_dict()
    assert set(original_state) == set(reload_state)
    assert all(torch.equal(original_state[k], reload_state[k]) for k in original_state)
    reload_max_diff = float((trained_logits-loaded_logits).abs().max())
    # 保存/重载后的权重逐元素相等；输出仍出现FP32微小差异，原因未隔离。
    # 记录误差和明确容差，不声称重载推理逐位相等。
    torch.testing.assert_close(trained_logits, loaded_logits, rtol=1e-5, atol=1e-6)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(adapter_dir,local_files_only=True)
    text_for_reload = dataset[0]["text"]
    assert reloaded_tokenizer(text_for_reload)["input_ids"] == tokenizer(text_for_reload)["input_ids"]
    assert reloaded_tokenizer.pad_token_id == reloaded_tokenizer.eos_token_id == 50256
    # q/v adapter shapes do not encode vocabulary size: wrong base may load silently.
    wrong_config = copy.deepcopy(resized_config); wrong_config.vocab_size=50320
    wrong = PeftModel.from_pretrained(LlamaForCausalLM(wrong_config),adapter_dir)
    wrong.eval()
    with torch.no_grad(): wrong_shape=list(wrong(**batch).logits.shape)
    assert wrong_shape[-1]==50320
    checkpoints = sorted(output.glob("checkpoint-*"))
    assert len(checkpoints)==1 and checkpoints[0].name=="checkpoint-2"
    checkpoint_adapter_files = [p.relative_to(output).as_posix() for p in checkpoints[0].rglob("adapter_model.bin")]
    assert checkpoint_adapter_files
    report = dict(unaltered_whole_cells=EXECUTED, skipped_original_cells={"3":"proxy", "4":"remote full Alpaca", "11":"remote XgenTokenizer", "13":"7B 4bit model"},
        variants={"24":{"fp16":False,"output_dir":"temporary directory","max_steps":2,"report_to":"none"},"25":{"max_seq_length":32}},
        original_cpu_fp16_failure=fp16_failure, tokenizer=dict(class_name=type(tokenizer).__name__,vocab=len(tokenizer),eos_id=tokenizer.eos_token_id,pad_id=tokenizer.pad_token_id),
        teaching_rows=6, shapes=sizes, base_parameters_after_resize=sum(v.numel() for v in base_state.values()),
        kbit_prepare=dict(base_frozen=True,fp16_to_fp32=True,loaded_4bit=False,loaded_8bit=False,gradient_checkpointing=False),
        lora=dict(targets=targets,trainable_parameters=1024,r=16,alpha=32,initial_output_matches_base=True,initial_gradient_norms=grads),
        labels=labels, forward=dict(shape=list(initial.logits.shape),loss=float(initial.loss),manual_shift_loss=float(manual)),
        training=dict(updates=2,trace=trace,optimizer="torch.optim.AdamW via AcceleratedOptimizer", optimizer_only_adapter=True,
                      base_unchanged=True,adapter_max_deltas={n:v for n,v in deltas.items() if n in trainable},guard_same_model=True,
                      default_collator=type(trainer.data_collator).__name__,infinite_dataset=True,loss=trainer.state.log_history[-1].get("train_loss")),
        completion_packing_failure=completion_failure,
        nonpacked_trainer=dict(default_collator=type(nonpacked_trainer.data_collator).__name__,
                               batch_shape=list(nonpacked_batch["input_ids"].shape),train_called=False),
        reload=dict(adapter_keys=sorted(saved),state_tensors_exactly_equal=True,logits_exactly_equal=torch.equal(trained_logits,loaded_logits),
                    logits_max_abs_difference=reload_max_diff,logits_tolerance=dict(rtol=1e-5,atol=1e-6),tokenizer_ids_equal=True,
                    correct_vocab=50257,wrong_base_load_succeeded=True,wrong_base_logits_shape=wrong_shape,
                    epoch_checkpoint_adapter_files=checkpoint_adapter_files),
        stdout=stream.getvalue())
    return report, adapter_state, trained_logits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tokenizer",type=Path,required=True,help="fixed local GPT2 tokenizer directory; no model download")
    args=parser.parse_args()
    torch.set_num_threads(2)
    raw, annotated, static=structural_audit()
    teaching=module_from_file(ROOT/"代码/监督标签_教学补充.py")
    with tempfile.TemporaryDirectory(prefix="lesson12-real-sft-") as work:
        original,state1,logits1=run_cells(raw["cells"],args.tokenizer,Path(work)/"original",teaching)
        derived_cells=[annotated["cells"][2+2*i] for i in range(27)]
        changed,state2,logits2=run_cells(derived_cells,args.tokenizer,Path(work)/"annotated",teaching)
    assert set(state1)==set(state2)
    assert all(torch.equal(state1[k],state2[k]) for k in state1)
    assert torch.equal(logits1,logits2)
    assert original["labels"]==changed["labels"]
    assert original["training"]["trace"]==changed["training"]["trace"]
    versions={m.__name__:m.__version__ for m in [torch,transformers,peft,trl,datasets,accelerate,tokenizers]}
    package_sources={}
    for obj in [SFTTrainer,ConstantLengthDataset,PeftModel]:
        path=Path(inspect.getfile(obj))
        package_sources[obj.__name__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    tokenizer_assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.tokenizer.glob("*.json"))}
    tracked_scripts=[Path(__file__),ROOT/"复现/scripts/build_annotated.py",ROOT/"复现/scripts/download_original.py",ROOT/"代码/监督标签_教学补充.py"]
    result=dict(status="部分成功",scope="真实CPU小LLaMA/六行自建文本/固定GPT2 tokenizer；非Xgen7B/Alpaca全量/4bitGPU复现",
        command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),versions=versions,cuda_available=torch.cuda.is_available(),
        tokenizer_path=str(args.tokenizer),tokenizer_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",tokenizer_assets=tokenizer_assets,
        official_installed_sources=package_sources,script_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_scripts},
        original=original,annotated=changed,original_annotated=dict(adapter_exact=True,logits_exact=True,labels_exact=True,trace_exact=True,adapter_sha256=digest_state(state1)),
        not_run=["original proxy","full Alpaca download","remote XgenTokenizer","Xgen7B weights","bitsandbytes/CUDA 4bit","original 1024-token training epoch","quality benchmark"])
    write_json(ROOT/"复现/outputs/runtime_verification.json",result)
    print(json.dumps(dict(schemas=3,code_ast=22,whole_cells_per_version=len(EXECUTED),train_updates_per_version=2,
                         exact_adapter_and_logits=True,packed_eos=original["labels"]["packed_supervised_eos"],
                         trainable=1024),ensure_ascii=False))


if __name__=="__main__": main()
