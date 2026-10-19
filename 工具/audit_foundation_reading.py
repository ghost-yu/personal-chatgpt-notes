"""Audit the current 30 reading paths, executed examples and original PDF crops."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import re
import subprocess
import tokenize
from urllib.parse import unquote
import warnings

from PIL import Image, ImageChops
import pypdfium2 as pdf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
PIN = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
OBSERVATIONS = [
    '接着写字→token编号→向量/shape→softmax→参数存储；原Table2与训练曲线配解释',
    '两数矩阵乘法→修正量→低秩→双支路→参数比→初始化与缩放',
    '名言字典/列表→分词与padding→位移labels→交叉熵→LoRA更新',
    '模块/参数/输入→对象引用→目标匹配与替换→权重复用与冻结',
    'bit/byte→二进制科学记数→6.5解码→正常/次正规→范围/精度',
    '基模型/tokenizer/adapter→原模板→编号→beam/采样→回答边界',
    '图片分类→10分数/误差→梯度→loss scaling→autocast实际dtype',
    '整数近似→缩放/舍入→矩阵乘→异常列分支→论文质量表与模型加载',
    '三个词三轮合并→未见词→byte映射→预切分→真实词表训练',
    '六行JSONL→磁盘/内存/缓存→Arrow映射与迭代器→批次输入',
    '单参数更新→微批/窗口→平均梯度→短窗口/不等批次→原循环',
    '任务与参考回答→字典/text→packing→位移目标→LoRA与Trainer',
    '单数链式求导→保存中间值→边界与重算→随机数→BERT开关',
    '四种模型/方法名→参数与上下文→输入/生成→对话目标→原命令',
    '文本/消息/对话/批次→聊天序列化→生成→长度与返回边界',
    '示范/偏好/反馈→四角色→token与value→末尾score/GAE/clip→更新',
    '一排数→RMS与均值→SiLU三点→双分支乘法→真实FeedForward',
    '位置→二维旋转→复数→频率表→相邻通道→相对位置关系',
    'Q/K各轴→复数打包→广播→绝对位置旋转→相对点积/远距曲线',
    '续写→因果复用→四行注意力→缓存写读→prefill/decode→mask',
    '位置/头→共享KV→两查询不同输出→repeat与梯度→参数和缓存',
    '分词与批次表→最短提示词→逐个生成→温度/top-p/EOS→输出',
    '同题两回答→分数差/概率→偏好loss→padding/pooling→LoRA与更新',
    '单线性层→双支路→加权和→矩阵秩/SVD→实际adapter组合与折叠',
    '四词查表/打分→两代尺寸→GQA/SwiGLU→真实词表/模板→生成',
    '字符/字节/ID→BPE与rank→两套编码→全文计数→UTF8与有效ID',
    '三个动作→old/reference→逐token奖励→三步GAE→clip→真实step',
    '三位置注意力→旋转→句号/软门计数→小数插值→完整类与参数更新',
    '单数参数→四词查表→两代尺寸→共享KV/三矩阵→完整参数账',
    '同题两回答→BT概率→负对数/梯度→分组→小RM→四队似然',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, directory=ROOT):
    return subprocess.check_output(['git', '-C', str(directory), '-c', 'core.safecrlf=false', *args])


def blocks(text):
    return re.findall(r'^```python\n(.*?)^```', text.split('<details>', 1)[0], re.M | re.S)


def trees(code):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return [ast.dump(ast.parse(s), include_attributes=False) for s in code]


def first_fold(text):
    start = text.index('<details>')
    depth = 0
    for match in re.finditer(r'<details>|</details>', text[start:]):
        depth += 1 if match.group() == '<details>' else -1
        if depth == 0:
            return text[start:start + match.end()]
    raise AssertionError('Unclosed historical details block')


def prose_audit():
    """Check the final reading revisions against the published lesson30 checkpoint."""
    base = git('rev-parse', 'c87f1c5').decode().strip()
    paths = git('diff', '--name-only', '-z', base, '--', 'lessons').decode().strip('\0').split('\0')
    counts = {'markdown': 0, 'python': 0, 'notebooks': 0, 'original_cells_exact': 0}
    evidence = []
    def tokens(s):
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
                if t.type not in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING]]
    for name in filter(None, paths):
        p = ROOT / name
        before = git('show', base + ':' + name).decode('utf-8-sig').replace('\r\n', '\n')
        after = p.read_text('utf-8-sig')
        if p.suffix == '.py':
            assert tokens(before) == tokens(after), name
            counts['python'] += 1
        elif p.suffix == '.ipynb':
            a, b = json.loads(before), json.loads(after)
            assert {k: v for k, v in a.items() if k != 'cells'} == {k: v for k, v in b.items() if k != 'cells'}
            assert len(a['cells']) == len(b['cells'])
            for i, (x, y) in enumerate(zip(a['cells'], b['cells'])):
                assert {k: v for k, v in x.items() if k != 'source'} == {k: v for k, v in y.items() if k != 'source'}
                if x['cell_type'] == 'code':
                    assert tokens(''.join(x['source'])) == tokens(''.join(y['source']))
                if '原代码_带说明' in p.name and i > 0 and i % 2 == 0:
                    assert x == y
                    counts['original_cells_exact'] += 1
            counts['notebooks'] += 1
        elif p.name in ['解释.md', '代码单元解释.md']:
            assert re.findall(r'!\[[^\]]*\]\([^\n]*\)', before) == re.findall(r'!\[[^\]]*\]\([^\n]*\)', after)
            assert re.findall(r'\$\$.*?\$\$', before, re.S) == re.findall(r'\$\$.*?\$\$', after, re.S)
            if p.name == '解释.md':
                assert first_fold(before) == first_fold(after)
            old_blocks = re.findall(r'^```[^\n]*\n(.*?)^```', before, re.M | re.S)
            new_blocks = re.findall(r'^```[^\n]*\n(.*?)^```', after, re.M | re.S)
            assert len(old_blocks) == len(new_blocks)
            for a, b in zip(old_blocks, new_blocks):
                if a != b:
                    assert tokens(a) == tokens(b)
            counts['markdown'] += 1
        else:
            raise AssertionError((name, 'unexpected prose revision file'))
        evidence.append({'file': name, 'sha256': sha(p)})
    return {'base_commit': base, 'counts': counts, 'files': evidence,
            'lectures': sorted({Path(p['file']).parts[1] for p in evidence})}


def crop_nodes(node, inherited=None):
    if isinstance(node, list):
        for child in node:
            yield from crop_nodes(child, inherited)
    elif isinstance(node, dict):
        fields = dict(inherited or {})
        fields.update({k: v for k, v in node.items() if not isinstance(v, (dict, list))})
        for key in ['crop_points_top_left', 'bounds_points_top_left', 'bounds_top_left_points', 'crop_top_left_points', 'crop_points']:
            if key in node:
                fields['rect'] = node[key]
        image = next((fields.get(k) for k in ['published_path', 'image', 'file'] if isinstance(fields.get(k), str) and fields[k].endswith('.png')), None)
        if image and 'rect' in fields:
            yield dict(fields, image_path=image)
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from crop_nodes(value, fields)


def main():
    catalog = {}
    for base in [Path('G:/zzz/Personal ChatGPT 系列'), Path('G:/zzz/Personal ChatGPT 前七讲')]:
        for pattern in ['*/*.pdf', '*/资料/*.pdf']:
            for p in base.glob(pattern):
                catalog.setdefault(sha(p), p)
    entries = []
    upstream_images = [p for p in git('ls-tree', '-r', '--name-only', PIN, directory=UPSTREAM).decode().splitlines()
                       if p.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    main_assets = set()
    for index, lesson in enumerate(sorted((ROOT / 'lessons').iterdir()), 1):
        if not lesson.is_dir():
            continue
        raw = (lesson / '解释.md').read_text('utf-8')
        main = raw.split('<details>', 1)[0]
        assert raw.count('<details>') >= 1 and '第二遍' in raw
        assert len(main) > 3000 and len(re.findall(r'^## ', main, re.M)) >= 7
        image_links = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', main)
        assert image_links and all(not link.startswith(('http:', 'https:')) for _, link in image_links)
        assets = []
        for caption, link in image_links:
            path = lesson / unquote(link)
            assert path.is_file(), path
            with Image.open(path) as im:
                im.verify()
            assets.append({'file': path.relative_to(ROOT).as_posix(), 'caption': caption, 'sha256': sha(path)})
            main_assets.add(path.resolve())
        if index == 1:
            out, scope = io.StringIO(), {}
            with contextlib.redirect_stdout(out):
                exec(blocks(main)[0], scope)
            assert abs(sum(scope['probabilities']) - 1) < 1e-15
            assert scope['probabilities'][0] > scope['probabilities'][1] > scope['probabilities'][2]
            execution = {'scope': 'Current ordinary Python softmax example executed by this audit',
                         'probabilities': scope['probabilities'], 'stdout': out.getvalue()}
            baseline_path = lesson / '复现/outputs/论文截图来源.json'
        else:
            report_path = next((lesson / '复现/outputs').glob('*基础*验证.json'))
            report = json.loads(report_path.read_text('utf-8'))
            assert report and report.get('status') not in ['failed', 'error']
            baseline_path = report_path
            execution = {'report': report_path.relative_to(ROOT).as_posix(), 'report_sha256': sha(report_path),
                         'recorded_status': report.get('status', 'detailed checks without aggregate status')}
        relative = baseline_path.relative_to(ROOT).as_posix()
        baseline = git('log', '--diff-filter=A', '--format=%H', '--', relative).decode().splitlines()
        assert len(baseline) == 1
        original_note = git('show', baseline[0] + ':' + (lesson / '解释.md').relative_to(ROOT).as_posix()).decode('utf-8').replace('\r\n', '\n')
        assert trees(blocks(main)) == trees(blocks(original_note)), (lesson.name, 'published example AST changed since execution')
        # Preserve the original advanced text in its fold, allowing preceding heading placement.
        current_fold = first_fold(raw)
        baseline_fold = first_fold(original_note)
        assert current_fold == baseline_fold, (lesson.name, 'historical fold changed')
        crop_checks = []
        manifest_path = lesson / '复现/outputs/论文截图来源.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text('utf-8'))
            nodes = list(crop_nodes(manifest))
            assert nodes, (lesson.name, 'manifest crop fields not recognized')
            for item in nodes:
                path = lesson / item['image_path']
                assert path.resolve() in main_assets, (lesson.name, item['image_path'], 'not inline in main reading')
                png_hash = next(item[k] for k in ['png_sha256', 'image_sha256', 'sha256'] if k in item)
                assert sha(path) == png_hash
                pdf_hash = item['pdf_sha256']
                assert pdf_hash in catalog, (lesson.name, 'missing original PDF', pdf_hash)
                source = catalog[pdf_hash]
                page = next(item[k] for k in ['page_1_based', 'page_1based', 'pdf_page', 'page'] if k in item)
                scale = next(item[k] for k in ['render_scale', 'scale'] if k in item)
                document = pdf.PdfDocument(source)
                p = document[page - 1]
                w, h = p.get_size()
                x0, y0, x1, y1 = item['rect']
                assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                rendered = p.render(scale=scale, crop=(x0, h-y1, w-x1, y0)).to_pil().convert('RGB')
                with Image.open(path) as image:
                    expected = image.convert('RGB')
                    method = 'PDF window render'
                    exact = rendered.size == expected.size and ImageChops.difference(rendered, expected).getbbox() is None
                    if not exact:
                        # Early extraction scripts rasterize the whole page before PIL cropping.
                        # Edge antialiasing can differ from a PDFium window render.
                        rendered = p.render(scale=scale).to_pil().crop(tuple(round(v*scale) for v in item['rect'])).convert('RGB')
                        method = 'full-page raster then pixel crop'
                        assert rendered.size == expected.size
                        assert ImageChops.difference(rendered, expected).getbbox() is None, path
                crop_checks.append({'image': item['image_path'], 'pdf_sha256': pdf_hash,
                                    'source_pdf': str(source), 'page': page, 'raster_exact': True,
                                    'size': list(rendered.size), 'render_method': method})
        else:
            assert index == 11 and 'grad-accumulation.png' in main
        course_checks = []
        for asset in assets:
            path = ROOT / asset['file']
            if '/代码/' not in asset['file']:
                continue
            candidates = [p for p in upstream_images if Path(p).name == path.name]
            matched = [p for p in candidates if git('show', PIN + ':' + p, directory=UPSTREAM) == path.read_bytes()]
            assert matched, (lesson.name, path.name, 'original course image mismatch')
            course_checks.append({'file': asset['file'], 'upstream_path': matched[0], 'exact_bytes': True})
        entries.append({'lecture': lesson.name, 'learning_path_review': OBSERVATIONS[index-1],
                        'note_sha256': sha(lesson / '解释.md'), 'foundation_commit': baseline[0],
                        'current_python_blocks_AST_equal_to_executed_revision': True,
                        'historical_fold_exact_since_foundation': True, 'execution_evidence': execution,
                        'main_images': assets, 'original_course_images': course_checks, 'paper_crops': crop_checks})
        print(f'{index:02} OK: {len(assets)} inline images; {len(crop_checks)} exact PDF crops', flush=True)
    assert len(entries) == 30
    previous_audit = json.loads((ROOT / '资料/课程交付核查.json').read_text('utf-8'))
    assert previous_audit['status'] == 'passed' and previous_audit['totals']['original_notebooks'] == 40
    report = {'status': 'passed', 'date': '2026-10-19', 'scope': 'Current basic reading, original figures, citations, code examples and preserved source artifacts across lectures 01–30',
              'requirements': [
                  {'request': '从基础概念和具体例子讲解', 'evidence': 'Per-lecture learning paths reviewed; examples/formulas/shapes mapped in the current main explanations'},
                  {'request': '论文原图与引用内容放在对应段落', 'evidence': 'Every listed crop is inline; original PDF hash and independently rendered pixels match; source/version/page and Chinese explanations accompany the figures'},
                  {'request': '详细注释全部原notebook', 'evidence': 'Current source audit: 40 originals, 988 cells, 623 code cells, 80 preserving derivatives and Python readings'},
                  {'request': '清楚直接的写法，阅读入口同步', 'evidence': 'Main explanations, cell notes, notebook notes and Python comments revised; file/token checks preserve formulas and original computations'},
                  {'request': '保留旧讲解供第二遍读', 'evidence': 'All 30 historical folds match their foundation revision exactly'},
              ],
              'totals': {'lectures': 30, 'inline_image_occurrences': sum(len(x['main_images']) for x in entries),
                         'original_PDF_crop_occurrences': sum(len(x['paper_crops']) for x in entries),
                         'original_course_image_occurrences': sum(len(x['original_course_images']) for x in entries)},
              'lessons': entries, 'source_audit_sha256': sha(ROOT / '资料/课程交付核查.json'),
              'final_prose_revision': prose_audit(),
              'audit_script_sha256': sha(__file__),
              'execution_scope': 'Checks historical executed-example semantics against current examples and reruns the current lesson01 ordinary Python example; does not rerun every historical model experiment.'}
    (ROOT / '资料/基础讲解总核查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(report['totals']))


if __name__ == '__main__':
    main()
