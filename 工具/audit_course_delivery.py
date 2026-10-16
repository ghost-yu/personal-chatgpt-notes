"""Read-only checks of all 30 deliveries; does not rerun historical training."""
import ast
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize
import warnings

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
PIN = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
TM = TransformerManager()


def git(*args, upstream=False):
    return subprocess.check_output(['git', '-C', str(UPSTREAM if upstream else ROOT), *args])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(cell):
    return ''.join(cell['source'])


def tree(source):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return ast.dump(ast.parse(TM.transform_cell(source)), include_attributes=False)


def noncomment_tokens(source):
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                              tokenize.ENCODING, tokenize.ENDMARKER)]


def no_source(cell):
    result = copy.deepcopy(cell)
    result.pop('source')
    return result


def markdown_comments_only(original, annotated):
    if annotated.startswith(original) and annotated != original:
        suffix = annotated[len(original):]
        if suffix.lstrip().startswith(('新增源码注释（上方原片段保留）：', '新增源码注释（保留上方原片段）：')):
            snippets = re.findall(r'```(?:python|jsonc)\n(.*?)```', suffix, re.S)
            original_lines = {line.strip() for line in original.splitlines()}
            return bool(snippets) and all(
                line.strip() in original_lines
                for snippet in snippets for line in snippet.splitlines()
                if line.strip() and not line.lstrip().startswith(('#', '//')))
    def normalize(line):
        return re.sub(r'^(\s*```)(?:python|jsonc)$', r'\1', line.rstrip())
    a = [normalize(line) for line in original.splitlines()]
    b = [normalize(line) for line in annotated.splitlines()]
    for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        if op != 'insert' or not all(not line or line.lstrip().startswith(('#', '//')) for line in b[k:l]):
            return False
    return True


def ordinary_statements(source, remove_ipython=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        body = ast.parse(source).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if remove_ipython:
        body = [x for x in body if not (
            isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
            and isinstance(x.value.func, ast.Attribute)
            and isinstance(x.value.func.value, ast.Call)
            and isinstance(x.value.func.value.func, ast.Name)
            and x.value.func.value.func.id == 'get_ipython'
            and x.value.func.attr in ('system', 'run_line_magic'))]
    return [ast.dump(x, include_attributes=False) for x in body]


def main():
    lessons = sorted(p for p in (ROOT / 'lessons').iterdir() if p.is_dir())
    assert [int(p.name[:2]) for p in lessons] == list(range(1, 31))
    upstream_paths = git('-c', 'core.quotepath=false', 'ls-tree', '-r', '--name-only', PIN, upstream=True).decode('utf-8').splitlines()
    report = {'scope': 'All 30 lectures: current source preservation, derivative semantics, reading artifacts and existing execution evidence. Historical training is not rerun by this audit.',
              'upstream_commit': PIN, 'lessons': [], 'exceptions': []}
    total_cells = total_code = total_notebooks = 0
    for lesson in lessons:
        number = int(lesson.name[:2])
        for name in ('README.md', '解释.md', '代码单元解释.md', '来源清单.json', '复现/README.md', '复现/requirements.txt'):
            assert (lesson / name).is_file(), (lesson.name, name)
        explanation = (lesson / '解释.md').read_text(encoding='utf-8')
        # Structural checks supplement, rather than replace, per-lecture reading.
        for term in ('必须掌握', '第二遍', 'Paper ↔ Code', '复现'):
            assert term in explanation, (lesson.name, term)
        assert re.search(r'未|边界|限制', explanation), lesson.name
        originals = [p for p in (lesson / '代码').rglob('*.ipynb')
                     if '代码注释' not in p.name and '原代码_带说明' not in p.name]
        assert originals
        derived_paths = [p for p in (lesson / '代码').rglob('*.ipynb') if p not in originals]
        derivatives = {p: json.loads(p.read_text(encoding='utf-8')) for p in derived_paths}
        entry = {'lecture': lesson.name, 'notebooks': []}
        for original in originals:
            raw_bytes = original.read_bytes()
            matches = [p for p in upstream_paths if Path(p).name == original.name]
            matching_paths = [p for p in matches if git('show', PIN + ':' + p, upstream=True) == raw_bytes]
            assert matching_paths, original
            raw = json.loads(raw_bytes)
            variants = [(p, obj) for p, obj in derivatives.items()
                        if len(obj['cells']) == 2 * len(raw['cells']) + 1
                        and all(no_source(a) == no_source(obj['cells'][2*i+2])
                                for i, a in enumerate(raw['cells']))]
            assert len(variants) == 2, (original, [p.name for p, _ in variants])
            assert any(all(text(a) == text(obj['cells'][2*i+2]) for i, a in enumerate(raw['cells']))
                       for _, obj in variants), original
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', nbformat.warnings.MissingIDFieldWarning)
                nbformat.validate(nbformat.from_dict(raw))
                for _, obj in variants:
                    nbformat.validate(nbformat.from_dict(obj))
            concatenated = []
            shell_lines = []
            for i, cell in enumerate(raw['cells']):
                if cell['cell_type'] == 'code' and any(
                        t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                       tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT)
                        for t in tokenize.generate_tokens(io.StringIO(text(cell)).readline)):
                    assert any(any(t.type == tokenize.COMMENT and re.search('[\u4e00-\u9fff]', t.string)
                                   for t in tokenize.generate_tokens(io.StringIO(text(obj['cells'][2*i+2])).readline))
                               for _, obj in variants), (original, i, 'missing Chinese inline code comment')
                for _, obj in variants:
                    note = obj['cells'][2*i+1]
                    assert note['cell_type'] == 'markdown' and re.search('[\u4e00-\u9fff]', text(note)), (original, i)
                    counterpart = obj['cells'][2*i+2]
                    if cell['cell_type'] == 'code':
                        try:
                            assert tree(text(cell)) == tree(text(counterpart)), (original, i)
                        except SyntaxError:
                            assert number == 9 and original.name == 'tokenizer_basics.ipynb' and i == 22
                            assert noncomment_tokens(text(cell)) == noncomment_tokens(text(counterpart))
                    else:
                        assert markdown_comments_only(text(cell), text(counterpart)), (original, i)
                if cell['cell_type'] == 'code':
                    total_code += 1
                    if number == 9 and original.name == 'tokenizer_basics.ipynb' and i == 22:
                        report['exceptions'].append('09 tokenizer_basics cell22: original SyntaxError preserved in notebooks; invalid cell commented in Python reading.')
                        continue
                    concatenated.append(TM.transform_cell(text(cell)))
                    if number in (3, 5, 6):
                        shell_lines.extend(line.strip() for line in text(cell).splitlines()
                                           if line.lstrip().startswith(('!', '%')) or line.rstrip().endswith('??'))
            expected = ordinary_statements(''.join(concatenated), remove_ipython=number in (3, 5, 6))
            py_candidates = list((lesson / '代码').rglob('*代码注释.py'))
            reading_matches = []
            for py in py_candidates:
                content = py.read_text(encoding='utf-8')
                try:
                    if ordinary_statements(content) == expected:
                        for command in shell_lines:
                            assert any(line.strip().startswith('#') and command in line for line in content.splitlines()), (py, command)
                        reading_matches.append(py)
                except SyntaxError:
                    continue
            assert reading_matches, original
            if shell_lines:
                report['exceptions'].append(f'{number:02}: IPython-only commands retained as comments in Python reading; all ordinary Python statements match AST.')
            total_cells += len(raw['cells'])
            total_notebooks += 1
            relative = original.relative_to(ROOT).as_posix()
            # Its own publication commit must introduce this original notebook.
            commit = git('log', '--diff-filter=A', '--format=%H', '--', relative).decode().splitlines()
            assert len(commit) == 1, relative
            entry['notebooks'].append({'file': relative, 'upstream_paths': matching_paths,
                                      'sha256': sha(original), 'cells': len(raw['cells']),
                                      'code_cells': sum(c['cell_type'] == 'code' for c in raw['cells']),
                                      'two_preserving_derivatives': [p.relative_to(ROOT).as_posix() for p, _ in variants],
                                      'reading_python': [p.relative_to(ROOT).as_posix() for p in reading_matches],
                                      'publication_commit': commit[0]})
        reports = []
        for p in sorted((lesson / '复现/outputs').glob('*.json')):
            if not any(term in p.name for term in ('verification', '完整交付', '教学验证')):
                continue
            data = json.loads(p.read_text(encoding='utf-8'))
            assert isinstance(data, dict) and data, p
            assert data.get('status') not in ('failed', 'error'), p
            reports.append({'file': p.relative_to(ROOT).as_posix(), 'sha256': sha(p),
                            'recorded_status': data.get('status', 'no aggregate status; inspect detailed fields'),
                            'fields': list(data)})
        assert reports and list((lesson / '复现/scripts').glob('*.py')), lesson.name
        entry['existing_execution_evidence'] = reports
        report['lessons'].append(entry)
    assert (total_notebooks, total_cells, total_code) == (40, 988, 623)
    publication_commits = {c['publication_commit'] for x in report['lessons'] for c in x['notebooks']}
    assert len(publication_commits) >= 30
    report['totals'] = {'lectures': 30, 'original_notebooks': total_notebooks, 'original_cells': total_cells,
                        'original_code_cells': total_code, 'derivative_notebooks': total_notebooks * 2,
                        'distinct_original_publication_commits': len(publication_commits)}
    import sys
    subprocess.run([sys.executable, str(ROOT / '工具/check_documents.py')], check=True)
    report['limitations'] = ['Existing per-lecture execution records are inventoried, not a fresh full training rerun.',
                             'Sections and Chinese-note checks establish coverage; they do not automatically prove pedagogical quality.',
                             'Original-model training, GPU benchmarks and quality claims remain bounded by each lecture reproduction README.']
    report['status'] = 'passed'
    report['audit_script_sha256'] = sha(Path(__file__))
    (ROOT / '资料/课程交付核查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['totals'], ensure_ascii=True))


if __name__ == '__main__':
    main()
