# -*- coding: utf-8 -*-
"""Check published local links, code fences, and required reading sections."""
import json
import re
from pathlib import Path
from urllib.parse import unquote

root=Path(__file__).resolve().parents[1]
errors=[]
docs=list((root/'lessons').glob('*/解释.md'))
for p in root.rglob('*.md'):
    s=p.read_text(encoding='utf-8')
    if s.count('```')%2:errors.append(str(p.relative_to(root))+': unclosed code fence')
    for link in re.findall(r'\]\(([^)]+)\)',s):
        if link.startswith(('https://','http://','#')):continue
        dest=p.parent/unquote(link.split('#')[0])
        if not dest.exists():errors.append(str(p.relative_to(root))+': missing '+link)
for p in docs:
    s=p.read_text(encoding='utf-8')
    for term in ['必须掌握','第二遍','Paper ↔ Code','复现']:
        if term not in s:errors.append(str(p.relative_to(root))+': missing '+term)
print(json.dumps({'lessons':len(docs),'errors':errors},ensure_ascii=True,indent=2))
assert not errors
