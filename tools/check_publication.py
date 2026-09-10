# SPDX-License-Identifier: Apache-2.0
"""Heuristic public-tree checks, not a security or privacy proof."""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', 'venv', '__pycache__', 'runs', 'outputs', 'local_work', '.pytest_cache'}
BINARY_SUFFIXES = {'.npz', '.npy', '.pt', '.pth', '.pkl', '.pickle', '.zip', '.pem', '.key', '.p12', '.pfx'}
SECRET_PATTERNS = [r'gh[pousr]_[A-Za-z0-9]{25,}', r'github_pat_[A-Za-z0-9_]{25,}',
                   r'AKIA[A-Z0-9]{16}', r'sk-[A-Za-z0-9]{24,}',
                   r'-----BEGIN [A-Z ]*PRIVATE KEY-----']


def public_files(root: Path):
    if (root / '.git').exists():
        raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
        return [root / n for n in raw.decode().split('\0') if n]
    return [p for p in root.rglob('*') if p.is_file() and not any(part in SKIP for part in p.relative_to(root).parts)]


def check(root: Path = ROOT) -> dict:
    errors = []
    paths = public_files(root)
    for p in paths:
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            errors.append(f'Symlink not allowed: {rel}')
            continue
        if p.suffix.lower() in BINARY_SUFFIXES or p.name == '.env' or p.name.startswith('.env.'):
            errors.append(f'Local or sensitive artifact: {rel}')
        if p.stat().st_size > 1_000_000:
            errors.append(f'Unexpected large source file: {rel}')
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            errors.append(f'Non-UTF8 artifact: {rel}')
            continue
        if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
            errors.append(f'Credential-like material: {rel}')
        forbidden = ['sandbox' + ':', '/' + 'mnt/data/', 'file_' + '000000']
        if any(term in text for term in forbidden):
            errors.append(f'Private-runtime reference: {rel}')
        if p.suffix == '.md':
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                target = link.split('#', 1)[0]
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    continue
                if not (p.parent / target).resolve().exists():
                    errors.append(f'Broken relative link: {rel} -> {target}')
    return {'status': 'passed' if not errors else 'failed', 'files_checked': len(paths), 'errors': errors,
            'scope': 'Selected secret patterns, file types, local paths, and relative document links only.'}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'passed' else 1)
