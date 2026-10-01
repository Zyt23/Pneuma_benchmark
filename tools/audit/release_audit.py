"""Read-only syntax and accidental-secret audit of a release source tree."""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    ignored = {'.git', '__pycache__', '.pytest_cache', 'checkpoints', 'test_results', 'results', 'logs', '.venv'}
    errors, warnings = [], []
    python_count = shell_count = 0
    tokens = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\b(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{20,}')
    private_path = re.compile(r'/share/home_migration/|C:[\\/]Users[\\/][^\\/\s]+', re.I)
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        relative = str(path.relative_to(root))
        if path.suffix not in {'.py', '.sh', '.md', '.txt', '.json', '.toml', '.yaml', '.yml',
                              '.java', '.csv', '.tsv', '.ps1', '.cfg', '.ini'} and path.name not in {'LICENSE', '.gitignore'}:
            continue
        text = path.read_text(encoding='utf-8-sig')
        if tokens.search(text):
            errors.append(dict(path=relative, kind='possible_secret'))
        if private_path.search(text) and path.name != 'release_audit.py':
            warnings.append(dict(path=relative, kind='private_absolute_path'))
        if path.suffix == '.py':
            python_count += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as exc:
                errors.append(dict(path=relative, kind='python_syntax', line=exc.lineno))
        if path.suffix == '.sh':
            shell_count += 1
            try:
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(dict(path=relative, kind='bash_syntax', detail=result.stderr))
            except FileNotFoundError:
                warnings.append(dict(path=relative, kind='bash_not_available'))
    report = dict(python_files=python_count, shell_files=shell_count, errors=errors, warnings=warnings,
                  scope='syntax and recognizable secret/path patterns only; not proof of license or full privacy compliance')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
