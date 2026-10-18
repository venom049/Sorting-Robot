#!/usr/bin/env python3
"""Check the prepared source snapshot without importing modules or contacting hardware."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    errors = []
    counts = {'original_files': 0, 'archives': 0, 'archive_members': 0,
              'retained_in_archive': 0, 'destination_files': 0,
              'python_files': 0, 'shell_files': 0, 'local_links': 0}

    def fail(message):
        errors.append(message)

    def check_file(relative, expected):
        path = ROOT / relative
        if not path.is_file():
            fail(f'Missing: {relative}')
        elif digest(path.read_bytes()) != expected:
            fail(f'SHA-256 mismatch: {relative}')

    manifest = json.loads((ROOT / 'docs/source-manifest.json').read_text(encoding='utf-8'))
    for original in manifest['original_files']:
        check_file(original['destination'], original['sha256'])
        counts['original_files'] += 1

    for record in manifest['archives']:
        check_file(record['path'], record['sha256'])
        counts['archives'] += 1
        try:
            with zipfile.ZipFile(ROOT / record['path']) as archive:
                bad = archive.testzip()
                if bad:
                    fail(f'ZIP CRC failed: {record["path"]}: {bad}')
                actual = {i.filename for i in archive.infolist() if not i.is_dir()}
                expected = {m['source_path'] for m in record['members']}
                if actual != expected:
                    fail(f'Archive manifest member mismatch: {record["path"]}')
                for info in archive.infolist():
                    path = PurePosixPath(info.filename)
                    if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                        fail(f'Unsafe archive path/type: {info.filename}')
                for member in record['members']:
                    counts['archive_members'] += 1
                    if digest(archive.read(member['source_path'])) != member['source_sha256']:
                        fail(f'Archive member SHA-256 mismatch: {member["source_path"]}')
                    if member['disposition'] == 'retained_in_original_archive':
                        counts['retained_in_archive'] += 1
                    else:
                        check_file(member['destination'], member['destination_sha256'])
                        counts['destination_files'] += 1
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            fail(f'Archive check failed: {exc}')

    source_dirs = ['modules', 'third_party', 'archives/legacy', 'tools']
    for relative in source_dirs:
        for path in sorted((ROOT / relative).rglob('*.py')):
            counts['python_files'] += 1
            try:
                compile(path.read_bytes(), str(path.relative_to(ROOT)), 'exec')
            except (SyntaxError, UnicodeError) as exc:
                fail(f'Python syntax: {exc}')
        for path in sorted((ROOT / relative).rglob('*.sh')):
            counts['shell_files'] += 1
            if shutil.which('bash'):
                result = subprocess.run(['bash', '-n', str(path)], text=True, capture_output=True)
                if result.returncode:
                    fail(f'Shell syntax {path.relative_to(ROOT)}: {result.stderr.strip()}')

    # Check project-authored prose, not untouched upstream historical documentation.
    documents = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for path in documents:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('#', 'http://', 'https://', 'mailto:')):
                continue
            target = unquote(target.split('#', 1)[0])
            counts['local_links'] += 1
            if not (path.parent / target).exists():
                fail(f'Broken local link: {path.relative_to(ROOT)} -> {target}')

    for key, value in counts.items():
        print(f'{key}: {value}')
    if not shutil.which('bash'):
        print('NOTE: Bash unavailable; shell syntax checks skipped.')
    if errors:
        for error in errors:
            print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print('PASS: snapshot integrity, Python syntax, available shell syntax, and project local links.')
    print('Model inference, calibration accuracy, ROS2 APIs, simulation, and hardware were not tested.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
