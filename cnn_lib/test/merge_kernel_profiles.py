#!/usr/bin/env python3
"""Merge the artifacts of a record_kernel_profiles run into the repository.

A profile is pooled from the jobs that share its fingerprint and merged only
once complete; profiles with identical outputs share one directory. Nothing
is committed.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def collect(artifacts):
    """Pool the reference outputs in the artifacts by fingerprint.

    :param artifacts: directory holding the downloaded artifacts
    :return: dictionary mapping a fingerprint to a dictionary mapping an
        output's file name to its content
    """
    found = {}
    for path in Path(artifacts).glob('**/consistency_outputs/*/*.txt'):
        fingerprint, content = path.parent.name, path.read_bytes()
        files = found.setdefault(fingerprint, {})
        # jobs sharing a fingerprint must have recorded the same output
        if files.setdefault(path.name, content) != content:
            sys.exit(
                f'error: {path.name} differs between jobs on {fingerprint} - '
                'the fingerprint is not capturing everything that varies'
            )
    return found


def merge(found, test_dir):
    """Merge the complete profiles into the repository.

    :param found: the reference outputs by fingerprint, as from collect()
    :param test_dir: directory holding kernel_profiles.json and
        consistency_outputs
    """
    outputs_dir = test_dir / 'consistency_outputs'
    profiles_fn = test_dir / 'kernel_profiles.json'
    profiles_json = json.loads(profiles_fn.read_text())
    recorded = profiles_json['profiles']

    # a complete profile has every output of this run and of the repository;
    # a recorded one already has the outputs it lacks here in the repository
    expected = {name for files in found.values() for name in files}
    expected.update(path.name for path in outputs_dir.glob('*/*.txt'))
    complete = {}
    for fingerprint, files in sorted(found.items()):
        if fingerprint in recorded or expected <= files.keys():
            print(f'{fingerprint}: merged')
            complete[fingerprint] = files
        else:
            print(
                f'{fingerprint}: INCOMPLETE ({len(files)} of {len(expected)} '
                'outputs), skipped'
            )
    if not complete:
        sys.exit('error: no complete profile to merge - sample more runners')

    # a new fingerprint gets a directory of its own, a recorded one keeps its
    profiles = {fingerprint: fingerprint for fingerprint in complete} | recorded
    for fingerprint, files in complete.items():
        target = outputs_dir / profiles[fingerprint]
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (target / name).write_bytes(content)

    # a directory identical to an earlier one is removed and its fingerprints
    # pointed at that one
    first, canonical = {}, {}
    for directory in sorted(outputs_dir.iterdir()):
        outputs = frozenset(
            (path.name, path.read_bytes()) for path in directory.glob('*.txt')
        )
        canonical[directory.name] = first.setdefault(outputs, directory.name)
        if canonical[directory.name] != directory.name:
            print(f'{directory.name} duplicates {canonical[directory.name]}')
            shutil.rmtree(directory)

    # profiles whose directory is gone are dropped
    profiles_json['profiles'] = {
        fingerprint: canonical[directory]
        for fingerprint, directory in profiles.items()
        if directory in canonical
    }
    profiles_fn.write_text(
        json.dumps(profiles_json, indent=2, sort_keys=True) + '\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'run', nargs='?', help='workflow run id, the newest by default'
    )
    parser.add_argument(
        '-d', dest='directory', help='already downloaded artifacts'
    )
    args = parser.parse_args()
    test_dir = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory(prefix='kernel-profiles-') as tmp:
        if not args.directory:
            if not args.run:
                args.run = subprocess.check_output(
                    'gh run list --workflow record_kernel_profiles.yml '
                    '--limit 1 --json databaseId --jq .[0].databaseId'.split(),
                    cwd=test_dir,
                    text=True,
                ).strip()
            print(f'Downloading artifacts of run {args.run}')
            subprocess.run(
                ['gh', 'run', 'download', args.run, '-D', tmp],
                cwd=test_dir,
                check=True,
            )
        found = collect(args.directory or tmp)

    merge(found, test_dir)
    print('Done. Review with: git status && git diff --stat')


if __name__ == '__main__':
    main()
