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

TEST_DIR = Path(__file__).resolve().parent
OUT = TEST_DIR / 'consistency_outputs'
PROFILES_FN = TEST_DIR / 'kernel_profiles.json'


def download(run, directory):
    """Download a run's artifacts with the gh CLI.

    :param run: run id, or None for the newest run of the workflow
    :param directory: directory to download the artifacts into
    """
    if run is None:
        run = subprocess.check_output(
            'gh run list --workflow record_kernel_profiles.yml --limit 1 '
            '--json databaseId --jq .[0].databaseId'.split(),
            cwd=TEST_DIR,
            text=True,
        ).strip()

    print(f'Downloading artifacts of run {run}')
    subprocess.run(
        ['gh', 'run', 'download', run, '-D', directory],
        cwd=TEST_DIR,
        check=True,
    )


def read(directory):
    """Read the reference outputs stored in a directory.

    :param directory: directory to read
    :return: dictionary mapping a file name to its content
    """
    return {path.name: path.read_bytes() for path in directory.glob('*.txt')}


def collect(artifacts):
    """Pool the reference outputs in the artifacts by fingerprint.

    Jobs sharing a fingerprint record different test functions, but an
    output recorded by more than one of them has to be the same in all.

    :param artifacts: directory holding the downloaded artifacts
    :return: dictionary mapping a fingerprint to its outputs, as read()
    """
    found = {}
    for path in Path(artifacts).glob('**/consistency_outputs/*/*.txt'):
        fingerprint, content = path.parent.name, path.read_bytes()
        files = found.setdefault(fingerprint, {})
        if files.setdefault(path.name, content) != content:
            sys.exit(
                f'error: {path.name} differs between jobs on {fingerprint} - '
                'the fingerprint is not capturing everything that varies'
            )
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'run', nargs='?', help='workflow run id, the newest by default'
    )
    parser.add_argument('-d', dest='directory', help='downloaded artifacts')
    parser.add_argument(
        '--partial',
        action='store_true',
        help='let an already recorded profile be updated with only some of '
        'its reference outputs, for a rerecord_outputs run',
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix='kernel-profiles-') as tmp:
        if not args.directory:
            download(args.run, tmp)
        found = collect(args.directory or tmp)

    # a fingerprint sharing another's directory never records - its runners
    # stop at the probe - so it is known only from here
    profiles_json = json.loads(PROFILES_FN.read_text())
    recorded = profiles_json['profiles']

    # every output of this run plus every output already in the repository:
    # a test function that ran on no new CPU in this run is still required
    expected = {name for files in found.values() for name in files}
    expected.update(path.name for path in OUT.glob('*/*.txt'))

    print(f'Checking completeness against {len(expected)} reference outputs:')
    complete = {}
    for fingerprint, files in sorted(found.items()):
        # a rerecord run carries only the outputs it was asked to redo; the
        # rest of an already recorded profile is in the repository
        if expected <= files.keys() or args.partial and fingerprint in recorded:
            complete[fingerprint] = files
        print(
            f'  {fingerprint}: {len(files)}/{len(expected)}'
            + ('' if fingerprint in complete else ' - INCOMPLETE, skipped')
        )
    if not complete:
        sys.exit('error: no complete profile to merge - sample more runners')

    # a new fingerprint gets a directory of its own, a recorded one keeps its
    profiles = {fingerprint: fingerprint for fingerprint in complete} | recorded
    for fingerprint, files in complete.items():
        target = OUT / profiles[fingerprint]
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (target / name).write_bytes(content)

    # a directory byte-identical to an earlier one is removed and its
    # fingerprints pointed at that one
    first, canonical = {}, {}
    for directory in sorted(set(profiles.values())):
        if not (OUT / directory).is_dir():
            print(f'warning: {directory} is gone, dropping its profiles')
            continue
        content = frozenset(read(OUT / directory).items())
        canonical[directory] = first.setdefault(content, directory)
        if canonical[directory] != directory:
            print(f'{directory} duplicates {canonical[directory]}, removed')
            shutil.rmtree(OUT / directory)

    profiles_json['profiles'] = {
        fingerprint: canonical[directory]
        for fingerprint, directory in profiles.items()
        if directory in canonical
    }
    PROFILES_FN.write_text(
        json.dumps(profiles_json, indent=2, sort_keys=True) + '\n'
    )

    print('Done. Review with: git status && git diff --stat')


if __name__ == '__main__':
    main()
