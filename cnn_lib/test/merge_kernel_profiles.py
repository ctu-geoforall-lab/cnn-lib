#!/usr/bin/env python3
"""Merge the artifacts of a record_kernel_profiles run into the repository.

    ./merge_kernel_profiles.py                 # newest run of the workflow
    ./merge_kernel_profiles.py <run-id>        # a specific run
    ./merge_kernel_profiles.py -d <directory>  # artifacts already downloaded

Run it from the root of the cnn-lib checkout you want to update. Nothing is
committed - inspect `git status` afterwards.

Each sampling job records one test function, so a profile is assembled from
several jobs that happened to land on CPUs with the same fingerprint. A
profile is merged only once every reference output is present; an incomplete
one would make the consistency test fail on a missing file instead of
skipping.
"""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

OUT = os.path.join('cnn_lib', 'test', 'consistency_outputs')
PROFILES_FN = os.path.join('cnn_lib', 'test', 'kernel_profiles.json')


def fail(message):
    """Abort with a message.

    :param message: text to print on standard error
    """
    print(f'error: {message}', file=sys.stderr)
    raise SystemExit(1)


def download(run):
    """Download a run's artifacts with the gh CLI.

    :param run: run id, or None for the newest run of the workflow
    :return: directory the artifacts were downloaded into
    """
    if run is None:
        run = subprocess.run(
            [
                'gh', 'run', 'list', '--workflow', 'record_kernel_profiles.yml',
                '--limit', '1', '--json', 'databaseId',
                '--jq', '.[0].databaseId',
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    directory = tempfile.mkdtemp(prefix='kernel-profiles-')
    print(f'Downloading artifacts of run {run} into {directory}')
    subprocess.run(['gh', 'run', 'download', run, '-D', directory], check=True)

    return directory


def collect(artifacts):
    """Find every recorded profile directory in the downloaded artifacts.

    :param artifacts: directory holding the downloaded artifacts
    :return: dictionary mapping a fingerprint to the list of directories
        recorded for it
    """
    found = {}

    for root, dirs, _ in os.walk(artifacts):
        if os.path.basename(root) != 'consistency_outputs':
            continue
        for fingerprint in dirs:
            found.setdefault(fingerprint, []).append(
                os.path.join(root, fingerprint)
            )

    return found


def check_agreement(fingerprint, directories):
    """Check that jobs sharing a fingerprint recorded identical outputs.

    Different jobs record different test functions, so only the files present
    in both of a pair are compared.

    :param fingerprint: the kernel fingerprint
    :param directories: directories recorded for that fingerprint
    :return: whether every overlapping file agrees
    """
    agreed = True
    compared = 0

    for i, left in enumerate(directories):
        for right in directories[i + 1:]:
            shared = set(os.listdir(left)) & set(os.listdir(right))
            for name in sorted(shared):
                compared += 1
                if not filecmp.cmp(
                    os.path.join(left, name),
                    os.path.join(right, name),
                    shallow=False,
                ):
                    print(f'  {fingerprint}: {name} DIFFERS between jobs')
                    agreed = False

    if agreed:
        print(
            f'  {fingerprint}: {compared} overlapping file(s) agree'
            if compared
            else f'  {fingerprint}: no overlap between jobs, nothing to check'
        )

    return agreed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', help='workflow run id')
    parser.add_argument('-d', dest='directory', help='downloaded artifacts')
    args = parser.parse_args()

    if not os.path.isdir(OUT):
        fail('run this from the root of the cnn-lib checkout')

    artifacts = args.directory or download(args.run)
    found = collect(artifacts)

    if not found:
        fail(f'no consistency_outputs/<fingerprint>/ directories in {artifacts}')

    print('\nFingerprints in this run (jobs each):')
    for fingerprint, directories in sorted(found.items()):
        print(f'  {fingerprint}: {len(directories)}')

    print('\nChecking that jobs sharing a fingerprint agree:')
    if not all(
        check_agreement(fingerprint, directories)
        for fingerprint, directories in sorted(found.items())
    ):
        fail('the fingerprint is not capturing everything that varies')

    expected = {
        name
        for directories in found.values()
        for directory in directories
        for name in os.listdir(directory)
        if name.endswith('.txt')
    }
    print(
        f'\nChecking completeness against the {len(expected)} distinct '
        'reference outputs seen:'
    )

    complete = {}
    for fingerprint, directories in sorted(found.items()):
        names = {
            name
            for directory in directories
            for name in os.listdir(directory)
            if name.endswith('.txt')
        }
        missing = sorted(expected - names)
        if missing:
            print(
                f'  {fingerprint}: INCOMPLETE, missing {len(missing)} '
                f'({", ".join(missing[:3])}{" ..." if len(missing) > 3 else ""})'
            )
        else:
            print(f'  {fingerprint}: complete')
            complete[fingerprint] = directories

    if not complete:
        fail('no complete profile to merge - sample more runners')

    # copy the recorded outputs in before removing the old flat ones, so the
    # directory never becomes empty (git would drop it)
    print(f'\nCopying {len(complete)} complete profile(s) into {OUT}/')
    for fingerprint, directories in complete.items():
        target = os.path.join(OUT, fingerprint)
        os.makedirs(target, exist_ok=True)
        for directory in directories:
            for name in os.listdir(directory):
                if name.endswith('.txt'):
                    shutil.copyfile(
                        os.path.join(directory, name),
                        os.path.join(target, name),
                    )

    stale = [
        os.path.join(OUT, name)
        for name in os.listdir(OUT)
        if name.endswith('.txt')
    ]
    if stale:
        print(f'Removing {len(stale)} stale flat reference output(s)')
        subprocess.run(['git', 'rm', '-q'] + stale, check=True)

    fingerprints = sorted(
        name for name in os.listdir(OUT)
        if os.path.isdir(os.path.join(OUT, name))
    )
    with open(PROFILES_FN, 'w') as profiles_file:
        json.dump(
            {
                'comment': 'Fingerprints of the float32 kernel paths the '
                'reference outputs were recorded on, mapped to their '
                'directory under consistency_outputs. See conftest.py.',
                'profiles': {fp: fp for fp in fingerprints},
            },
            profiles_file,
            indent=2,
            sort_keys=True,
        )
        profiles_file.write('\n')

    print(f'\n{PROFILES_FN} now lists {len(fingerprints)} profile(s):')
    for fingerprint in fingerprints:
        print(f'  {fingerprint}')

    print('\nProfiles whose outputs are identical (one directory would do):')
    duplicates = [
        f'  {a} == {b}'
        for i, a in enumerate(fingerprints)
        for b in fingerprints[i + 1:]
        if not filecmp.dircmp(
            os.path.join(OUT, a), os.path.join(OUT, b)
        ).diff_files
        and not filecmp.dircmp(
            os.path.join(OUT, a), os.path.join(OUT, b)
        ).left_only
    ]
    print('\n'.join(duplicates) if duplicates else '  none - every profile differs')

    if not args.directory:
        shutil.rmtree(artifacts, ignore_errors=True)

    print('\nDone. Review with: git status && git diff --stat')


if __name__ == '__main__':
    main()
