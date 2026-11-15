"""Compare the consistency tests' outputs with those recorded on the same CPU.

The tests compare training outputs byte by byte, but float32 results depend on
the CPU: TF derives its kernels' blocking, and so the summation order, from the
CPU's cache sizes. Each CPU class therefore has its own reference outputs (a
kernel profile), identified by a fingerprint of the CPU and of a few TF
results. kernel_profiles.json maps fingerprints to directories in
consistency_outputs.

On a CPU without a profile the tests fail; CNN_LIB_RECORD_PROFILE=1 records
its outputs instead. `python3 conftest.py` prints the fingerprint and whether
it is recorded. See .github/workflows/record_kernel_profiles.yml.
"""

import filecmp
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


def _read(path):
    """Read a file, or '?' if it is not there."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return '?'


def _cpu_identity():
    """Describe the CPU model and its caches.

    Needed besides the numbers: two CPUs can agree on the fingerprint's
    matmul and convolution, but not on the models the tests train.
    """
    cpuinfo = {}
    for line in _read('/proc/cpuinfo').splitlines():
        key, _, value = line.partition(':')
        cpuinfo.setdefault(key.strip(), value.strip())

    fields = ('vendor_id', 'cpu family', 'model', 'model name', 'stepping')
    caches = [
        _read(f'/sys/devices/system/cpu/cpu0/cache/index{index}/size')
        for index in range(4)
    ]
    return '|'.join([cpuinfo.get(field, '?') for field in fields] + caches)


def kernel_fingerprint():
    """Hash the CPU identity with the bytes of a matmul and a convolution.

    :return: 16 hexadecimal digits identifying the kernel profile
    """
    rng = np.random.default_rng(0)

    def random(*shape):
        return tf.constant(rng.standard_normal(shape, dtype=np.float32))

    gemm = tf.linalg.matmul(random(256, 2048), random(2048, 256))
    conv = tf.nn.conv2d(
        random(1, 64, 64, 32), random(3, 3, 32, 32), strides=1, padding='SAME'
    )

    digest = hashlib.sha256(_cpu_identity().encode())
    digest.update(gemm.numpy().tobytes())
    digest.update(conv.numpy().tobytes())
    return digest.hexdigest()[:16]


def find_profile(test_dir, record):
    """Find the reference outputs recorded for this CPU.

    :param test_dir: directory holding kernel_profiles.json and
        consistency_outputs
    :param record: whether to register this CPU if it is not recorded yet
    :return: this CPU's fingerprint, and the directory holding its reference
        outputs or None if it is not recorded
    """
    # the tests run under these settings, so the fingerprint is taken under them
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    fingerprint = kernel_fingerprint()

    profiles_fn = test_dir / 'kernel_profiles.json'
    profiles_json = json.loads(profiles_fn.read_text())
    profiles = profiles_json['profiles']

    if fingerprint not in profiles:
        if not record:
            return fingerprint, None
        profiles[fingerprint] = fingerprint
        profiles_fn.write_text(
            json.dumps(profiles_json, indent=2, sort_keys=True) + '\n'
        )

    return fingerprint, test_dir / 'consistency_outputs' / profiles[fingerprint]


@pytest.fixture(scope='session', autouse=True)
def kernel_profile():
    """Point the tests at the reference outputs recorded for this CPU."""
    record = os.environ.get('CNN_LIB_RECORD_PROFILE') == '1'
    fingerprint, outputs_dir = find_profile(Path(__file__).parent, record)

    if outputs_dir is None:
        pytest.fail(
            f'Kernel profile {fingerprint} is not recorded, so there is '
            'nothing to compare with. Record it with the '
            'record_kernel_profiles workflow and merge_kernel_profiles.py.',
            pytrace=False,
        )

    # the stored outputs contain model.summary(), which wraps at COLUMNS
    os.environ['COLUMNS'] = '80'
    os.environ['CNN_LIB_OUTPUTS_DIR'] = str(outputs_dir)

    if record:
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # store each output instead of comparing it
        def store(produced, stored, *args, **kwargs):
            shutil.copyfile(produced, stored)
            return True

        filecmp.cmp = store


def main():
    """Print this CPU's fingerprint and whether it is recorded.

    Lets record_kernel_profiles.yml skip the CPUs already recorded.
    """
    fingerprint, outputs_dir = find_profile(Path(__file__).parent, False)
    print(fingerprint, 'unknown' if outputs_dir is None else 'known')


if __name__ == '__main__':
    main()
