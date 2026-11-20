"""Find the reference outputs of the consistency tests recorded on this CPU.

The tests compare training outputs byte by byte, but float32 results depend on
the CPU: TF derives its kernels' blocking, and so the summation order, from the
CPU's cache sizes. Each CPU class therefore has its own reference outputs (a
kernel profile), identified by a fingerprint of the CPU and of a few TF
results. kernel_profiles.json maps fingerprints to directories in
consistency_outputs.

On a CPU without a profile the tests fail; `pytest --record-profile` records
its outputs instead (see conftest.py). Run as a script, this prints the
fingerprint and whether it is recorded. See
.github/workflows/record_kernel_profiles.yml.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _read(path):
    """Read a file.

    :param path: path to the file
    :return: the file's stripped content, or '?' if it cannot be read
    """
    try:
        return Path(path).read_text().strip()
    except OSError:
        return '?'


def _cpu_identity():
    """Describe the CPU model and its caches.

    Needed besides the numbers: two CPUs can agree on the fingerprint's
    matmul and convolution, but not on the models the tests train.

    :return: the CPU's vendor, family, model, name and stepping and the sizes
        of its caches, joined by '|'
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
    shapes = ((256, 2048), (2048, 256), (1, 64, 64, 32), (3, 3, 32, 32))
    left, right, image, filters = (
        tf.constant(rng.standard_normal(shape, dtype=np.float32))
        for shape in shapes
    )

    gemm = tf.linalg.matmul(left, right)
    conv = tf.nn.conv2d(image, filters, strides=1, padding='SAME')

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


def main():
    """Print this CPU's fingerprint and whether it is recorded.

    Lets record_kernel_profiles.yml skip the CPUs already recorded.
    """
    fingerprint, outputs_dir = find_profile(Path(__file__).parent, False)
    print(fingerprint, 'unknown' if outputs_dir is None else 'known')


if __name__ == '__main__':
    main()
