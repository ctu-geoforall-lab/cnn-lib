"""Keep the byte-exact consistency comparison valid on changing CI hardware.

The comparison in consistency_test.py is only meaningful on a machine whose
float32 kernels accumulate in the same order as the machine the reference
outputs were recorded on, and that is not something a seed,
``enable_op_determinism()`` or a thread limit can give across different CPUs:
TF's CPU convolutions go through Eigen, whose GEMM blocking factor is derived
at runtime from the L1 data cache size reported by CPUID (32 KiB up to Cascade
Lake and on Zen, 48 KiB from Ice Lake on). A different split means a different
summation order, which means a few ULP - and the printed value is the
inference-mode val_loss of a BatchNorm net at batch_size=1, which turns 1 ULP
into ~10%.

So each distinct kernel path gets its own set of reference outputs, and a
machine whose path has not been recorded skips instead of failing at random:

    CNN_LIB_RECORD_PROFILE=1 pytest ...   record this machine's profile and
                                          write its reference outputs
    CNN_LIB_REQUIRE_PROFILE=1 pytest ...  fail instead of skipping

Each profile gets consistency_outputs/<fingerprint>/ to itself, so recordings
made by separate CI jobs merge by copying. Which profiles GitHub's runner pool
can hand out is not published and changes over time, so they are sampled
empirically - see .github/workflows/record_kernel_profiles.yml

Rebuilding the image against a different TensorFlow or numpy changes the
fingerprint as well. That is intended: the reference outputs change with it,
so every profile has to be re-recorded anyway.

To have the tests always run, use fixed hardware - a self-hosted runner.
"""

import os
import json
import shutil
import filecmp
import hashlib

import numpy as np
import pytest
import tensorflow as tf


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(TEST_DIR, 'consistency_outputs')
PROFILES_FN = os.path.join(TEST_DIR, 'kernel_profiles.json')

# Keras renders model.summary() through rich, which takes its width from
# COLUMNS; without pinning it the stored outputs only match in an 80-column
# environment
os.environ['COLUMNS'] = '80'


def _enabled(name):
    """Read a boolean switch from the environment.

    :param name: name of the environment variable
    :return: whether the switch is on
    """
    return os.environ.get(name, '') not in ('', '0', 'false', 'False')


def kernel_fingerprint():
    """Fingerprint the float32 kernel path of the current machine.

    A matmul big enough for Eigen to cut into blocks, plus a convolution,
    hashed over their raw float32 bytes. Two machines share a fingerprint
    only if they accumulate in the same order.

    :return: short hexadecimal fingerprint of the kernel path
    """
    rng = np.random.default_rng(0)
    gemm = tf.linalg.matmul(
        tf.constant(rng.standard_normal((256, 2048), dtype=np.float32)),
        tf.constant(rng.standard_normal((2048, 256), dtype=np.float32)),
    )
    conv = tf.nn.conv2d(
        tf.constant(rng.standard_normal((1, 64, 64, 32), dtype=np.float32)),
        tf.constant(rng.standard_normal((3, 3, 32, 32), dtype=np.float32)),
        strides=1,
        padding='SAME',
    )

    digest = hashlib.sha256()
    for tensor in (gemm, conv):
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()[:16]


def _read_profiles():
    """Read the recorded kernel profiles.

    :return: dictionary mapping a fingerprint to a directory name relative
        to consistency_outputs
    """
    if not os.path.isfile(PROFILES_FN):
        return {}

    with open(PROFILES_FN) as profiles_file:
        return json.load(profiles_file)['profiles']


def _write_profiles(profiles):
    """Store the recorded kernel profiles.

    :param profiles: dictionary mapping a fingerprint to a directory name
    """
    with open(PROFILES_FN, 'w') as profiles_file:
        json.dump(
            {
                'comment': 'Fingerprints of the float32 kernel paths the '
                'reference outputs were recorded on, mapped to their '
                'directory under consistency_outputs. See conftest.py.',
                'profiles': profiles,
            },
            profiles_file,
            indent=2,
            sort_keys=True,
        )
        profiles_file.write('\n')


@pytest.fixture(scope='session', autouse=True)
def known_kernel_profile():
    """Point the tests at the reference outputs valid for this machine."""
    # mirror train.run(), so the fingerprint describes the configuration the
    # tests themselves run under
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    fingerprint = kernel_fingerprint()
    profiles = _read_profiles()

    if fingerprint not in profiles:
        if not _enabled('CNN_LIB_RECORD_PROFILE'):
            message = (
                f'This runner computes float32 differently (kernel profile '
                f'{fingerprint}, known: {sorted(profiles) or "none"}), so a '
                'byte-exact comparison against the stored outputs would say '
                'nothing about the code. Record this machine with '
                'CNN_LIB_RECORD_PROFILE=1, or run on fixed hardware.'
            )
            if _enabled('CNN_LIB_REQUIRE_PROFILE'):
                pytest.fail(message, pytrace=False)
            pytest.skip(message)

        # always a directory of its own, so that recordings made by separate
        # CI jobs merge by copying, without any of them claiming a shared
        # directory. Two profiles that turn out to produce identical outputs
        # can afterwards be pointed at the same directory by hand.
        profiles[fingerprint] = fingerprint
        _write_profiles(profiles)

    outputs_dir = os.path.join(OUTPUTS_DIR, profiles[fingerprint])
    os.environ['CNN_LIB_OUTPUTS_DIR'] = outputs_dir

    if _enabled('CNN_LIB_RECORD_PROFILE'):
        os.makedirs(outputs_dir, exist_ok=True)
        print(f'\nRecording kernel profile {fingerprint} into {outputs_dir}')

        # there is nothing to compare against while recording, so store the
        # produced output instead of comparing it
        def record(produced, expected, *args, **kwargs):
            shutil.copyfile(produced, expected)
            return True

        filecmp.cmp = record
