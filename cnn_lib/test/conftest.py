"""Run the consistency tests against the reference outputs of this CPU.

See kernel_profiles.py for why each CPU class has its own reference outputs.
"""

import filecmp
import os
import shutil
from pathlib import Path

import pytest

from kernel_profiles import find_profile


def _store(produced, stored, *args, **kwargs):
    """Replace filecmp.cmp: store the output instead of comparing it.

    :param produced: path to the output the test produced
    :param stored: path to store it at
    :param args: further arguments of filecmp.cmp, ignored
    :param kwargs: further keyword arguments of filecmp.cmp, ignored
    :return: True, so that the test passes
    """
    shutil.copyfile(produced, stored)
    return True


def pytest_addoption(parser):
    """Add the --record-profile option to pytest.

    :param parser: pytest's command line parser
    """
    parser.addoption(
        '--record-profile',
        action='store_true',
        help="record this CPU's reference outputs instead of comparing them",
    )


@pytest.fixture(scope='session', autouse=True)
def kernel_profile(pytestconfig):
    """Point the tests at the reference outputs recorded for this CPU.

    :param pytestconfig: pytest's configuration, holding --record-profile
    """
    record = pytestconfig.getoption('record_profile')
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
        filecmp.cmp = _store
