"""Test helper: make temp directories writable inside restricted Windows environments.

Python's tempfile.mkdtemp() creates directories with mode 0o700; some sandboxed Windows setups
deny writes inside such a directory (only directories with inherited rights are writable), so
tests would fail with "unable to open database file" for a reason that has nothing to do with the
code. The workaround is only installed when that problem is actually observed; elsewhere mkdtemp is
untouched, so CI and ordinary machines keep the standard behaviour.
"""
import os
import shutil
import tempfile
import uuid

_original_mkdtemp = tempfile.mkdtemp


def _writable(path: str) -> bool:
    probe = os.path.join(path, ".write-probe")
    try:
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def _mkdtemp(suffix=None, prefix=None, dir=None):
    base = dir or tempfile.gettempdir()
    while True:
        path = os.path.join(base, f"{prefix or 'tmp'}{uuid.uuid4().hex[:10]}{suffix or ''}")
        try:
            os.mkdir(path)                      # default mode: inherits the sandbox's write rights
            return path
        except FileExistsError:
            continue


def _install_if_needed():
    try:
        probe = _original_mkdtemp()
    except OSError:
        tempfile.mkdtemp = _mkdtemp
        return
    if not _writable(probe):
        tempfile.mkdtemp = _mkdtemp
    shutil.rmtree(probe, ignore_errors=True)


_install_if_needed()
