"""
v1 integrity: nothing that defines the finished v1 platform has changed.

Plain English: compares today's files with the fingerprint taken at tag
v1.0-evaluated (v2/manifests/v1_manifest.json). The repo check runs
everywhere, including CI; the full check (database, MLflow, git-ignored
model files) runs only where they exist.
"""

import pytest

from conftest import DB_AVAILABLE, LOCAL_PACKAGE, load_module

manifest = load_module("v1_manifest", "v2/tools/v1_manifest.py")


def test_committed_v1_artefacts_unchanged(capsys):
    manifest.verify(full=False)
    assert "v1 unchanged." in capsys.readouterr().out


@pytest.mark.skipif(not (DB_AVAILABLE and LOCAL_PACKAGE), reason="needs the local database and model package")
def test_full_v1_unchanged(capsys):
    manifest.verify(full=True)
    assert "v1 unchanged." in capsys.readouterr().out
