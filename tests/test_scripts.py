"""Smoke test: every script's argument parser builds and parses its documented flags."""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_parser_builds(script):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    args = parser.parse_args([])
    assert hasattr(args, "repo_root")
    assert parser.parse_args(["--stage", "2", "--stage", "4"]).stage == [2, 4]
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "9"])
