"""The file-mutation identity must be one key per on-disk target, including skills.

A turn that tries ``save_skill``-style writes as ``<skill-name>/SKILL.md`` resolves
under the task cwd, where no such directory exists, so the attempt is recorded
against a phantom path. The skill-managed write that later lands at the real
``<skills-root>/<skill-name>/SKILL.md`` then carries a different identity and the
turn keeps a stale FAILED footer for a mutation that succeeded.

These tests are hermetic: the skills root comes from a ``HERMES_HOME`` pointed at
``tmp_path`` and the task cwd is a separate ``tmp_path`` directory, so nothing
reads the operator's real ``~/.hermes``.
"""

import pytest

from agent.turn_explainers import _file_mutation_identity


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """A throwaway ``HERMES_HOME`` with an empty ``skills/`` root."""
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    return home


@pytest.fixture()
def real_skill(isolated_home, tmp_path, monkeypatch):
    """A skill on disk under the isolated home, with the task cwd somewhere else."""
    skill_dir = isolated_home / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\nname: demo-skill\n---\nbody\n", encoding="utf-8")
    cwd = tmp_path / "task-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return skill_md


def test_bare_skill_spelling_and_real_path_share_one_identity(real_skill):
    """The bare spelling and the resolved path must meet on one identity."""
    bare = _file_mutation_identity("demo-skill/SKILL.md", "default")
    real = _file_mutation_identity(str(real_skill), "default")
    assert bare == real


def test_bare_skill_spelling_stays_task_relative_when_no_skill_matches(real_skill):
    """No matching skill dir anywhere means the attempt keeps its task-cwd identity."""
    bare = _file_mutation_identity("not-a-skill/SKILL.md", "default")
    real = _file_mutation_identity(str(real_skill), "default")
    assert bare != real
    assert "not-a-skill" in bare


def test_hoist_requires_the_first_segment_to_be_a_directory(real_skill):
    """A *file* named like the first segment must not anchor a hoist.

    ``<root>/evil`` shadowing a skill is the classic shape: a plain file at a
    skills root would otherwise capture every ``evil/...`` attempt.
    """
    shadow = real_skill.parent.parent / "evil"
    shadow.write_text("not a skill dir\n", encoding="utf-8")
    identity = _file_mutation_identity("evil/SKILL.md", "default")
    assert identity != _file_mutation_identity(str(shadow), "default")
    assert str(shadow.parent) not in identity


def test_absolute_path_is_never_repointed(real_skill):
    """An absolute miss stays exactly where the caller pointed it."""
    missing = real_skill.parent / "missing" / "SKILL.md"
    identity = _file_mutation_identity(str(missing), "default")
    assert identity.endswith("SKILL.md")
    assert "missing" in identity
