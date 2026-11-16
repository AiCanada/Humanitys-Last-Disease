"""The BioIR capture check must not treat reusable allocator cache as occupied."""

from types import SimpleNamespace

import torch

from rfproteina.nn.bioir_full import _reclaim_cached_blocks_before_rejecting_capture


def test_rejected_capture_reclaims_cached_blocks_and_checks_again(monkeypatch):
    seen = {"checks": 0, "reclaims": 0}

    def check(working_set_bytes, input_container=None):
        seen["checks"] += 1
        if seen["checks"] == 1:
            return SimpleNamespace(ok=False, reason="insufficient GPU memory")
        return SimpleNamespace(ok=True, reason="ok")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.__setitem__("reclaims", seen["reclaims"] + 1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    result = _reclaim_cached_blocks_before_rejecting_capture(check)(128)

    assert result.ok
    assert seen["reclaims"] == 1
    assert seen["checks"] == 2


def test_accepted_capture_does_not_reclaim(monkeypatch):
    seen = {"reclaims": 0}

    def check(working_set_bytes, input_container=None):
        return SimpleNamespace(ok=True, reason="ok")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.__setitem__("reclaims", seen["reclaims"] + 1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    result = _reclaim_cached_blocks_before_rejecting_capture(check)(128)

    assert result.ok
    assert seen["reclaims"] == 0


def test_reclaim_waits_while_a_stream_is_capturing(monkeypatch):
    seen = {"checks": 0, "reclaims": 0}

    def check(working_set_bytes, input_container=None):
        seen["checks"] += 1
        return SimpleNamespace(ok=False, reason="insufficient GPU memory")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.__setitem__("reclaims", seen["reclaims"] + 1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    result = _reclaim_cached_blocks_before_rejecting_capture(check)(128)

    assert not result.ok
    assert seen["reclaims"] == 0
    assert seen["checks"] == 1
