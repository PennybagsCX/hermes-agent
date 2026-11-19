"""Stale same-vendor endpoint pins on persisted session routes must not outvote config.

Gateway sessions that ran a built-in provider DIRECT before a proxy/gate was configured kept
``base_url=https://<vendor>/...`` in their persisted runtime route. Resume/rehydrate re-applied
the pin verbatim (``gateway_runtime`` in the session row's ``model_config``), so those sessions
bypassed the new endpoint forever. Config (``model.base_url`` / ``model.providers.<p>.base_url``)
owns endpoint routing for a built-in provider; a persisted pin of the SAME vendor host at a
different URL is a pre-route-move relic and must adopt the configured endpoint.

Covers:
  - ``heal_stale_vendor_pinned_base_url`` unit behavior (heal / keep / noop cases)
  - ``GatewayAgentCacheMixin._rehydrate_session_model_override`` adopting the configured
    endpoint for a persisted direct pin
"""
from types import SimpleNamespace

import pytest

# Import gateway.run at collection time: its payload/environment discovery reads a
# manifest beside the checkout, which the per-test real-home I/O guard forbids once a
# test is running (tests/home_io_guard.py).
import gateway.run  # monkeypatch target module (see the comment above)
from gateway.run_agent_cache import GatewayAgentCacheMixin

GATE = "http://127.0.0.1:8791"
ZAI_DIRECT = "https://ai.z.ai/api/coding/paas/v4".replace("ai.z", "api.z")


# ---------------------------------------------------------------------------
# heal_stale_vendor_pinned_base_url — unit behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def gate_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "zai", "default": "glm-5.3-flash", "base_url": GATE})


def test_direct_pin_heals_to_configured_gate(gate_config):
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", ZAI_DIRECT) == GATE


def test_trailing_slash_variant_heals(gate_config):
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", ZAI_DIRECT + "/") == GATE


def test_providers_table_entry_wins(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "openai", "providers": {"zai": {"base_url": GATE}}})
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", ZAI_DIRECT) == GATE


def test_custom_proxy_pin_is_deliberate(gate_config):
    """A different host is a deliberate pin (enterprise proxy, region gateway): kept."""
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    kept = "https://my-proxy.example/v1"
    assert heal_stale_vendor_pinned_base_url("zai", kept) == kept


def test_same_vendor_surface_change_is_never_healed(monkeypatch):
    """api.z.ai hosts separately-billed surfaces (/api/paas/v4 general, /api/coding/paas/v4
    coding); a pin on one surface while config declares the other is a platform selection,
    not a pre-gate relic — the relic signature requires a NON-vendor configured endpoint."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "zai", "base_url": "https://api.z.ai/api/paas/v4"})
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", ZAI_DIRECT) == ZAI_DIRECT


def test_loopback_canonical_provider_pins_never_heal(monkeypatch):
    """lmstudio's canonical host IS loopback: a pin on a second local instance (:1235) is
    deliberate and must survive a config that declares the supervisor port (:1234)."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "lmstudio", "base_url": "http://127.0.0.1:1234/v1"})
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url(
        "lmstudio", "http://127.0.0.1:1235/v1") == "http://127.0.0.1:1235/v1"


def test_deliberate_proxy_pin_kept_when_config_goes_vendor(monkeypatch):
    """Asymmetry: only vendor-host pins heal. A pin on a non-vendor host (the gate/proxy)
    is never rewritten back to the vendor even if config changes."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "zai", "base_url": ZAI_DIRECT})
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", GATE) == GATE


def test_no_configured_endpoint_keeps_pin(monkeypatch):
    """No config-declared endpoint for the provider -> nothing to heal to; pin kept."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config", lambda: {"provider": "zai"})
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", ZAI_DIRECT) == ZAI_DIRECT


def test_same_url_is_noop(gate_config):
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", GATE) == GATE


def test_unregistered_provider_untouched(gate_config):
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    kept = "https://weird-endpoint.example/v9"
    assert heal_stale_vendor_pinned_base_url("not-a-provider", kept) == kept


def test_empty_pin_untouched(gate_config):
    from hermes_cli.runtime_provider import heal_stale_vendor_pinned_base_url
    assert heal_stale_vendor_pinned_base_url("zai", "") == ""


# ---------------------------------------------------------------------------
# Gateway rehydrate integration — the persisted-pin path end to end
# ---------------------------------------------------------------------------

def _make_runner(states):
    class _Conv:
        model_override = None

    class _State:
        def __init__(self):
            self.conversation = _Conv()

    class _Runner(GatewayAgentCacheMixin):
        def _session_model_override(self, session_key):
            return None  # nothing in memory: force the persisted-read path

        def _peek_session_state(self, session_key):
            return states.get(session_key)

        def _session_state(self, session_key):
            return states.setdefault(session_key, _State())

    runner = _Runner()
    runner.session_store = SimpleNamespace(get_model_override=lambda key: {
        "model": "glm-5.3-flash", "provider": "zai", "base_url": ZAI_DIRECT})
    return runner


def _patch_runtime_resolution(monkeypatch, base_url=GATE):
    monkeypatch.setattr(
        gateway.run, "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider, target_model=None: {
            "api_key": "k", "api_mode": "chat_completions", "base_url": base_url,
            "request_overrides": {}, "capabilities": {}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config",
        lambda: {"provider": "zai", "default": "glm-5.3-flash", "base_url": GATE})


def test_rehydrate_adopts_configured_endpoint(monkeypatch):
    """A session persisted on the direct endpoint rehydrates onto the configured gate."""
    states = {}
    runner = _make_runner(states)
    _patch_runtime_resolution(monkeypatch)

    runner._rehydrate_session_model_override("direct-pin-thread")

    override = states["direct-pin-thread"].conversation.model_override
    assert override is not None
    assert override["base_url"] == GATE
    assert override["provider"] == "zai"


def test_rehydrate_keeps_deliberate_proxy_pin(monkeypatch):
    """A persisted pin on a non-vendor host (deliberate proxy) is not rewritten."""
    states = {}
    runner = _make_runner(states)
    runner.session_store.get_model_override = lambda key: {
        "model": "glm-5.3-flash", "provider": "zai", "base_url": "https://my-proxy.example/v1"}
    _patch_runtime_resolution(monkeypatch)

    runner._rehydrate_session_model_override("proxy-thread")

    override = states["proxy-thread"].conversation.model_override
    assert override["base_url"] == "https://my-proxy.example/v1"
