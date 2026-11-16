"""The closed page shows each screen as its own tab."""

import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from rfproteina.disease.ui import (
    BrowserSession,
    OptionForm,
    OptionPanel,
    RestartFreshError,
    ReturnToStartError,
    _close_script,
    _done_page,
    _launch_script,
    render_options,
)

_AXES = (
    "Immunogenicity",
    "Proteome off-target",
    "MHC class I",
    "MHC class II",
    "ADA",
    "Paralog",
    "Plasma abundance",
    "Plasma hub",
    "Plasma fold",
)


def _screen() -> dict[str, object]:
    def untested(blocker: str, **extra: object) -> dict[str, object]:
        record: dict[str, object] = {"state": "untested", "measurement": None, "blocker": blocker, "source": "untested"}
        record.update(extra)
        return record

    return {
        "call": "untested",
        "untested": [
            "immunogenicity",
            "proteome_offtarget",
            "mhc_i",
            "mhc_ii",
            "ada",
            "paralog",
            "plasma_abundance",
            "plasma_hub",
            "plasma_fold",
        ],
        "pre_oracle_pass": True,
        "ppi_pass": "absent",
        "ppi_blocker": "oracle_not_configured",
        "sequence_developability": {
            "gravy": -0.4833,
            "instability_index": 54.353,
            "sequence_pass": False,
        },
        "designed_interface": {"contact_ms": 1396.7285, "clash_fraction": 0.0},
        "sequence_liability": {
            "immunogenicity": untested(
                "no_pass_cut",
                measurement=0.51,
                positive_windows=32,
                method="calis_2013_class_i_9mer",
                presentation="not_configured",
            ),
            "proteome_offtarget": untested("proteome_not_configured"),
            "mhc_i": untested("predictor_not_configured"),
            "mhc_ii": untested("predictor_not_configured"),
            "ada": {
                "state": "hit",
                "measurement": 12.0,
                "blocker": None,
                "source": "measured",
                "method": "mhc_ii_dr_breadth_paul_2015",
            },
        },
        "target_liability": {
            "paralog": untested("network_not_permitted"),
            "plasma_abundance": untested("threshold_not_set", measurement=1500.0, units="pg/L"),
            "plasma_hub": untested("network_not_permitted"),
            "plasma_fold": untested("fold_not_configured"),
        },
    }


def test_each_screen_has_a_tab_with_high_low_guidance() -> None:
    session = BrowserSession(opener=lambda _url: None)
    lines = [
        "Screen 4PCW chain 1: untested. Untested: immunogenicity, proteome_offtarget.",
        "Measured 4PCW chain 1: gravy -0.4833, instability 54.353, sequence_pass false.",
        "Follow-up command stays visible.",
    ]
    session.note_closing(lines, [{"entry_id": "4PCW", "chain": "1", "screen": _screen()}])
    page = session.done_page()

    assert "Panel closed" in page
    # The user asked cure wording out of this flow on 2026-10-31, and the board
    # is part of it. This pins the absence rather than the sentence that was here.
    assert "cure" not in page.casefold()
    assert page.index("4PCW chain 1") < page.index("Technical record")
    assert page.index("Immunogenicity") < page.index("Technical record")
    for label in _AXES:
        assert f">{label}</label>" in page
    assert page.count('class="axis-panel"') == len(_AXES)
    assert page.count("<svg") >= 3
    assert ">Low</text>" in page
    assert ">High</text>" in page
    assert "0.51" in page
    assert "32 nine-residue windows scored above 0." in page
    # ExPASy ProtParam states the stability rule as "smaller than 40", so the
    # page says "not below" rather than "above": exactly 40 fails too, and one
    # wording has to hold for every value. The reason still names the number
    # and the cut.
    assert "instability 54.353 is not below 40" in page
    assert "-0.4833" in page
    assert "1396.7285" in page
    assert "The refolding oracle is not configured." in page
    assert "was not recorded" not in page
    assert "No number yet" not in page
    assert "was not measured" not in page
    assert "At or below 20 is the concerning side" in page
    assert "A count of zero is not a clear." in page
    assert "Follow-up command stays visible." in page
    assert "Screen 4PCW chain 1: untested." in page
    assert page.index("<details>") < page.index("Screen 4PCW chain 1: untested.")
    assert ">Return to start page</button>" in page
    assert ">Restart Server</button>" in page
    assert page.rindex(">Close</button>") > page.rindex(">Restart Server</button>")


def test_an_empty_close_still_says_return_to_the_terminal() -> None:
    session = BrowserSession(opener=lambda _url: None)
    page = session.done_page()
    assert "Panel closed" in page
    assert "Return to the terminal." in page
    assert 'class="axis-panel"' not in page


def test_every_page_can_return_or_restart_and_close_is_last() -> None:
    landing = render_options(OptionForm())
    done = _done_page()
    for page in (landing, done):
        assert 'name="__nav" value="start" formnovalidate' in page
        assert 'name="__nav" value="fresh" formnovalidate' in page
        assert ">Return to start page</button>" in page
        assert ">Restart Server</button>" in page
    assert ">Close</button>" in done
    assert done.rindex(">Close</button>") > done.rindex(">Restart Server</button>")
    # Enter in a field activates the first submit. That is Return, so the script
    # keeps the page action unless that button itself has focus.
    assert 'pressed.getAttribute("name") === "__nav"' in done
    assert "document.activeElement !== pressed" in done
    session = BrowserSession(opener=lambda _url: None)
    session._shutdown = True
    assert "window.close()" in session.done_page()


def test_return_to_start_keeps_the_typed_disease_and_restart_drops_it() -> None:
    panel = OptionPanel(
        lambda _page: {"disease": ["Asthma"], "binder_kind": ["protein"], "where": ["local"], "__nav": ["start"]}
    )
    panel.seal()
    with pytest.raises(ReturnToStartError):
        panel.ask(render_options(OptionForm()))
    panel.prepare_saved_start()
    assert panel._landing_override is not None
    assert panel._landing_override.disease == "Asthma"
    assert 'name="disease" value="Asthma"' in render_options(panel._landing_override)
    assert panel._sealed is False
    assert panel._ahead[0]["disease"] == ["Asthma"]

    fresh = OptionPanel(lambda _page: {"disease": ["Asthma"], "binder_kind": ["protein"], "__nav": ["fresh"]})
    fresh._answers.append({"disease": ["Kept"], "binder_kind": ["protein"]})
    with pytest.raises(RestartFreshError):
        fresh.ask("<p>later</p>")
    fresh.prepare_fresh_start()
    assert fresh._answers == []
    assert fresh._ahead == []
    assert fresh._landing_override is None
    assert fresh.finish_choice() == "close"


def test_close_targets_only_the_panel_window() -> None:
    launch = _launch_script("http://127.0.0.1:41947/")
    close = _close_script("41947")
    assert "--app=" in launch
    assert "http://127.0.0.1:41947/" in launch
    assert "rfd4-panel-" in launch
    assert "Start-Process $url" in launch
    assert "LOCALAPPDATA" in launch
    assert "rfd4-panel-" in close
    assert "41947" in close
    assert "Stop-Process" in close
    assert "msedge.exe" in close
    assert "chrome.exe" in close
    assert "taskkill" not in close


def test_close_answers_with_the_page_that_closes_the_window() -> None:
    """Close keeps the server up until the browser has the shutdown page."""
    session = BrowserSession(opener=lambda _url: None)
    found: dict[str, dict[str, list[str]]] = {}
    body: dict[str, bytes] = {}
    errors: list[BaseException] = []

    def serve() -> None:
        found["fields"] = session.exchange("<p>Generation finished.</p>", step=4)

    def post() -> None:
        try:
            assert session._server is not None
            url = f"http://127.0.0.1:{session._server.server_address[1]}/"
            data = urllib.parse.urlencode({}).encode()
            response = urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5)
            body["page"] = response.read()
        except (OSError, urllib.error.URLError) as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve)
    poster = threading.Thread(target=post)
    worker.start()
    try:
        for _ in range(100):
            if session._server is not None:
                break
            time.sleep(0.02)
        assert session._server is not None
        poster.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert found["fields"] == {}
        assert session._window_port is None
        session.close()
        poster.join(timeout=5)
    finally:
        session.close()
    assert errors == []
    assert not poster.is_alive()
    page = body["page"].decode("utf-8")
    assert "Closing the panel." in page
    assert "window.close()" in page


def test_lines_without_a_screen_record_stay_on_the_page() -> None:
    session = BrowserSession(opener=lambda _url: None)
    session.note_closing(["Screen 8TLD chain 2: untested. Untested: immunogenicity."])
    page = session.done_page()
    assert "Panel closed" in page
    assert "Screen 8TLD chain 2: untested. Untested: immunogenicity." in page
    assert "<details>" not in page
    assert 'class="axis-panel"' not in page
