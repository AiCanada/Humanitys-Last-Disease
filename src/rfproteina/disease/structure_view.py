"""Three atomic views of a finished design, for the page shown when generation ends.

The disease view is the chain whose sequence the condition spec fixed, taken from
the structure that was designed against. The second view is that chain with the
generated binder. The third is the binder on its own. Every atom in the chosen
chains is drawn. Nothing here is a measurement, and a picture of a design is
not a cure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import yaml

from rfproteina.disease.design_spec import _atom_site_rows, cif_text
from rfproteina.disease.logs import log
from rfproteina.disease.screen import base_name, designed_chain, sample_cifs

_SELECT_CHAIN = re.compile(r"""sel\(\s*['"]([A-Za-z0-9]+)(?:/|['"])""")
_MISSING = frozenset({".", "?"})
_SPEC_WALK = 8
_VIEWER_PATH = Path(__file__).resolve().parents[3] / "website" / "vendor" / "3Dmol-min.js"
_VIEWER_BYTES: bytes | None = None


@dataclass(frozen=True)
class StructureTrio:
    """One generated sample and the two files the three views are drawn from."""

    sample: str
    title: str
    disease_file: Path
    disease_chains: tuple[str, ...]
    complex_file: Path
    target_chains: tuple[str, ...]
    binder_chains: tuple[str, ...]


def structure_trios(output_dir: Path, *, entry_id: str, chain: str) -> list[StructureTrio]:
    """The disease structure, the complex, and the binder for each generated sample.

    A sample whose file cannot be read is left out. The others are still shown.
    """
    samples = sample_cifs(output_dir)
    if not samples:
        return []
    spec = _spec_path(output_dir)
    fixed, disease_source = _fixed_labels(spec) if spec is not None else ([], None)
    try:
        designed = designed_chain(output_dir)
    except (OSError, ValueError):
        designed = ""
    title = _title(entry_id, chain)
    disease_labels = _label_chains(disease_source) if disease_source is not None else []
    trios: list[StructureTrio] = []
    for sample, path in samples.items():
        labels = _label_chains(path)
        if not labels:
            log().warning("a generated structure could not be drawn")
            continue
        targets, binders = _split(labels, fixed, designed)
        disease_present = {pair[0] for pair in disease_labels}
        disease_chains = [label for label in fixed if label in disease_present]
        if disease_source is not None and disease_chains:
            disease_file: Path = disease_source
            shown_disease = tuple(disease_chains)
        else:
            disease_file = path
            shown_disease = tuple(targets)
        trios.append(
            StructureTrio(
                sample=sample,
                title=title,
                disease_file=disease_file,
                disease_chains=shown_disease,
                complex_file=path,
                target_chains=tuple(targets),
                binder_chains=tuple(binders),
            )
        )
    return trios


def catalog_entry(trio: StructureTrio, disease_token: str, complex_token: str) -> dict[str, object]:
    """The fields the page's viewer reads. Paths stay out of this record."""
    return {
        "sample": trio.sample,
        "title": trio.title,
        "disease_token": disease_token,
        "disease_chains": list(trio.disease_chains),
        "complex_token": complex_token,
        "target_chains": list(trio.target_chains),
        "binder_chains": list(trio.binder_chains),
        # Filled in later, by a results-page retrieval. Present and empty from
        # the start so the viewer has one shape to read rather than two, and so
        # an overlay with nothing in it says "no model retrieved yet" instead
        # of failing on a missing field.
        "af2_token": "",
        "af2_detail": "",
        "af3_token": "",
        "af3_detail": "",
        "af2c_token": "",
        "af2c_detail": "",
        "af3c_token": "",
        "af3c_detail": "",
    }


def render_structure_views(catalog: Sequence[dict[str, object]]) -> str:
    """The three viewers, or nothing when this run has no generated structure."""
    if not catalog:
        return ""
    options: list[str] = []
    for index, item in enumerate(catalog):
        sample = str(item.get("sample", ""))
        title = str(item.get("title", ""))
        label = f"{sample} — {title}" if title else sample
        options.append(f'<option value="{index}">{escape(label)}</option>')
    picker = ""
    if len(catalog) > 1:
        picker = '<label class="atom-sample">Design ' f'<select id="atom-sample">{"".join(options)}</select></label>'
    payload = json.dumps(list(catalog), ensure_ascii=True).replace("<", "\\u003c")
    return f"""<section class="atom-views">
<style>{_ATOM_STYLE}</style>
<h2>Structures</h2>
<p class="hint">Three views of the generated design. Every heavy atom is drawn.
Gray carbon is the disease structure. Dark green carbon is the binder. Orange is any other atom.
Drag to rotate. Scroll to zoom. Point at an atom to name it.</p>
<p class="atom-legend"><span><i class="swatch disease"></i>Disease</span>
<span><i class="swatch binder"></i>Binder</span>
<span><i class="swatch other"></i>Other</span></p>
<p class="atom-title"></p>
{picker}
<div class="atom-grid">
{_figure("atom-disease", "Disease structure")}
{_figure("atom-complex", "Disease structure with binder")}
{_figure("atom-binder", "Binder")}
</div>
<h2>Retrieved AlphaFold models</h2>
<p class="hint">Each overlay draws a retrieved model over this run's own structure, in the
same frame. Blue carbon is the retrieved model. The line under each one says what was
compared, over how many residues, and where the model came from. An overlay stays empty
until a model has been retrieved for it.</p>
<p class="atom-legend"><span><i class="swatch retrieved"></i>Retrieved model</span>
<span><i class="swatch binder"></i>This run</span></p>
<div class="atom-grid">
{_figure("atom-af2", "AlphaFold 2 model with binder")}
{_figure("atom-af3", "AlphaFold 3 model with binder")}
</div>
<h2>Retrieved models of the combined structure</h2>
<p class="hint">The same two services asked the other question: whether the whole complex is
reproduced, not just the binder. Each detail line gives the fit over both chains, each chain
on its own, and -- the one that matters -- how far the binder sits from where it belongs once
the target is superposed.</p>
<div class="atom-grid">
{_figure("atom-af2c", "AlphaFold 2 model with the combined structure")}
{_figure("atom-af3c", "AlphaFold 3 model with the combined structure")}
</div>
<p class="atom-missing" hidden>The 3D viewer did not load.</p>
<script type="application/json" id="atom-catalog">{payload}</script>
<script src="/viewer/3dmol.js"></script>
<script>{_VIEWER_SCRIPT}</script>
</section>
"""


def viewer_javascript() -> bytes | None:
    """The vendored 3Dmol build, or None when this checkout does not have it."""
    global _VIEWER_BYTES
    if _VIEWER_BYTES is not None:
        return _VIEWER_BYTES
    if not _VIEWER_PATH.is_file():
        return None
    try:
        _VIEWER_BYTES = _VIEWER_PATH.read_bytes()
    except OSError:
        return None
    return _VIEWER_BYTES


def _figure(stage_id: str, caption: str) -> str:
    return (
        "<figure>"
        f"<figcaption>{escape(caption)}</figcaption>"
        '<p class="atom-detail"></p>'
        f'<div id="{stage_id}" class="atom-stage" role="img" aria-label="{escape(caption)}"></div>'
        '<p class="atom-tools"><button type="button" class="atom-reset">Reset view</button></p>'
        '<p class="atom-status" role="status"></p>'
        '<p class="atom-hover"></p>'
        "</figure>"
    )


def _title(entry_id: str, chain: str) -> str:
    entry = entry_id.strip()
    named = chain.strip()
    if entry and named:
        return f"{entry} chain {named}"
    return entry or named


def _spec_path(output_dir: Path) -> Path | None:
    names = [base_name(output_dir), output_dir.name]
    seen: set[Path] = set()
    for name in names:
        for candidate in (output_dir / f"{name}.yaml", output_dir.parent / f"{name}.yaml"):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


def _fixed_labels(spec: Path) -> tuple[list[str], Path | None]:
    """Chains whose sequence is fixed, and the structure that was designed against."""
    try:
        payload = yaml.safe_load(spec.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return [], None
    if not isinstance(payload, dict):
        return [], None
    binder = payload.get("binder")
    if not isinstance(binder, dict):
        return [], None
    raw_input = binder.get("input")
    source = _find_input(spec, raw_input) if isinstance(raw_input, str) else None
    conditions = binder.get("conditions")
    sequenced = conditions.get("C_SEQ") if isinstance(conditions, dict) else None
    rows: object = None
    if isinstance(sequenced, dict):
        for key in (True, "true", "True"):
            if key in sequenced:
                rows = sequenced[key]
                break
    chains: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            select = row.get("select")
            if not isinstance(select, str):
                continue
            match = _SELECT_CHAIN.search(select)
            if match is None:
                continue
            chain = match.group(1)
            if chain not in chains:
                chains.append(chain)
    return chains, source


def _find_input(spec: Path, raw: str) -> Path | None:
    text = raw.strip()
    if not text:
        return None
    direct = Path(text)
    if direct.is_file():
        return direct
    current = spec.parent
    for _step in range(_SPEC_WALK):
        found = current / text
        if found.is_file():
            return found
        if current.parent == current:
            return None
        current = current.parent
    return None


def _label_chains(path: Path | None) -> list[tuple[str, bool]]:
    """Label chain ids in the order they appear, and whether each has a CA.

    Goes through the decompressing read because ``sample_cifs`` returns
    compressed designs now, and because the disease structure beside a run may
    itself be a ``.cif.zst``. Read as plain text either one yielded no chains,
    and the sample was dropped with "a generated structure could not be drawn".
    """
    if path is None:
        return []
    text = cif_text(path)
    if text is None:
        return []
    seen: dict[str, bool] = {}
    order: list[str] = []
    for row in _atom_site_rows(text):
        label = row.get("label_asym_id", "").strip()
        if not label or label in _MISSING:
            continue
        if label not in seen:
            seen[label] = False
            order.append(label)
        if row.get("label_atom_id") == "CA":
            seen[label] = True
    return [(label, seen[label]) for label in order]


def _split(labels: list[tuple[str, bool]], fixed: list[str], designed: str) -> tuple[list[str], list[str]]:
    """Disease chains and binder chains. A chain without CA is left in the complex only."""
    polymers = [label for label, has_ca in labels if has_ca]
    present = set(polymers)
    targets = [label for label in fixed if label in present]
    matched = _match_label(polymers, designed)
    if targets:
        if matched and matched not in set(targets):
            return targets, [matched]
        binders = [label for label in polymers if label not in set(targets)]
        if matched and matched in binders and len(binders) > 1:
            binders = [matched]
        return targets, binders
    if matched:
        return [label for label in polymers if label != matched], [matched]
    return [], []


def _match_label(labels: Sequence[str], designed: str) -> str | None:
    """The label a metrics chain names. ``A_1`` is label ``A`` when that chain exists."""
    text = designed.strip()
    if text in labels:
        return text
    letter = text.split("_", 1)[0]
    if letter and letter != text and letter in labels:
        return letter
    return None


_ATOM_STYLE = """
.atom-views { margin: 0 0 18px; }
.atom-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 10px;
  font-size: 12px; color: #3d4a57; }
.atom-legend i { display: inline-block; width: 0.75rem; height: 0.75rem;
  margin-right: 6px; border-radius: 2px; vertical-align: -1px; }
.swatch.disease { background: #a9a9a9; }
.swatch.binder { background: #006400; }
.swatch.other { background: #ffa500; }
.swatch.retrieved { background: #1f4fd8; }
.atom-title { margin: 0 0 8px; font-weight: 700; min-height: 1.2em; }
.atom-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.atom-grid figure { margin: 0; min-width: 0; }
.atom-grid figcaption { font-weight: 700; margin: 0 0 2px; }
.atom-detail { min-height: 1.2em; margin: 0 0 6px; color: #52616e; font-size: 12px; }
.atom-stage { position: relative; overflow: hidden; height: 420px; touch-action: none;
  background: #f7f9fb; border: 1px solid #d5dee6; border-radius: 8px; }
.atom-tools { margin: 6px 0 0; }
.atom-views button.atom-reset { margin: 0; padding: 3px 8px; border-radius: 6px;
  background: #e3e9ef; color: #26323d; font-size: 12px; font-weight: 600; }
.atom-views button.atom-reset:focus-visible { outline: 2px solid #0b6e4f; outline-offset: 2px; }
.atom-status, .atom-hover { min-height: 1.2em; margin: 4px 0 0; color: #3d4a57; font-size: 12px; }
.atom-sample { display: flex; gap: 8px; align-items: center; margin: 0 0 12px; font-weight: 700; }
.atom-views select { width: auto; min-width: 14rem; max-width: 100%; }
@media (max-width: 900px) {
  .atom-grid { grid-template-columns: 1fr; }
  .atom-stage { height: 480px; }
}
"""

_VIEWER_SCRIPT = """
(function () {
  var node = document.getElementById("atom-catalog");
  var missing = document.querySelector(".atom-missing");
  if (!node || !window.$3Dmol) {
    if (missing) missing.hidden = false;
    return;
  }
  var catalog = JSON.parse(node.textContent || "[]");
  var picker = document.getElementById("atom-sample");
  var title = document.querySelector(".atom-title");
  var stages = {
    af2: document.getElementById("atom-af2"),
    af3: document.getElementById("atom-af3"),
    af2c: document.getElementById("atom-af2c"),
    af3c: document.getElementById("atom-af3c"),
    disease: document.getElementById("atom-disease"),
    complex: document.getElementById("atom-complex"),
    binder: document.getElementById("atom-binder")
  };
  var viewers = {};
  var cache = {};
  var ticket = 0;
  function field(name, cls) {
    return stages[name].parentNode.querySelector(cls);
  }
  function viewerFor(name) {
    if (viewers[name]) return viewers[name];
    if (!stages[name]) return null;
    try {
      viewers[name] = $3Dmol.createViewer(stages[name], {
        backgroundColor: "white",
        antialias: true
      });
    } catch (error) {
      if (missing) missing.hidden = false;
      return null;
    }
    return viewers[name];
  }
  // A missing alt id is ".". "*" keeps that atom in parsers that honor altLoc.
  // Text CIF leaves altLoc unset, and paint then falls back to the chain.
  function selection(chains) {
    var sel = {altLoc: ["", ".", "?", "A"]};
    if (chains && chains.length) sel.lchain = chains.length === 1 ? chains[0] : chains;
    return sel;
  }
  function appearance(viewer, sel, scheme) {
    var atoms = viewer.selectedAtoms(sel);
    var bonded = false;
    for (var i = 0; i < atoms.length; i++) {
      if (atoms[i].bonds && atoms[i].bonds.length) { bonded = true; break; }
    }
    if (!bonded) return {sphere: {scale: 0.75, colorscheme: scheme}};
    return {
      stick: {radius: 0.15, colorscheme: scheme},
      sphere: {scale: 0.25, colorscheme: scheme}
    };
  }
  function paintAtoms(viewer, chains, scheme) {
    var sel = selection(chains);
    if (!viewer.selectedAtoms(sel).length) {
      var broad = {};
      if (chains && chains.length) broad.lchain = sel.lchain;
      if (viewer.selectedAtoms(broad).length) sel = broad;
    }
    if (!viewer.selectedAtoms(sel).length) return null;
    viewer.setStyle(sel, appearance(viewer, sel, scheme));
    return {sel: sel, count: viewer.selectedAtoms(sel).length};
  }
  function paintComplex(viewer, item) {
    var base = paintAtoms(viewer, null, "orangeCarbon");
    if (!base) return null;
    if (item.target_chains.length) paintAtoms(viewer, item.target_chains, "grayCarbon");
    if (item.binder_chains.length) paintAtoms(viewer, item.binder_chains, "darkgreenCarbon");
    return base;
  }
  function clean(value, max) {
    return String(value == null ? "" : value).replace(/[^A-Za-z0-9_.-]/g, "").slice(0, max);
  }
  function nameAtom(atom) {
    var bits = [];
    var chain = clean(atom.lchain || atom.chain, 4);
    if (chain) bits.push("chain " + chain);
    var residue = (clean(atom.resn, 8) + " " + clean(atom.resi, 6)).trim();
    if (residue) bits.push(residue);
    var atomName = clean(atom.atom, 6);
    var elem = clean(atom.elem, 3);
    if (atomName) bits.push(atomName);
    if (elem && elem !== atomName) bits.push(elem);
    return bits.join(" · ");
  }
  function chainLine(chains) {
    if (!chains || !chains.length) return "";
    return (chains.length > 1 ? "Chains " : "Chain ") + chains.join(", ");
  }
  function summary(count, chains) {
    var who = chainLine(chains);
    var noun = count === 1 ? " atom. " : " atoms. ";
    return (who ? who + ". " : "") + count + noun + "Point at an atom to name it.";
  }
  function forget(viewer) {
    if (!viewer) return;
    viewer._atomView = null;
    viewer._atomLook = null;
  }
  function remember(viewer) {
    viewer._atomView = viewer.getView();
    viewer._atomLook = viewer.lookingAt && viewer.lookingAt.clone ? viewer.lookingAt.clone() : null;
  }
  function restore(viewer) {
    if (!viewer || !viewer._atomView) return;
    if (viewer._atomLook && viewer.lookingAt && viewer.lookingAt.copy) {
      viewer.lookingAt.copy(viewer._atomLook);
      if (viewer.camera && viewer.camera.lookAt) viewer.camera.lookAt(viewer.lookingAt);
    }
    viewer.setView(viewer._atomView);
  }
  function blank(name, message) {
    var status = field(name, ".atom-status");
    var hover = field(name, ".atom-hover");
    if (viewers[name]) {
      forget(viewers[name]);
      viewers[name].clear();
      viewers[name].render();
    }
    if (hover) hover.textContent = "";
    status.hidden = false;
    status.textContent = message;
  }
  function load(token) {
    if (!cache[token]) {
      cache[token] = fetch("/structure/" + token, {cache: "no-store"}).then(function (response) {
        if (!response.ok) throw new Error("missing");
        return response.text();
      });
      cache[token].catch(function () { delete cache[token]; });
    }
    return cache[token];
  }
  function frame(viewer, painted) {
    viewer.zoomTo(painted.sel);
    viewer.zoom(0.85, 0);
    remember(viewer);
  }
  function watch(viewer, painted, status, hover) {
    status.hidden = false;
    status.textContent = summary(painted.count, null);
    viewer.setHoverable(painted.sel, true, function (atom) {
      hover.textContent = nameAtom(atom);
    }, function () {
      hover.textContent = "";
    });
  }
  function show(name, token, paint, mine) {
    var status = field(name, ".atom-status");
    var hover = field(name, ".atom-hover");
    var viewer = viewerFor(name);
    if (!viewer) {
      status.textContent = "The 3D viewer did not load.";
      return;
    }
    if (hover) hover.textContent = "";
    status.hidden = false;
    status.textContent = "Loading atoms…";
    load(token).then(function (text) {
      if (mine !== ticket) return;
      viewer.clear();
      viewer.addModel(text, "cif", {altLoc: "*"});
      viewer.setStyle({}, {});
      var painted = paint(viewer);
      if (!painted) {
        forget(viewer);
        viewer.clear();
        viewer.render();
        status.hidden = false;
        status.textContent = "This view has no atoms to draw.";
        return;
      }
      frame(viewer, painted);
      watch(viewer, painted, status, hover);
      viewer.resize();
      viewer.render();
    }).catch(function () {
      if (mine !== ticket) return;
      forget(viewer);
      viewer.clear();
      viewer.render();
      status.hidden = false;
      status.textContent = "This structure could not be loaded.";
    });
  }
  function overlay(name, item, which) {
    var token = item[which + "_token"];
    var detail = field(name, ".atom-detail");
    var status = field(name, ".atom-status");
    detail.textContent = item[which + "_detail"] || "";
    if (!token) {
      blank(name, item[which + "_detail"] || "No model retrieved yet.");
      return;
    }
    var mine = ++ticket;
    var viewer = viewerFor(name);
    if (!viewer) {
      status.textContent = "The 3D viewer did not load.";
      return;
    }
    status.hidden = false;
    status.textContent = "Loading atoms…";
    // Both structures before either is painted: a model added after styling
    // is drawn with the default style and reads as a third thing on screen.
    Promise.all([load(item.complex_token), load(token)]).then(function (both) {
      if (mine !== ticket) return;
      viewer.clear();
      viewer.addModel(both[0], "cif", {altLoc: "*"});
      var retrieved = viewer.addModel(both[1], "cif", {altLoc: "*"});
      viewer.setStyle({}, {});
      var own = null;
      if (which === "af2c" || which === "af3c") {
        // The complex panes draw both of this run's chains, because the
        // question is whether the whole thing was reproduced.
        if (item.target_chains.length) paintAtoms(viewer, item.target_chains, "grayCarbon");
        own = item.binder_chains.length
          ? paintAtoms(viewer, item.binder_chains, "darkgreenCarbon")
          : null;
      } else if (item.binder_chains.length) {
        own = paintAtoms(viewer, item.binder_chains, "darkgreenCarbon");
      }
      var sel = {model: retrieved};
      viewer.setStyle(sel, appearance(viewer, sel, "blueCarbon"));
      var painted = own || {sel: sel, count: 1};
      frame(viewer, {sel: {}, count: 1});
      watch(viewer, painted, status, field(name, ".atom-hover"));
      viewer.resize();
      viewer.render();
    }).catch(function () {
      if (mine !== ticket) return;
      forget(viewer);
      viewer.clear();
      viewer.render();
      status.hidden = false;
      status.textContent = "That model could not be drawn.";
    });
  }
  function draw(index) {
    var item = catalog[index];
    var mine = ++ticket;
    if (!item) return;
    if (title) {
      var named = item.sample && item.title ? item.sample + " — " + item.title : "";
      title.textContent = named || item.sample || item.title || "";
    }
    field("disease", ".atom-detail").textContent = chainLine(item.disease_chains);
    var complexLine = [];
    if (item.target_chains.length) complexLine.push("Disease " + item.target_chains.join(", "));
    if (item.binder_chains.length) complexLine.push("Binder " + item.binder_chains.join(", "));
    field("complex", ".atom-detail").textContent = complexLine.join(" · ");
    field("binder", ".atom-detail").textContent = chainLine(item.binder_chains);
    if (!item.disease_chains.length) blank("disease", "This view has no atoms to draw.");
    else show("disease", item.disease_token, function (viewer) {
      return paintAtoms(viewer, item.disease_chains, "grayCarbon");
    }, mine);
    show("complex", item.complex_token, function (viewer) {
      return paintComplex(viewer, item);
    }, mine);
    if (!item.binder_chains.length) blank("binder", "This view has no atoms to draw.");
    else show("binder", item.complex_token, function (viewer) {
      return paintAtoms(viewer, item.binder_chains, "darkgreenCarbon");
    }, mine);
    overlay("af2", item, "af2");
    overlay("af3", item, "af3");
    overlay("af2c", item, "af2c");
    overlay("af3c", item, "af3c");
  }
  function resizeAll() {
    Object.keys(viewers).forEach(function (name) {
      viewers[name].resize();
      viewers[name].render();
    });
  }
  var names = ["disease", "complex", "binder"];
  var buttons = document.querySelectorAll(".atom-views .atom-reset");
  for (var b = 0; b < buttons.length && b < names.length; b++) {
    (function (name) {
      buttons[b].addEventListener("click", function () {
        restore(viewers[name]);
      });
    })(names[b]);
  }
  if (picker) picker.addEventListener("change", function () { draw(Number(picker.value)); });
  draw(0);
  window.addEventListener("resize", resizeAll);
  if (window.ResizeObserver) {
    var observer = new ResizeObserver(resizeAll);
    names.forEach(function (name) { if (stages[name]) observer.observe(stages[name]); });
  }
})();
"""
