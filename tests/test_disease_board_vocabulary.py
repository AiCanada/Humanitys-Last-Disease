"""Every blocker the disease code can name has a sentence where it is shown.

There are four places it can be shown, not one: the screening board, and the
results page's retrieval, comparison and export surfaces, each with its own
vocabulary. :data:`_VOCABULARIES` maps a module to the table that renders its
names, and records what checking all four against the board alone cost.

A blocker name is the whole explanation an untested axis gets. ``_blocker_lines``
falls back to the name with its underscores removed, so a name the board does not
know still renders -- as "fold target not in table", with no sentence saying what
that means and no action line saying what to do about it. Nothing fails, no test
goes red, and the reader is left with a slug.

That is how six of them arrived: ``fold_target_not_in_table``,
``reference_ic50_not_nm``, ``ic50_not_nm``, ``width_without_cutoff``,
``calis_mask_unpublished_for_allele`` and ``reference_too_short_for_width`` were
each emitted by a module before the board had a word for them -- the last of
them found by this test, minutes after it was written, against a module still
being changed. This test reads the blocker names out of the source rather than
asking anyone to remember, so the next one cannot arrive quietly.

The scan is static, and what it covers is worth stating exactly rather than
promising everything. It reads a literal passed as the third argument of
``_axis`` (called plainly or through a module attribute), a ``blocker=``
keyword, an assignment to a ``blocker`` or ``*_blocker`` subscript including a
tuple-unpacking one, a ``blocker``/``*_blocker`` key in a dict literal, the
first element of a ``return`` from a function annotated to hand a blocker back
in a tuple, and a bare ``return`` literal from a function whose name ends in
``_blocker`` or ``_reason``. In both of the last two the annotation must be
OPTIONAL -- see ``_BLOCKER_TEXT``, which records what accepting a plain ``str``
cost when it was tried.

A literal handed to a helper that declares a ``blocker`` parameter is seen too,
at whatever position that helper puts it. That used to be the single name
``_axis``, matched literally, so a sibling helper of the same shape under
another name emitted blockers nothing checked; the helpers are discovered from
the source now, by the same optional-annotation signal everything else here
uses.

It still does NOT see: a name reaching the record through a variable (those are
covered by their own modules' tests); a literal appended into a structure that
is later read as a blocker, such as ``_AXES.append(("name", "untested"))``; or a
bare ``return "name"`` from a function whose name follows neither convention.
Those shapes carry no signal to search for, and widening the scan until it
caught them would make it demand board sentences for ordinary strings -- which
was tried and measured, and is recorded in ``_BLOCKER_TEXT``.

What closes that gap is the check from the other end:
``test_no_new_blocker_hides_in_a_shape_the_scan_cannot_see`` pins the board
sentences this scan cannot account for, so the next one to appear fails rather
than arriving quietly. The floor below guards against the scan matching nothing
at all.
"""

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from rfproteina.disease import (
    ddd_benchmark,
    ddd_endpoint,
    deepsp,
    deepviscosity,
    dynamics,
    human_protocol,
    results_actions,
    results_compare,
    results_retrieve,
    screen_board,
)

_PACKAGE = Path(screen_board.__file__).parent

#: The board's own vocabulary, which is what a module is checked against unless
#: it renders its blockers somewhere else.
_BOARD = (("screen_board._BLOCKERS", screen_board._BLOCKERS),)

#: Which vocabulary turns a module's blocker names into words for the reader.
#:
#: There is more than one surface. The screening board explains an untested axis
#: on the screen; the results page explains a retrieval, a comparison and an
#: export, and holds its own words for each because it speaks in its own voice --
#: a comparison says "The retrieved model has no chains to compare", an export
#: says "(design_007: its structure file could not be read)", and neither reads
#: right in the other's place.
#:
#: This mapping exists because the check used to ignore that. It asked every
#: name in the package for a sentence in ``screen_board._BLOCKERS``, and when the
#: results modules arrived with 21 names of their own it reported 21 missing
#: sentences -- while 17 of them already had one, in the table that actually
#: renders them. The failure was real and the diagnosis was not: the suggested
#: fix, adding all 21 to the board, would have duplicated 17 sentences into a
#: table that never renders them and left the two copies free to drift apart.
#:
#: Checking each name against the surface that shows it keeps the guarantee on
#: all four surfaces instead of one, and keeps it honest: a results blocker with
#: no words anywhere can no longer be quieted by adding it to the board.
_VOCABULARIES: dict[str, tuple[tuple[str, Mapping[str, str]], ...]] = {
    "results_retrieve.py": (("results_retrieve._BLOCKER_TEXT", results_retrieve._BLOCKER_TEXT),),
    "results_compare.py": (
        ("results_compare._BLOCKER_TEXT", results_compare._BLOCKER_TEXT),
        ("results_compare._COMPLEX_BLOCKERS", results_compare._COMPLEX_BLOCKERS),
    ),
    "results_actions.py": (("results_actions._BLOCKER_TEXT", results_actions._BLOCKER_TEXT),),
    # The benchmark adapter's blockers answer an external scorer, not a reader of
    # the board, which never renders them. Routed to its own table for the same
    # reason the three results modules are: adding them to the board would put a
    # second copy of each sentence somewhere nobody reads.
    "ddd_benchmark.py": (("ddd_benchmark._BLOCKER_TEXT", ddd_benchmark._BLOCKER_TEXT),),
    "ddd_endpoint.py": (("ddd_endpoint._BLOCKER_TEXT", ddd_endpoint._BLOCKER_TEXT),),
    # The protocol is read on the results page and in a terminal. Its names are
    # about a comparison a reader set up themselves -- which protein was not in
    # the proteome, which model was not supplied -- and the advice rides in the
    # sentence, as it does on the three results surfaces, rather than in a
    # separate action line the board would need.
    "human_protocol.py": (("human_protocol._BLOCKER_TEXT", human_protocol._BLOCKER_TEXT),),
    # These two put their names on a ``deepsp_blocker`` / ``viscosity_blocker``
    # key that nothing reads: the screen reaches these descriptors through an
    # oracle table ``developability.py`` reads, rather than by running the
    # models. So the board does not render them and four sentences added there
    # would have been four lines nobody draws -- the mistake this file's
    # docstring records costing seventeen duplicated lines once already. The
    # words live beside the code that emits them instead, ready for whichever
    # surface shows them first.
    "deepsp.py": (("deepsp.BLOCKER_TEXT", deepsp.BLOCKER_TEXT),),
    "deepviscosity.py": (("deepviscosity.BLOCKER_TEXT", deepviscosity.BLOCKER_TEXT),),
    # The analyst is read on its own tab and in a terminal, like the protocol
    # and the three results surfaces. Its names are about an ensemble and an
    # install, which the screening board has no view of.
    "dynamics.py": (("dynamics._BLOCKER_TEXT", dynamics._BLOCKER_TEXT),),
    # The runner prints names for the screen to report, and the screen is what
    # renders them. One table, not two: it runs under another interpreter and
    # cannot import this package, but it does not need to -- the sentences are
    # read on the analyst's surface, so that is where they live.
    "confrover_runner.py": (("dynamics._BLOCKER_TEXT", dynamics._BLOCKER_TEXT),),
}


def _vocabulary_for(where: str) -> tuple[tuple[str, Mapping[str, str]], ...]:
    """The tables that put one module's blocker names into words."""
    return _VOCABULARIES.get(where, _BOARD)


#: Every sentence written on any surface, for the checks that work backwards
#: from the words to the names. Anything less would let a name added to a
#: results table in an unscannable shape escape the ratchet below.
_ALL_SENTENCES = frozenset(
    name for tables in (_BOARD, *_VOCABULARIES.values()) for _, table in tables for name in table
)

# The scan found this many on the day it was written. It is a floor, not a count:
# it fails if a refactor moves the emissions out of the shapes below, which would
# otherwise leave this test passing while checking nothing.
_FLOOR = 70


def _literals(node: ast.AST) -> list[str]:
    """The string literals a blocker expression can evaluate to.

    Covers the conditional and ``or`` forms the modules use to pick between two
    named blockers. Anything else evaluates at runtime and is not read here.
    """
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, ast.IfExp):
        return _literals(node.body) + _literals(node.orelse)
    if isinstance(node, ast.BoolOp):
        return [name for value in node.values for name in _literals(value)]
    return []


#: A blocker slot is OPTIONAL, because a measurement that was not blocked has no
#: blocker. That is the signal this scan follows, and widening it to a plain
#: ``str`` was measured and reverted: eleven functions in this package return
#: ``tuple[str, ...]`` or ``tuple[str, str]`` whose first element is a kind or a
#: status rather than a blocker -- ``coevolution._classified``,
#: ``toxicology._parse_cell``, ``ui._where`` -- and the scan then demanded board
#: sentences for "absent", "local", "nm" and twelve more. ``screen._ppi_blocker``
#: is the same shape under a blocker-ish name: ``-> str``, returning "recorded"
#: when the flag is present, which is the absence of a blocker rather than one.
_BLOCKER_TEXT = {"str|None", "None|str"}


def _returns_a_blocker_first(node: ast.AST) -> bool:
    """Whether a function hands a blocker back as a tuple's first element.

    ``_reference_density`` and ``_load_proteome`` do this rather than going
    through ``_axis``, which is how ``reference_ic50_not_nm`` escaped the first
    version of this scan.
    """
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    returns = node.returns
    if not isinstance(returns, ast.Subscript):
        return False
    name = returns.value
    if not (isinstance(name, ast.Name) and name.id == "tuple"):
        return False
    inner = returns.slice
    if not (isinstance(inner, ast.Tuple) and inner.elts):
        return False
    return ast.unparse(inner.elts[0]).replace(" ", "") in _BLOCKER_TEXT


def _names_a_blocker(node: ast.AST) -> bool:
    """Whether a function's own name says its return value is a blocker.

    The name alone is not enough: ``screen._ppi_blocker`` is named for a blocker
    and returns ``-> str``, whose "recorded" means the flag was present and
    nothing was blocked. So the annotation has to agree -- an optional return,
    the shape a real blocker slot has. A naming convention is still a weaker
    test than a type, which is why the module docstring discloses it rather than
    presenting it as full coverage.
    """
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    if not node.name.endswith(("_blocker", "_reason")):
        return False
    if node.returns is None:
        return False
    return ast.unparse(node.returns).replace(" ", "") in _BLOCKER_TEXT


def _called_name(node: ast.Call) -> str | None:
    """The bare name a call uses, whether plain or reached through a module."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parsed(path: Path) -> ast.Module:
    """One module of the package, as a syntax tree.

    Read as ``utf-8-sig`` and not as ``utf-8``, because that is what Python
    itself accepts: its source loader strips a leading byte-order mark, so a
    module saved with one imports perfectly well and nothing about it is
    broken. Handing such a file to :func:`ast.parse` as plain ``utf-8`` leaves
    the mark in the text as U+FEFF, which is a syntax error, and the whole of
    this file then fails to collect over an encoding chosen in some unrelated
    module. A scan that claims to read every module in the package has to read
    everything the interpreter will. On a file saved without a mark the codec
    is a no-op, so this costs nothing.
    """
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _blocker_helpers() -> dict[str, int]:
    """Helpers that take a blocker, and which positional argument it is.

    Found from the source rather than named here. This used to look for
    ``_axis`` by name, which covered the three identical ``_axis(state,
    measurement, blocker)`` helpers in ``liability``, ``oligo`` and
    ``target_liability`` and nothing else -- a fourth helper of the same shape
    under any other name would have emitted blockers the scan never saw, which
    is the gap this module's docstring used to disclose as ``_untested("name")``.

    The signal is the same one the rest of this scan follows: a parameter called
    ``blocker`` (or ``*_blocker``) whose annotation is OPTIONAL. A required
    ``str`` is not a blocker slot -- see :data:`_BLOCKER_TEXT` for what accepting
    one cost when it was tried.

    Keyword-only parameters are deliberately not given a position: a call can
    only reach them by name, and the ``blocker=`` keyword is already read below.
    """
    found: dict[str, int] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = _parsed(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            for index, argument in enumerate(positional):
                if argument.arg != "blocker" and not argument.arg.endswith("_blocker"):
                    continue
                if argument.annotation is None:
                    continue
                if ast.unparse(argument.annotation).replace(" ", "") in _BLOCKER_TEXT:
                    found[node.name] = index
    return found


_HELPERS = _blocker_helpers()


def _blocker_names() -> dict[str, set[str]]:
    """Every blocker name written as a literal, and the files writing it."""
    found: dict[str, set[str]] = {}

    def note(name: str, where: str) -> None:
        found.setdefault(name, set()).add(where)

    # Recursive, because a subpackage module is still this package.
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = _parsed(path)
        where = path.relative_to(_PACKAGE).as_posix()
        for node in ast.walk(tree):
            # A blocker handed to a helper that takes one, at whatever position
            # that helper declares it: _axis("untested", measurement, "name"),
            # and any sibling of the same shape, including one reached through
            # its module as liability._axis(...).
            if isinstance(node, ast.Call):
                index = _HELPERS.get(_called_name(node) or "")
                if index is not None and len(node.args) > index:
                    for name in _literals(node.args[index]):
                        note(name, where)
            # any call taking blocker="blocker_name"
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "blocker":
                        for name in _literals(keyword.value):
                            note(name, where)
            # record["blocker"] = "blocker_name", record["oligo_blocker"] = ...,
            # and the tuple-unpacking spelling of the same assignment.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    pairs: list[tuple[ast.expr, ast.expr]] = []
                    if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                        pairs = list(zip(target.elts, node.value.elts, strict=False))
                    else:
                        pairs = [(target, node.value)]
                    for slot, value in pairs:
                        if not (isinstance(slot, ast.Subscript) and isinstance(slot.slice, ast.Constant)):
                            continue
                        key = slot.slice.value
                        if isinstance(key, str) and (key == "blocker" or key.endswith("_blocker")):
                            for name in _literals(value):
                                note(name, where)
            # {"blocker": "blocker_name"} and {"solubility_blocker": ...}
            if isinstance(node, ast.Dict):
                for key_node, value_node in zip(node.keys, node.values, strict=False):
                    if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                        continue
                    key = key_node.value
                    if key == "blocker" or key.endswith("_blocker"):
                        for name in _literals(value_node):
                            note(name, where)
            # return "blocker_name", bar, detail
            if _returns_a_blocker_first(node):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Tuple) and inner.value.elts:
                        for name in _literals(inner.value.elts[0]):
                            note(name, where)
            # def _designs_blocker(...) -> str: return "blocker_name"
            if _names_a_blocker(node):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and inner.value is not None:
                        for name in _literals(inner.value):
                            note(name, where)
    return found


_FOUND = _blocker_names()


def test_the_scan_still_finds_the_blocker_names() -> None:
    """Guards the two tests below against passing on an empty set."""
    assert len(_FOUND) >= _FLOOR, (
        f"the blocker scan found only {len(_FOUND)} names, below the floor of {_FLOOR}; "
        "the modules have probably moved how they name a blocker, and the two "
        "tests below are no longer checking anything"
    )


@pytest.mark.parametrize("name", sorted(_FOUND))
def test_every_named_blocker_has_a_sentence_on_its_own_surface(name: str) -> None:
    """A name with no sentence renders as its own slug and explains nothing.

    Checked once per emitting file, against that file's surface. A name emitted
    in two places has to be explained in both, because each site renders through
    its own table: ``no_chains`` is written by the retriever and by the
    comparison, and a reader meeting it on either page needs words there.
    """
    for where in sorted(_FOUND[name]):
        vocabulary = _vocabulary_for(where)
        assert any(name in table for _, table in vocabulary), (
            f"{name} is named as a blocker in {where}, but "
            f"{' / '.join(label for label, _ in vocabulary)} has no sentence for it, so the "
            f"reader is shown '{name.replace('_', ' ')}' and nothing else. Add a line there. "
            "On the board, add an action to _ACTIONS too unless there is genuinely nothing "
            "the reader can do; the results surfaces carry their advice in the sentence."
        )


def test_every_action_belongs_to_a_blocker() -> None:
    """An action keyed to no blocker is advice that can never be shown."""
    orphans = sorted(set(screen_board._ACTIONS) - set(screen_board._BLOCKERS))
    assert not orphans, f"actions with no blocker of that name: {orphans}"


#: Board sentences for blockers this scan cannot locate. Each one reaches the
#: board through a shape no signal-free scan can recognise -- a name held in a
#: variable, a literal appended into a structure that is read as a blocker
#: later, a bare ``return "name"`` from a function whose name follows neither
#: convention -- and each is covered by its own module's tests instead.
#:
#: This list is a ratchet, not an allowlist to grow. The scan cannot be made to
#: see those shapes without matching ordinary strings too, which was measured
#: and reverted (see _BLOCKER_TEXT). What it can do is refuse to let a
#: twenty-fifth appear unnoticed: a new name here means a blocker was added in a
#: shape nothing checks, and the fix is almost always to emit it through a
#: helper that takes a ``blocker`` argument, or as a ``blocker`` key, so the
#: scan sees it and this list gets shorter rather than longer.
_UNSEEN_BY_THE_SCAN = frozenset(
    {
        "designs_compressed",
        "metrics_columns_repeated",
        "metrics_not_configured",
        "metrics_not_reported",
        "metrics_row_absent",
        "metrics_row_ambiguous",
        "metrics_row_too_long",
        "metrics_row_too_short",
        "metrics_too_large",
        "metrics_unreadable",
        "no_nine_mer_ligand",
        # The next three are handed back by ``refold_export.design_fasta`` and
        # ``binder_fasta`` as a bare ``return "name"`` from a function named for
        # neither convention -- the third shape the docstring above discloses.
        # They reach the reader through ``results_actions._why``, which is why
        # that table spells them out even though nothing can demand it.
        "no_polymer_sequence",
        "no_presented_ligand",
        "not_requested",
        "oracle_not_configured",
        "over_residue_limit",
        "target_accession_ambiguous",
        "too_many_chains",
        "toxicology_not_configured",
        "toxicology_unreadable",
        "unscored_windows",
    }
)


def test_no_new_blocker_hides_in_a_shape_the_scan_cannot_see() -> None:
    """The other direction, which is what catches an uncoverable shape.

    Every test above asks "does this scanned name have a sentence". None of them
    can ask the question that matters more -- was a blocker added in a shape the
    scan is blind to -- because a shape with no signal cannot be searched for.

    It can be detected from the board instead. A sentence written for a name the
    scan never produces is the footprint of exactly that: somebody added a
    blocker, saw it render as its own slug, and wrote the sentence. Pinning the
    set turns the next one into a failure.
    """
    unseen = sorted(_ALL_SENTENCES - set(_FOUND))
    added = sorted(set(unseen) - _UNSEEN_BY_THE_SCAN)
    assert not added, (
        f"these sentences name blockers the scan cannot find: {added}. "
        "That means they are emitted in a shape nothing checks -- a name held in "
        "a variable, appended into a structure, or returned bare. Emit the name "
        "through a helper with a `blocker` argument or as a `blocker` key so the "
        "scan sees it, or add it to _UNSEEN_BY_THE_SCAN with the reason."
    )


def test_the_unseen_list_does_not_outlive_what_it_describes() -> None:
    """A name that became visible must leave the list, or it hides a real gap.

    Without this the ratchet only ever loosens: a blocker reworked to go through
    ``_axis`` would stay listed as unseen, and the list would slowly become a
    place where anything may be parked.
    """
    stale = sorted(_UNSEEN_BY_THE_SCAN & set(_FOUND))
    assert not stale, (
        f"the scan now finds these, so they are no longer unseen: {stale}. Remove " "them from _UNSEEN_BY_THE_SCAN."
    )


def test_the_unseen_list_names_only_real_sentences() -> None:
    """A deleted blocker must not leave an entry behind to excuse a future one."""
    ghosts = sorted(_UNSEEN_BY_THE_SCAN - _ALL_SENTENCES)
    assert not ghosts, f"_UNSEEN_BY_THE_SCAN names blockers no surface has a sentence for: {ghosts}"
