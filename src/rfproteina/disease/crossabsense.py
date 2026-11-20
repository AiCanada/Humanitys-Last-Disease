"""Size gate for CrossAbSense. The oracles are not downloaded.

The five v0.9 heads on ``SimonCrouzet/CrossAbSense`` are Apache-2.0 and small.
Running them needs the encoders named in each property's published config,
measured here on 2026-11-17 from Hugging Face tree sizes and ``Content-Length``:

* ESM-C 6B (``biohub/esmc-6b-2024-12`` safetensors) for HIC, Titer, AC-SINS and Tm2
* ESM-C 300M (``EvolutionaryScale/esmc-300m-2024-12``) for PR_CHO
* ProtT5 XL, one ``pytorch_model.bin`` (``Rostlab/prot_t5_xl_uniref50``) for AC-SINS and Tm2

The 6B file set is already over 25 GB. The full set the five axes need is
larger. :func:`download_permitted` is therefore false, and nothing is fetched.
Fold checkpoints are not in the total: prediction uses ``final.ckpt`` only.
"""

from __future__ import annotations

#: The add cap. A download that reaches this is not started.
CAP_BYTES = 25_000_000_000

#: Six safetensors shards. There is no separate fp16 file in that repository.
ESMC_6B_BYTES = 25_408_195_509
ESMC_300M_BYTES = 1_332_095_738
#: One weight file. The repository also carries two more copies of the same bin.
PROTT5_BYTES = 11_275_562_724
#: ``final.ckpt`` only, from HTTP Content-Length. Fold checkpoints are omitted.
ORACLE_FINAL_BYTES: dict[str, int] = {
    "HIC": 123_891_008,
    "PR_CHO": 211_554_214,
    "AC-SINS_pH7.4": 167_980_725,
    "Tm2": 363_493_027,
    "Titer": 495_799_530,
}
#: Which published encoder each v0.9 head loads. Taken from that head's config.yaml.
ORACLE_ENCODERS: dict[str, tuple[str, ...]] = {
    "HIC": ("esmc_6b",),
    "PR_CHO": ("esmc_300m",),
    "AC-SINS_pH7.4": ("esmc_6b", "prott5"),
    "Tm2": ("esmc_6b", "prott5"),
    "Titer": ("esmc_6b",),
}


#: Each published encoder's size, so a per-property total can be taken without
#: restating a number that already exists above.
ENCODER_BYTES: dict[str, int] = {
    "esmc_6b": ESMC_6B_BYTES,
    "esmc_300m": ESMC_300M_BYTES,
    "prott5": PROTT5_BYTES,
}


def oracle_bytes() -> int:
    """Bytes of the five ``final.ckpt`` files."""
    return sum(ORACLE_FINAL_BYTES.values())


def property_bytes(name: str) -> int:
    """Bytes one property's head needs: its own ``final.ckpt`` plus its encoders.

    Raises:
        KeyError: If ``name`` is not one of the five heads, rather than
            returning a zero that would read as "nothing to download".
    """
    encoders = ORACLE_ENCODERS[name]
    return ORACLE_FINAL_BYTES[name] + sum(ENCODER_BYTES[encoder] for encoder in encoders)


def property_permitted(name: str, cap: int = CAP_BYTES) -> bool:
    """Whether one property alone fits under ``cap``.

    :func:`download_permitted` asks whether *all five* fit, and they do not, so
    it is false as an all-or-nothing fact. That hides a per-property answer:
    ``PR_CHO`` loads only ESM-C 300M and fits comfortably, while every other
    head pulls ESM-C 6B and does not. Asking per property makes the gate a
    measurement instead of a single refusal covering five different cases.

    Fitting is not a reason to fetch it. ``PR_CHO`` is a primary column of the
    corpus CrossAbSense was fitted on, so a number from it is the most
    contaminated one this program could produce; see
    ``corpus_provenance.overlap_with``.
    """
    return property_bytes(name) < cap


def required_bytes() -> int:
    """Bytes that have to land on disk before all five axes can run."""
    return ESMC_6B_BYTES + ESMC_300M_BYTES + PROTT5_BYTES + oracle_bytes()


def download_permitted(cap: int = CAP_BYTES) -> bool:
    """False when the full download reaches ``cap``. Nothing is fetched here."""
    return required_bytes() < cap
