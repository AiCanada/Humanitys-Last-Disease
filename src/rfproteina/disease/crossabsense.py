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


def oracle_bytes() -> int:
    """Bytes of the five ``final.ckpt`` files."""
    return sum(ORACLE_FINAL_BYTES.values())


def required_bytes() -> int:
    """Bytes that have to land on disk before all five axes can run."""
    return ESMC_6B_BYTES + ESMC_300M_BYTES + PROTT5_BYTES + oracle_bytes()


def download_permitted(cap: int = CAP_BYTES) -> bool:
    """False when the full download reaches ``cap``. Nothing is fetched here."""
    return required_bytes() < cap
