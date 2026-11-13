# NeoGen Data Flywheel

The data flywheel is NeoGen's bounded, cited, replayable ingestion layer for the
DDI pipeline. It turns public/licensed biomedical sources into a durable,
provenance-tracked moat that compounds over time.

> **This package does not simulate pharmacology.** The DDI _science engine_
> lives in `neogen/engine/ddi/`. This package handles **identity, structure,
> evidence, provenance, and crawl bookkeeping** that feed that engine.

## Modules

| Module               | Purpose                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `source_registry.py` | Load + validate the YAML source registry (`data/flywheel/`).                                                     |
| `provenance.py`      | `Provenance` — where/when/how each datum was obtained + content hash + reviewer status.                          |
| `identity_spine.py`  | `IdentitySpineRecord` — canonical drug identity + cross-source crosswalk + stereo status.                        |
| `evidence_schema.py` | `DDIEvidenceRecord` — one cited DDI assertion; enforces "no severity without citation" and stereo→schema gating. |
| `crawl_manifest.py`  | `CrawlManifest` — auditable record of each crawl run.                                                            |

## Hard safety rules (enforced in code)

1. **No DDI severity without a citation.** `DDIEvidenceRecord` raises if `severity`
   is set with no `citation`.
2. **Stereo-sensitive records may not use a chirality-blind feature schema.**
   `enantiomer_sensitive` / `diastereomer_sensitive` / `ez_isomer_sensitive`
   records reject `legacy_2d_v1`.
3. **Canonical pair IDs** are computed with the SAME function the science engine
   uses (`neogen.engine.ddi.pair_record.make_pair_id`).
4. **Everything carries provenance.** `Provenance.is_usable_for_claims` is true
   only when human-reviewed AND citable.

## Source registry

- Index: `data/flywheel/source_registry.yaml`
- Per-group entries: `data/flywheel/sources/group{1,2,3}_*.yaml`
- 75 sources across 10 categories; loader merges + validates.

```python
from neogen.data_flywheel.source_registry import load_registry
reg = load_registry()
print(reg.counts())
print([s["source_id"] for s in reg.by_priority("P0")])
```

## Crawl outputs

- `data/flywheel/raw/<source_id>/` — raw/cleaned fetch metadata (NOT full
  copyrighted pages when terms are unclear).
- `data/flywheel/normalized/<source_id>.json` — normalized summary + provenance.

## Compliance posture

Prefer official APIs and public bulk dumps over scraping. Respect robots.txt,
Terms of Service, copyright, paywalls, and rate limits. `crawl_allowed: false`
sources must not be scraped. `license_status: requires_review` /
`commercial` / `restricted` require human clearance before ingestion.
