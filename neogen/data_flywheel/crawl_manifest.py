"""Crawl manifest schema.

Every crawl/extraction run produces a ``CrawlManifest`` — an auditable record
of what was attempted, what succeeded, what was written, and what failed. This
is the bookkeeping that makes the flywheel replayable and accountable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CrawlManifest:
    """Immutable record of a single crawl/extraction run."""

    crawl_id: str
    source_id: str
    started_at: str  # ISO-8601 UTC
    completed_at: str | None
    tool_used: str  # e.g. "firecrawl_cloud", "webfetch", "playwright", "api"
    urls_attempted: tuple[str, ...] = field(default_factory=tuple)
    urls_successful: tuple[str, ...] = field(default_factory=tuple)
    files_written: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    next_action: str | None = None

    def __post_init__(self) -> None:
        if not self.crawl_id:
            raise ValueError("CrawlManifest.crawl_id is required")
        if not self.source_id:
            raise ValueError("CrawlManifest.source_id is required")
        if not self.started_at:
            raise ValueError("CrawlManifest.started_at is required")
        if not self.tool_used:
            raise ValueError("CrawlManifest.tool_used is required")
        # urls_successful must be a subset of urls_attempted.
        attempted = set(self.urls_attempted)
        extra = [u for u in self.urls_successful if u not in attempted]
        if extra:
            raise ValueError(
                f"urls_successful contains URLs not in urls_attempted: {extra}"
            )

    @property
    def success_rate(self) -> float:
        if not self.urls_attempted:
            return 0.0
        return len(self.urls_successful) / len(self.urls_attempted)

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("urls_attempted", "urls_successful", "files_written", "errors"):
            d[k] = list(d[k])
        return json.dumps(d, sort_keys=True, indent=2)
