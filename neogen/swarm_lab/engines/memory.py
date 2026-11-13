"""
Multi-level Memory System — retains knowledge across agents, clusters, and generations.

Memory levels:
  - Agent-local: individual agent's experience
  - Cluster: shared within a debate cluster
  - Generation: best results from one generation
  - Lineage: accumulated knowledge of a family line
  - Global: best-ever results across all generations
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    entry_type: str             # "hypothesis" | "critique" | "equation" | "insight" | "failure"
    content: str
    source_agent: str = ""
    source_generation: int = 0
    fitness_score: float = 0.0
    tags: list[str] = field(default_factory=list)


class MemoryStore:
    """Multi-level memory system for the evolutionary swarm."""

    def __init__(self, output_dir: str = "swarm_runs"):
        self.agent_memory: dict[str, list[MemoryEntry]] = {}
        self.cluster_memory: dict[str, list[MemoryEntry]] = {}
        self.generation_memory: dict[int, list[MemoryEntry]] = {}
        self.lineage_memory: dict[str, list[MemoryEntry]] = {}
        self.global_memory: list[MemoryEntry] = []
        self.output_dir = output_dir

    def add_agent_memory(self, agent_id: str, entry: MemoryEntry) -> None:
        self.agent_memory.setdefault(agent_id, []).append(entry)

    def add_cluster_memory(self, cluster_id: str, entry: MemoryEntry) -> None:
        self.cluster_memory.setdefault(cluster_id, []).append(entry)

    def add_generation_memory(self, gen: int, entry: MemoryEntry) -> None:
        self.generation_memory.setdefault(gen, []).append(entry)

    def add_lineage_memory(self, lineage_id: str, entry: MemoryEntry) -> None:
        self.lineage_memory.setdefault(lineage_id, []).append(entry)

    def add_global_memory(self, entry: MemoryEntry) -> None:
        self.global_memory.append(entry)
        # Keep global memory bounded
        if len(self.global_memory) > 100:
            # Keep highest-fitness entries
            self.global_memory.sort(key=lambda e: e.fitness_score, reverse=True)
            self.global_memory = self.global_memory[:100]

    def get_context_for_agent(self, agent_id: str, lineage_id: str,
                               generation: int, max_entries: int = 10) -> str:
        """Build a memory context string for an agent's LLM prompt."""
        entries = []

        # Agent's own memory
        for e in self.agent_memory.get(agent_id, [])[-3:]:
            entries.append(f"[Your memory] {e.content}")

        # Lineage memory (inherited knowledge)
        for e in self.lineage_memory.get(lineage_id, [])[-3:]:
            entries.append(f"[Lineage knowledge] {e.content}")

        # Previous generation's best results
        prev_gen = generation - 1
        for e in self.generation_memory.get(prev_gen, [])[-2:]:
            entries.append(f"[Previous gen best] {e.content}")

        # Global best
        for e in self.global_memory[:2]:
            entries.append(f"[Global best] {e.content}")

        if not entries:
            return ""

        return "MEMORY CONTEXT (from prior work):\n" + "\n".join(entries[:max_entries])

    def promote_best_to_global(self, generation: int) -> None:
        """Promote the best entries from a generation to global memory."""
        gen_entries = self.generation_memory.get(generation, [])
        for entry in gen_entries:
            if entry.fitness_score > 0.6:
                self.add_global_memory(entry)

    def inherit_to_offspring(self, parent_id: str, child_id: str,
                              lineage_id: str) -> None:
        """Transfer relevant memory from parent to offspring."""
        parent_mem = self.agent_memory.get(parent_id, [])
        # Transfer top-scoring memories
        best = sorted(parent_mem, key=lambda e: e.fitness_score, reverse=True)[:5]
        for entry in best:
            inherited = MemoryEntry(
                entry_type=entry.entry_type,
                content=f"[Inherited from ancestor] {entry.content}",
                source_agent=parent_id,
                source_generation=entry.source_generation,
                fitness_score=entry.fitness_score,
                tags=entry.tags,
            )
            self.add_agent_memory(child_id, inherited)
            self.add_lineage_memory(lineage_id, inherited)

    def save(self, filepath: str) -> None:
        """Persist memory to JSON file."""
        data = {
            "global": [asdict(e) for e in self.global_memory],
            "generations": {
                str(k): [asdict(e) for e in v]
                for k, v in self.generation_memory.items()
            },
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> dict:
        """Return summary statistics of the memory system."""
        return {
            "agents_with_memory": len(self.agent_memory),
            "clusters_with_memory": len(self.cluster_memory),
            "generations_tracked": len(self.generation_memory),
            "lineages_tracked": len(self.lineage_memory),
            "global_entries": len(self.global_memory),
        }
