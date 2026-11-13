"""In-process conformer worker queue for Week 3."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from neogen.chem.conformers import ConformerConfig, ConformerSet, generate_conformer_set
from neogen.chem.identity import Stereoisomer


@dataclass(frozen=True)
class ConformerJob:
    isomer: Stereoisomer
    config: ConformerConfig


queue: asyncio.Queue[ConformerJob] = asyncio.Queue()


async def enqueue(job: ConformerJob) -> None:
    await queue.put(job)


async def worker_loop() -> None:
    while True:
        job = await queue.get()
        try:
            generate_conformer_set(job.isomer, job.config)
        finally:
            queue.task_done()


def should_run_sync(isomer: Stereoisomer, config: ConformerConfig) -> bool:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(isomer.isomeric_smiles)
    return bool(mol is not None and mol.GetNumHeavyAtoms() <= 30 and config.num_candidates <= 20)
