import json

from rdkit import Chem

from neogen.validation.tier4_stereo import evaluate_chiral_directionality, load_chiral_cohort


class PerfectDirectionModel:
    def __init__(self, cohort):
        self.values = {}
        for row in cohort:
            if row["published_direction"] == "S>R":
                self.values[row["r_isomeric_smiles"]] = 1.0
                self.values[row["s_isomeric_smiles"]] = float(row["published_value"])
            else:
                self.values[row["r_isomeric_smiles"]] = float(row["published_value"])
                self.values[row["s_isomeric_smiles"]] = 1.0

    def predict_pair_member(self, *, smiles, endpoint, features):
        return self.values[smiles]


class InvertedDirectionModel(PerfectDirectionModel):
    def __init__(self, cohort):
        self.values = {}
        for row in cohort:
            if row["published_direction"] == "S>R":
                self.values[row["r_isomeric_smiles"]] = float(row["published_value"])
                self.values[row["s_isomeric_smiles"]] = 1.0
            else:
                self.values[row["r_isomeric_smiles"]] = 1.0
                self.values[row["s_isomeric_smiles"]] = float(row["published_value"])


def test_chiral_cohort_v1_parses_has_nine_pairs_and_explicit_stereo():
    rows = load_chiral_cohort()

    assert len(rows) == 9
    assert [row["pair_id"] for row in rows] == sorted(row["pair_id"] for row in rows)
    for row in rows:
        assert row["citation"]
        assert row["confidence"] in {"high", "medium", "low"}
        for key in ("r_isomeric_smiles", "s_isomeric_smiles"):
            assert "@" in row[key]
            assert Chem.MolFromSmiles(row[key]) is not None


def test_tier4_evaluator_runs_on_chiral_cohort():
    rows = load_chiral_cohort()
    result = evaluate_chiral_directionality(PerfectDirectionModel(rows), rows)

    assert len(result.per_pair_breakdown) == 9
    assert result.direction_pass_rate == 100.0


def test_tier4_pass_criterion_fires_for_perfect_mock_model():
    rows = load_chiral_cohort()
    result = evaluate_chiral_directionality(PerfectDirectionModel(rows), rows)

    assert result.overall_pass is True
    assert result.direction_pass_rate >= 80.0
    assert result.magnitude_pass_rate >= 60.0


def test_tier4_fail_criterion_fires_for_inverted_mock_model():
    rows = load_chiral_cohort()
    result = evaluate_chiral_directionality(InvertedDirectionModel(rows), rows)

    assert result.overall_pass is False
    assert result.direction_pass_rate == 0.0
