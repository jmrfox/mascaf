from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKELETONS_DATA = ROOT / "data" / "mcf_skeletons"

TS2_SKELETON_FILENAME = "TS2_qst0.5_mcst5.polylines.txt"


def get_ts2_skeleton_path() -> Path:
    return SKELETONS_DATA / TS2_SKELETON_FILENAME
