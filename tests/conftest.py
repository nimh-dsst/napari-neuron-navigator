"""Shared offline atlas fixtures."""

import json

import pytest
from brainglobe_atlasapi.core import Atlas


@pytest.fixture
def asr_atlas(tmp_path):
    """Use BrainGlobe's real hemisphere implementation, without a download.

    The left-right dimension matches Allen 25 um; the other two are small to
    keep the generated hemisphere volume cheap. No hemisphere labels are mocked.
    """
    atlas_dir = tmp_path / "asr_atlas"
    atlas_dir.mkdir()
    (atlas_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "test_asr",
                "orientation": "asr",
                "shape": [8, 6, 456],
                "resolution": [25.0, 25.0, 25.0],
                "symmetric": True,
                "additional_references": [],
            }
        )
    )
    (atlas_dir / "structures.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "name": "root",
                    "acronym": "root",
                    "structure_id_path": [1],
                    "rgb_triplet": [255, 255, 255],
                }
            ]
        )
    )
    return Atlas(atlas_dir)
