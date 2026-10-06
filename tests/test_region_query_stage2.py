"""Stage-2 typed Boolean regional-profile query tests."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_neuron_navigator.analysis.region_profile import (
    Compartment,
    RegionalProfileAtlas,
    build_region_profile,
    resolve_physical_midline,
)
from napari_neuron_navigator.analysis.region_query import (
    AllOf,
    AnyOf,
    NeuriteIntersects,
    Not,
    QueryLaterality,
    QueryScope,
    RegionalMeasurement,
    RegionalThreshold,
    RegionQueryEngine,
    RegionQueryValidationError,
    RegionReference,
    RegionSelection,
    SomaIn,
    ThresholdComparison,
    UnknownRegionError,
    deserialize_query,
    serialize_query,
)


def _query_atlas() -> RegionalProfileAtlas:
    annotation = np.empty((6, 1, 4), dtype=np.int32)
    for index, region_id in enumerate((11, 12, 31, 32, 41, 0)):
        annotation[index, :, :] = region_id
    hemispheres = np.empty(annotation.shape, dtype=np.uint8)
    hemispheres[:, :, :2] = 2
    hemispheres[:, :, 2:] = 1
    resolution = (10.0, 10.0, 10.0)
    midline, lower_code, upper_code = resolve_physical_midline(
        hemispheres,
        resolution,
        left_right_axis=2,
    )
    structures = {
        1: {
            "id": 1,
            "acronym": "root",
            "name": "root",
            "structure_id_path": [1],
        },
        10: {
            "id": 10,
            "acronym": "MO",
            "name": "Motor areas",
            "structure_id_path": [1, 10],
        },
        11: {
            "id": 11,
            "acronym": "MOp5",
            "name": "Primary motor area, layer 5",
            "structure_id_path": [1, 10, 11],
        },
        12: {
            "id": 12,
            "acronym": "MOs5",
            "name": "Secondary motor area, layer 5",
            "structure_id_path": [1, 10, 12],
        },
        30: {
            "id": 30,
            "acronym": "CP",
            "name": "Caudoputamen",
            "structure_id_path": [1, 30],
        },
        31: {
            "id": 31,
            "acronym": "CPa",
            "name": "Caudoputamen fixture A",
            "structure_id_path": [1, 30, 31],
        },
        32: {
            "id": 32,
            "acronym": "CPb",
            "name": "Caudoputamen fixture B",
            "structure_id_path": [1, 30, 32],
        },
        40: {
            "id": 40,
            "acronym": "pons",
            "name": "Pons",
            "structure_id_path": [1, 40],
        },
        41: {
            "id": 41,
            "acronym": "PON-fixture",
            "name": "Pons fixture child",
            "structure_id_path": [1, 40, 41],
        },
    }
    return RegionalProfileAtlas(
        atlas_name="stage2_fixture",
        atlas_version="1.0",
        annotation=annotation,
        resolution_um=resolution,
        structures=structures,
        left_right_axis=2,
        midline_um=midline,
        lower_hemisphere_code=lower_code,
        upper_hemisphere_code=upper_code,
    )


def _query_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_neuron(
        file_id: str,
        neuron_id: str,
        subject: str,
        soma_xyz: tuple[float, float, float],
        soma_region: int,
        child_xyz: tuple[float, float, float] | None,
        child_region: int | None,
    ) -> None:
        rows.append(
            {
                "file_id": file_id,
                "neuron_id": neuron_id,
                "subject": subject,
                "node_id": 100,
                "type": 1,
                "x": soma_xyz[0],
                "y": soma_xyz[1],
                "z": soma_xyz[2],
                "parent_id": -1,
                "region_id": soma_region,
            }
        )
        if child_xyz is not None:
            rows.append(
                {
                    "file_id": file_id,
                    "neuron_id": neuron_id,
                    "subject": subject,
                    "node_id": 900,
                    "type": 0,
                    "x": child_xyz[0],
                    "y": child_xyz[1],
                    "z": child_xyz[2],
                    "parent_id": 100,
                    "region_id": child_region,
                }
            )

    # A and its mirror both project from motor cortex into contralateral CP.
    add_neuron(
        "contra-lower.swc",
        "duplicate-display-id",
        "subject-a",
        (5.0, 5.0, 5.0),
        11,
        (25.0, 5.0, 25.0),
        31,
    )
    add_neuron(
        "contra-upper.swc",
        "mirror-display-id",
        "subject-g",
        (5.0, 5.0, 35.0),
        11,
        (25.0, 5.0, 15.0),
        31,
    )
    # B and E traverse both CP child regions before reaching ipsilateral pons.
    add_neuron(
        "pons-upper.swc",
        "pons-upper",
        "subject-b",
        (15.0, 5.0, 35.0),
        12,
        (45.0, 5.0, 35.0),
        41,
    )
    add_neuron(
        "pons-lower.swc",
        "pons-lower",
        "subject-e",
        (5.0, 5.0, 5.0),
        11,
        (45.0, 5.0, 5.0),
        41,
    )
    add_neuron(
        "ipsi-cp.swc",
        "duplicate-display-id",
        "subject-c",
        (5.0, 5.0, 5.0),
        11,
        (25.0, 5.0, 5.0),
        31,
    )
    add_neuron(
        "cp-soma.swc",
        "cp-soma",
        "subject-d",
        (25.0, 5.0, 5.0),
        31,
        (25.0, 5.0, 25.0),
        31,
    )
    # A one-node tree proves that a soma-only bucket cannot satisfy neurite.
    add_neuron(
        "pons-soma-only.swc",
        "pons-soma-only",
        "subject-h",
        (45.0, 5.0, 5.0),
        41,
        None,
        None,
    )
    return pd.DataFrame(rows)


def _catalog(source: pd.DataFrame) -> pd.DataFrame:
    return (
        source[["file_id", "neuron_id", "subject"]]
        .drop_duplicates("file_id")
        .sort_values("file_id")
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def query_context(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("region-query-stage2")
    source_path = root / "neurons.parquet"
    sidecar_path = root / "neurons.region_profile.parquet"
    source = _query_source()
    source.to_parquet(source_path, index=False)
    atlas = _query_atlas()
    build_region_profile(source_path, atlas, output_path=sidecar_path)
    connection = duckdb.connect()
    engine = RegionQueryEngine(
        connection,
        sidecar_path=sidecar_path,
        source_path=source_path,
        atlas=atlas,
        catalog=_catalog(source),
    )
    try:
        yield connection, engine, source, atlas, source_path, sidecar_path
    finally:
        connection.close()


def _selection(
    region_ids: list[int],
    atlas: RegionalProfileAtlas,
    *,
    descendants: bool = False,
) -> RegionSelection:
    return RegionSelection.from_ids(
        region_ids,
        include_descendants=descendants,
        structures=atlas.structures,
    )


def _file_ids(result) -> list[str]:
    return result.rows["file_id"].tolist()


def test_query_serialization_round_trips_all_semantics(
    query_context,
) -> None:
    _connection, _engine, _source, atlas, *_paths = query_context
    query = AllOf(
        (
            SomaIn(_selection([11, 12], atlas)),
            AnyOf(
                (
                    NeuriteIntersects(
                        _selection([30], atlas, descendants=True),
                        QueryLaterality.CONTRALATERAL,
                    ),
                    Not(
                        RegionalThreshold(
                            compartment=Compartment.NEURITE,
                            measurement=RegionalMeasurement.TERMINUS_COUNT,
                            comparison=ThresholdComparison.GREATER_THAN_OR_EQUAL,
                            value=2,
                            region_selection=_selection(
                                [40], atlas, descendants=True
                            ),
                            laterality=QueryLaterality.IPSILATERAL,
                        )
                    ),
                )
            ),
        )
    )

    serialized = serialize_query(query)

    assert deserialize_query(serialized) == query
    assert serialize_query(deserialize_query(serialized)) == serialized
    assert '"format_version":1' in serialized
    assert '"include_descendants":true' in serialized
    assert '"laterality":"contralateral"' in serialized


def test_query_models_reject_invalid_or_ambiguous_values() -> None:
    with pytest.raises(RegionQueryValidationError, match="at least one"):
        RegionSelection(())
    with pytest.raises(RegionQueryValidationError, match="repeat"):
        RegionSelection((RegionReference(11), RegionReference(11)))
    with pytest.raises(RegionQueryValidationError, match="at least one child"):
        AllOf(())
    with pytest.raises(RegionQueryValidationError, match="cannot contain null"):
        QueryScope.explicit([None])
    with pytest.raises(RegionQueryValidationError, match="finite"):
        RegionalThreshold(
            compartment=Compartment.NEURITE,
            measurement=RegionalMeasurement.CABLE_LENGTH_UM,
            comparison=ThresholdComparison.GREATER_THAN,
            value=float("nan"),
            region_selection=RegionSelection((RegionReference(11),)),
        )
    with pytest.raises(RegionQueryValidationError, match="unsupported"):
        deserialize_query('{"format_version":999,"query":{}}')


def test_two_motivating_queries_and_relative_laterality(query_context) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    motor_somas = SomaIn(_selection([11, 12], atlas))
    contralateral_cp = NeuriteIntersects(
        _selection([30], atlas, descendants=True),
        QueryLaterality.CONTRALATERAL,
    )
    ipsilateral_pons = NeuriteIntersects(
        _selection([40], atlas, descendants=True),
        QueryLaterality.IPSILATERAL,
    )

    contra_result = engine.execute(AllOf((motor_somas, contralateral_cp)))
    pons_result = engine.execute(AllOf((motor_somas, ipsilateral_pons)))

    assert _file_ids(contra_result) == [
        "contra-lower.swc",
        "contra-upper.swc",
    ]
    assert _file_ids(pons_result) == ["pons-lower.swc", "pons-upper.swc"]
    assert contra_result.metadata.matched_count == 2
    assert contra_result.metadata.returned_catalog_count == 2
    assert contra_result.metadata.atlas_identity == engine.profile_metadata.atlas
    assert contra_result.metadata.sidecar_identity == engine.sidecar_identity


def test_parent_resolution_deduplicates_overlapping_parent_and_child(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    parent_and_child = _selection([30, 31], atlas, descendants=True)
    parent_only = _selection([30], atlas, descendants=True)
    overlapping_query = RegionalThreshold(
        compartment=Compartment.NEURITE,
        measurement=RegionalMeasurement.CABLE_LENGTH_UM,
        comparison=ThresholdComparison.GREATER_THAN,
        value=15,
        region_selection=parent_and_child,
    )
    parent_query = RegionalThreshold(
        compartment=Compartment.NEURITE,
        measurement=RegionalMeasurement.CABLE_LENGTH_UM,
        comparison=ThresholdComparison.GREATER_THAN,
        value=15,
        region_selection=parent_only,
    )

    explanation = engine.explain(overlapping_query)

    assert explanation.conditions[0].resolved_direct_region_ids == (30, 31, 32)
    assert _file_ids(engine.execute(overlapping_query)) == _file_ids(
        engine.execute(parent_query)
    )


def test_length_threshold_sums_all_descendants_before_having(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    parent_query = RegionalThreshold(
        compartment=Compartment.NEURITE,
        measurement=RegionalMeasurement.CABLE_LENGTH_UM,
        comparison=ThresholdComparison.GREATER_THAN,
        value=15,
        region_selection=_selection([30], atlas, descendants=True),
    )
    first_child_query = RegionalThreshold(
        compartment=Compartment.NEURITE,
        measurement=RegionalMeasurement.CABLE_LENGTH_UM,
        comparison=ThresholdComparison.GREATER_THAN,
        value=15,
        region_selection=_selection([31], atlas),
    )

    assert _file_ids(engine.execute(parent_query)) == [
        "cp-soma.swc",
        "pons-lower.swc",
        "pons-upper.swc",
    ]
    assert _file_ids(engine.execute(first_child_query)) == ["cp-soma.swc"]


@pytest.mark.parametrize(
    ("comparison", "value", "expected"),
    [
        (
            ThresholdComparison.GREATER_THAN,
            0,
            ["pons-lower.swc", "pons-upper.swc"],
        ),
        (
            ThresholdComparison.GREATER_THAN_OR_EQUAL,
            1,
            ["pons-lower.swc", "pons-upper.swc"],
        ),
        (
            ThresholdComparison.LESS_THAN,
            1,
            [
                "contra-lower.swc",
                "contra-upper.swc",
                "cp-soma.swc",
                "ipsi-cp.swc",
                "pons-soma-only.swc",
            ],
        ),
        (
            ThresholdComparison.LESS_THAN_OR_EQUAL,
            0,
            [
                "contra-lower.swc",
                "contra-upper.swc",
                "cp-soma.swc",
                "ipsi-cp.swc",
                "pons-soma-only.swc",
            ],
        ),
        (
            ThresholdComparison.EQUAL,
            0,
            [
                "contra-lower.swc",
                "contra-upper.swc",
                "cp-soma.swc",
                "ipsi-cp.swc",
                "pons-soma-only.swc",
            ],
        ),
    ],
)
def test_threshold_comparisons_treat_absent_buckets_as_zero(
    query_context,
    comparison: ThresholdComparison,
    value: float,
    expected: list[str],
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    query = RegionalThreshold(
        compartment=Compartment.NEURITE,
        measurement=RegionalMeasurement.NODE_COUNT,
        comparison=comparison,
        value=value,
        region_selection=_selection([40], atlas, descendants=True),
    )

    assert _file_ids(engine.execute(query)) == expected


def test_boolean_set_operations_nesting_and_scope_relative_not(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    motor = SomaIn(_selection([10], atlas, descendants=True))
    contra_cp = NeuriteIntersects(
        _selection([30], atlas, descendants=True),
        QueryLaterality.CONTRALATERAL,
    )

    assert _file_ids(engine.execute(AllOf((motor, contra_cp)))) == [
        "contra-lower.swc",
        "contra-upper.swc",
    ]
    assert _file_ids(engine.execute(AnyOf((motor, contra_cp)))) == [
        "contra-lower.swc",
        "contra-upper.swc",
        "cp-soma.swc",
        "ipsi-cp.swc",
        "pons-lower.swc",
        "pons-upper.swc",
    ]
    result = engine.execute(
        Not(contra_cp),
        scope=QueryScope.explicit(
            ["contra-lower.swc", "pons-upper.swc", "cp-soma.swc"]
        ),
    )
    assert _file_ids(result) == ["pons-upper.swc"]
    nested = engine.execute(AllOf((motor, Not(contra_cp))))
    assert _file_ids(nested) == [
        "ipsi-cp.swc",
        "pons-lower.swc",
        "pons-upper.swc",
    ]


def test_empty_explicit_scope_returns_typed_empty_result_without_sql_error(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context

    result = engine.execute(
        SomaIn(_selection([11], atlas)), scope=QueryScope.explicit([])
    )

    assert result.rows.empty
    assert list(result.rows.columns) == ["file_id", "neuron_id", "subject"]
    assert result.metadata.matched_count == 0


def test_soma_and_neurite_compartments_cannot_substitute_for_each_other(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context
    pons = _selection([40], atlas, descendants=True)

    assert _file_ids(engine.execute(SomaIn(pons))) == ["pons-soma-only.swc"]
    assert _file_ids(engine.execute(NeuriteIntersects(pons))) == [
        "pons-lower.swc",
        "pons-upper.swc",
    ]


def test_direct_any_node_compatibility_matches_raw_source_query(
    query_context,
) -> None:
    _connection, engine, source, atlas, *_paths = query_context
    direct_cp = _selection([31], atlas)
    compatibility_query = AnyOf(
        (
            SomaIn(direct_cp),
            RegionalThreshold(
                compartment=Compartment.NEURITE,
                measurement=RegionalMeasurement.NODE_COUNT,
                comparison=ThresholdComparison.GREATER_THAN,
                value=0,
                region_selection=direct_cp,
            ),
        )
    )
    expected = sorted(source.loc[source["region_id"] == 31, "file_id"].unique())

    assert _file_ids(engine.execute(compatibility_query)) == expected


def test_unknown_region_fails_during_explanation_before_execution(
    query_context,
) -> None:
    connection, engine, _source, _atlas, *_paths = query_context
    query = SomaIn(RegionSelection((RegionReference(999, "foreign"),)))

    with pytest.raises(UnknownRegionError, match="999"):
        engine.execute(query)

    assert connection.execute(
        "SELECT COUNT(*) FROM region_query_catalog"
    ).fetchone()[0] == 7


def test_results_remain_unique_by_file_id_when_display_ids_repeat(
    query_context,
) -> None:
    _connection, engine, _source, atlas, *_paths = query_context

    result = engine.execute(SomaIn(_selection([10], atlas, descendants=True)))

    assert result.rows["file_id"].is_unique
    duplicate_display = result.rows.loc[
        result.rows["neuron_id"] == "duplicate-display-id"
    ]
    assert set(duplicate_display["file_id"]) == {
        "contra-lower.swc",
        "ipsi-cp.swc",
    }


def test_execution_uses_profile_and_catalog_without_a_source_node_relation(
    query_context,
) -> None:
    connection, engine, _source, atlas, *_paths = query_context
    table_names = {
        str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()
    }

    assert "neurons" not in table_names
    assert {"region_profile", "region_query_catalog"} <= table_names
    assert _file_ids(engine.execute(SomaIn(_selection([11], atlas))))


def test_missing_catalog_rows_are_reported_without_display_id_substitution(
    query_context,
) -> None:
    connection, _engine, source, atlas, source_path, sidecar_path = query_context
    incomplete_catalog = _catalog(source).loc[
        lambda frame: frame["file_id"] != "pons-soma-only.swc"
    ]
    engine = RegionQueryEngine(
        connection,
        sidecar_path=sidecar_path,
        source_path=source_path,
        atlas=atlas,
        catalog=incomplete_catalog,
        profile_table_name="region_profile_incomplete_catalog",
        catalog_table_name="region_query_incomplete_catalog",
    )

    result = engine.execute(SomaIn(_selection([40], atlas, descendants=True)))

    assert result.rows.empty
    assert result.metadata.matched_count == 1
    assert result.metadata.returned_catalog_count == 0
    assert result.metadata.missing_catalog_file_ids == ("pons-soma-only.swc",)


def test_display_metadata_cannot_change_sql_structure(query_context) -> None:
    connection, engine, _source, _atlas, *_paths = query_context
    selection = RegionSelection(
        (
            RegionReference(
                11,
                acronym="MOp5'); DROP TABLE region_profile; --",
                name="untrusted display text",
            ),
        )
    )

    result = engine.execute(SomaIn(selection))

    assert _file_ids(result)
    assert connection.execute("SELECT COUNT(*) FROM region_profile").fetchone()[0] > 0
