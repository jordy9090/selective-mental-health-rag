import pytest

from src.gate import decide_retrieval
from scripts.plot_calibration import simulated_route


@pytest.mark.parametrize(
    ("r_safe", "scores", "retrieve", "route"),
    [
        (1, {}, True, "safety"),
        (0, {"u_info": 4, "u_cope": 1, "u_spec": 1}, True, "psychoeducation"),
        (0, {"u_info": 1, "u_cope": 4, "u_spec": 1}, True, "coping"),
        (0, {"u_info": 3, "u_cope": 3, "u_spec": 4}, True, "all_non_safety"),
        (0, {"u_info": 3, "u_cope": 3, "u_spec": 3}, False, "none"),
    ],
)
def test_decide_retrieval(r_safe, scores, retrieve, route):
    decision = decide_retrieval(r_safe, scores)
    assert decision["retrieve"] is retrieve
    assert decision["route"] == route
    assert decision["mean_threshold"] == 3.25
    assert decision["route_threshold"] == 4.0


def test_high_axis_tie_routes_to_coping():
    decision = decide_retrieval(0, {"u_info": 4, "u_cope": 4, "u_spec": 1})
    assert decision["high_axis_gate"] is True
    assert decision["mean_gate"] is False
    assert decision["route"] == "coping"


def test_thresholds_are_configurable():
    decision = decide_retrieval(
        0,
        {"u_info": 3, "u_cope": 2, "u_spec": 2},
        mean_threshold=2.25,
        route_threshold=3.0,
    )
    assert decision["retrieve"] is True
    assert decision["route"] == "psychoeducation"
    assert decision["high_axis_gate"] is True
    assert decision["mean_gate"] is True


def test_calibration_keeps_high_axis_low_mean_activation():
    route = simulated_route(
        {"r_safe": 0, "u_info": 4, "u_cope": 1, "u_spec": 1, "mean_need": 2},
        tau=3.25,
        route_tau=4.0,
    )
    assert route == "psychoeducation"
