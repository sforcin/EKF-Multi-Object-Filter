"""Check every analytic Jacobian against finite differences.

A wrong Jacobian is the #1 EKF bug: the filter still runs, just badly.
"""
import numpy as np
import pytest

from mot.models import CartesianMeasurement, CoordinatedTurnModel, RangeBearingMeasurement


def numerical_jacobian(fn, x, eps=1e-6):
    fx = fn(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (fn(x + dx) - fn(x - dx)) / (2 * eps)
    return J


@pytest.mark.parametrize("w", [0.0, 1e-8, 9e-4, 2e-3, 0.05, -0.3, 1.2])
def test_ct_jacobian_matches_finite_difference(w):
    m = CoordinatedTurnModel()
    x = np.array([100.0, 200.0, 30.0, -12.0, w])
    J_num = numerical_jacobian(lambda s: m.f(s, 0.1), x)
    np.testing.assert_allclose(m.F(x, 0.1), J_num, atol=1e-5)


def test_ct_straight_line_when_no_turn():
    m = CoordinatedTurnModel()
    x = np.array([0.0, 0.0, 10.0, 5.0, 0.0])
    np.testing.assert_allclose(m.f(x, 2.0), [20.0, 10.0, 10.0, 5.0, 0.0])


def test_ct_preserves_speed():
    m = CoordinatedTurnModel()
    x = np.array([0.0, 0.0, 10.0, 5.0, 0.7])
    y = m.f(x, 0.5)
    assert np.isclose(np.hypot(*y[2:4]), np.hypot(*x[2:4]))


def test_q_is_symmetric_psd():
    Q = CoordinatedTurnModel().Q(0.1)
    np.testing.assert_allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)


@pytest.mark.parametrize("sensor", [CartesianMeasurement(), RangeBearingMeasurement()])
def test_measurement_jacobian(sensor):
    x = np.array([730.0, 310.0, 4.0, -3.0, 0.1])
    np.testing.assert_allclose(sensor.H(x), numerical_jacobian(sensor.h, x), atol=1e-6)


def test_bearing_residual_wraps():
    s = RangeBearingMeasurement()
    y = s.residual(np.array([10.0, np.pi - 0.01]), np.array([10.0, -np.pi + 0.01]))
    assert np.isclose(y[1], -0.02)
