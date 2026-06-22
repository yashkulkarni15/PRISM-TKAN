"""Zero-order Altintas-Budak milling stability-lobe helpers.

The implementation keeps the core deterministic and small: NumPy is the only
runtime dependency. Units are SI unless a function name states otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class ModalParameters:
    """Single degree-of-freedom modal data for one structural direction."""

    natural_frequency_hz: float
    stiffness_n_per_m: float
    damping_ratio: float
    label: str = "x"

    def __post_init__(self) -> None:
        if self.natural_frequency_hz <= 0.0:
            raise ValueError("natural_frequency_hz must be positive")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("stiffness_n_per_m must be positive")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio must be non-negative")

    @property
    def natural_angular_frequency_rad_s(self) -> float:
        return TWO_PI * self.natural_frequency_hz

    def receptance(self, angular_frequency_rad_s: float | np.ndarray) -> complex | np.ndarray:
        """Return dynamic compliance G(jw), in m/N."""

        ratio = np.asarray(angular_frequency_rad_s, dtype=float) / self.natural_angular_frequency_rad_s
        denominator = 1.0 - ratio**2 + 2.0j * self.damping_ratio * ratio
        value = (1.0 / self.stiffness_n_per_m) / denominator
        if np.ndim(value) == 0:
            return complex(value)
        return value


@dataclass(frozen=True)
class CuttingParameters:
    """Tool and force parameters for zero-order milling stability."""

    teeth: int
    tangential_force_coefficient_n_per_m2: float
    radial_to_tangential_ratio: float = 0.0
    radial_immersion: float = 1.0
    cut_dir: str | int = "slot"
    entry_angle_rad: float | None = None
    exit_angle_rad: float | None = None

    def __post_init__(self) -> None:
        if self.teeth <= 0:
            raise ValueError("teeth must be positive")
        if self.tangential_force_coefficient_n_per_m2 <= 0.0:
            raise ValueError("tangential_force_coefficient_n_per_m2 must be positive")
        if self.radial_immersion <= 0.0 or self.radial_immersion > 1.0:
            raise ValueError("radial_immersion must be in (0, 1]")
        if (self.entry_angle_rad is None) ^ (self.exit_angle_rad is None):
            raise ValueError("entry_angle_rad and exit_angle_rad must be provided together")


@dataclass(frozen=True)
class DirectionalFactors:
    """Unnormalized Altintas-Budak directional factor integrals."""

    alpha_xx: float
    alpha_xy: float
    alpha_yx: float
    alpha_yy: float
    entry_angle_rad: float
    exit_angle_rad: float
    cut_dir: str

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.alpha_xx, self.alpha_xy], [self.alpha_yx, self.alpha_yy]],
            dtype=float,
        )


@dataclass(frozen=True)
class StabilityPoint:
    """One candidate point on a zero-order stability lobe."""

    spindle_speed_rpm: float
    axial_depth_m: float
    chatter_frequency_hz: float
    lobe_order: int
    eigenvalue_index: int
    phase_rad: float
    kappa: float
    inverse_eigenvalue_real: float
    inverse_eigenvalue_imag: float

    @property
    def axial_depth_mm(self) -> float:
        return 1000.0 * self.axial_depth_m

    def as_dict(self) -> dict[str, float | int]:
        return {
            "spindle_speed_rpm": self.spindle_speed_rpm,
            "axial_depth_m": self.axial_depth_m,
            "axial_depth_mm": self.axial_depth_mm,
            "chatter_frequency_hz": self.chatter_frequency_hz,
            "lobe_order": self.lobe_order,
            "eigenvalue_index": self.eigenvalue_index,
            "phase_rad": self.phase_rad,
            "kappa": self.kappa,
            "inverse_eigenvalue_real": self.inverse_eigenvalue_real,
            "inverse_eigenvalue_imag": self.inverse_eigenvalue_imag,
        }


def normalize_cut_direction(cut_dir: str | int | float) -> str:
    """Normalize local dataset/textbook cut-direction encodings."""

    if isinstance(cut_dir, (int, float)) and float(cut_dir).is_integer():
        code = int(cut_dir)
        if code == 1:
            return "up"
        if code == 2:
            return "down"
        if code == 0:
            return "slot"

    text = str(cut_dir).strip().lower()
    aliases = {
        "up": "up",
        "conventional": "up",
        "conventional_milling": "up",
        "1": "up",
        "down": "down",
        "climb": "down",
        "climb_milling": "down",
        "2": "down",
        "slot": "slot",
        "full": "slot",
        "full_slot": "slot",
        "0": "slot",
    }
    if text not in aliases:
        raise ValueError(f"unsupported cut_dir: {cut_dir!r}")
    return aliases[text]


def engagement_angles(
    radial_immersion: float = 1.0,
    cut_dir: str | int = "slot",
) -> tuple[float, float, str]:
    """Return entry/exit angles in radians for a radial immersion fraction."""

    if radial_immersion <= 0.0 or radial_immersion > 1.0:
        raise ValueError("radial_immersion must be in (0, 1]")

    normalized = normalize_cut_direction(cut_dir)
    if normalized == "slot" or math.isclose(radial_immersion, 1.0):
        return 0.0, math.pi, normalized

    immersion_angle = math.acos(1.0 - 2.0 * radial_immersion)
    if normalized == "down":
        return 0.0, immersion_angle, normalized
    return math.pi - immersion_angle, math.pi, normalized


def directional_factors(cutting: CuttingParameters) -> DirectionalFactors:
    """Compute zero-order directional factors for the engaged arc."""

    if cutting.entry_angle_rad is None or cutting.exit_angle_rad is None:
        entry, exit_, normalized = engagement_angles(cutting.radial_immersion, cutting.cut_dir)
    else:
        entry = float(cutting.entry_angle_rad)
        exit_ = float(cutting.exit_angle_rad)
        normalized = "explicit"

    if exit_ <= entry:
        raise ValueError("exit angle must be greater than entry angle")

    kr = cutting.radial_to_tangential_ratio

    def primitive(phi: float) -> tuple[float, float, float, float]:
        return (
            0.5 * (math.cos(2.0 * phi) - 2.0 * kr * phi + kr * math.sin(2.0 * phi)),
            0.5 * (-math.sin(2.0 * phi) - 2.0 * phi + kr * math.cos(2.0 * phi)),
            0.5 * (-math.sin(2.0 * phi) + 2.0 * phi + kr * math.cos(2.0 * phi)),
            0.5 * (-math.cos(2.0 * phi) - 2.0 * kr * phi - kr * math.sin(2.0 * phi)),
        )

    start = primitive(entry)
    end = primitive(exit_)
    values = tuple(end_value - start_value for start_value, end_value in zip(start, end))
    return DirectionalFactors(
        alpha_xx=values[0],
        alpha_xy=values[1],
        alpha_yx=values[2],
        alpha_yy=values[3],
        entry_angle_rad=entry,
        exit_angle_rad=exit_,
        cut_dir=normalized,
    )


def inverse_characteristic_eigenvalues(
    chatter_frequency_hz: float,
    modal_x: ModalParameters,
    cutting: CuttingParameters,
    modal_y: ModalParameters | None = None,
    directional: DirectionalFactors | None = None,
) -> np.ndarray:
    """Return eigenvalues of inv(GA), the form used in the lobe formulas."""

    if chatter_frequency_hz <= 0.0:
        raise ValueError("chatter_frequency_hz must be positive")

    omega = TWO_PI * chatter_frequency_hz
    factors = directional if directional is not None else directional_factors(cutting)

    if modal_y is None:
        scalar_factor = factors.alpha_xx
        if math.isclose(scalar_factor, 0.0, abs_tol=1e-14):
            raise ValueError("alpha_xx is near zero; provide modal_y for 2D directional coupling")
        receptance_product = np.array([modal_x.receptance(omega) * scalar_factor], dtype=complex)
    else:
        transfer = np.diag([modal_x.receptance(omega), modal_y.receptance(omega)])
        receptance_product = np.linalg.eigvals(transfer @ factors.as_matrix())

    nonzero = receptance_product[np.abs(receptance_product) > 0.0]
    return 1.0 / nonzero


def critical_depth_m(
    inverse_eigenvalue: complex,
    cutting: CuttingParameters,
) -> float | None:
    """Return limiting axial depth for a stable branch, or None if inactive."""

    real = float(np.real(inverse_eigenvalue))
    imag = float(np.imag(inverse_eigenvalue))
    if real >= 0.0 or math.isclose(real, 0.0, abs_tol=1e-18):
        return None

    kappa = imag / real
    depth = (
        -TWO_PI
        * real
        * (1.0 + kappa**2)
        / (cutting.teeth * cutting.tangential_force_coefficient_n_per_m2)
    )
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    return depth


def phase_shift_rad(inverse_eigenvalue: complex) -> float:
    """Return tooth-period phase lag for the candidate lobe branch."""

    real = float(np.real(inverse_eigenvalue))
    imag = float(np.imag(inverse_eigenvalue))
    if math.isclose(real, 0.0, abs_tol=1e-18):
        raise ValueError("inverse eigenvalue real part is too close to zero")
    kappa = imag / real
    phase = math.pi - 2.0 * math.atan(kappa)
    phase = phase % TWO_PI
    return phase if phase > 0.0 else TWO_PI


def spindle_speed_rpm(
    chatter_frequency_hz: float,
    teeth: int,
    phase_rad: float,
    lobe_order: int,
) -> float:
    """Map chatter frequency and phase to a candidate spindle speed."""

    if chatter_frequency_hz <= 0.0:
        raise ValueError("chatter_frequency_hz must be positive")
    if teeth <= 0:
        raise ValueError("teeth must be positive")
    if lobe_order < 0:
        raise ValueError("lobe_order must be non-negative")

    denominator = teeth * (lobe_order + phase_rad / TWO_PI)
    if denominator <= 0.0:
        raise ValueError("invalid phase/order combination")
    return 60.0 * chatter_frequency_hz / denominator


def frequency_grid_around_modes(
    modal_x: ModalParameters,
    modal_y: ModalParameters | None = None,
    points: int = 401,
    span: float = 0.35,
) -> np.ndarray:
    """Create a deterministic chatter-frequency grid around modal frequencies."""

    if points < 2:
        raise ValueError("points must be at least 2")
    if span <= 0.0:
        raise ValueError("span must be positive")

    frequencies = [modal_x.natural_frequency_hz]
    if modal_y is not None:
        frequencies.append(modal_y.natural_frequency_hz)

    lower = max(1e-9, min(frequencies) * (1.0 - span))
    upper = max(frequencies) * (1.0 + span)
    return np.linspace(lower, upper, points)


def candidate_stability_lobes(
    modal_x: ModalParameters,
    cutting: CuttingParameters,
    modal_y: ModalParameters | None = None,
    chatter_frequencies_hz: Iterable[float] | None = None,
    lobe_orders: Iterable[int] = range(0, 8),
) -> list[StabilityPoint]:
    """Generate candidate spindle-speed/blim points for zero-order lobes."""

    frequencies = (
        frequency_grid_around_modes(modal_x, modal_y)
        if chatter_frequencies_hz is None
        else np.asarray(list(chatter_frequencies_hz), dtype=float)
    )
    orders = [int(order) for order in lobe_orders]
    if any(order < 0 for order in orders):
        raise ValueError("lobe_orders must be non-negative")

    factors = directional_factors(cutting)
    points: list[StabilityPoint] = []
    for chatter_frequency_hz in frequencies:
        inverse_values = inverse_characteristic_eigenvalues(
            float(chatter_frequency_hz),
            modal_x,
            cutting,
            modal_y=modal_y,
            directional=factors,
        )
        for eigenvalue_index, inverse_value in enumerate(inverse_values):
            depth = critical_depth_m(inverse_value, cutting)
            if depth is None:
                continue
            phase = phase_shift_rad(inverse_value)
            real = float(np.real(inverse_value))
            imag = float(np.imag(inverse_value))
            kappa = imag / real
            for order in orders:
                rpm = spindle_speed_rpm(float(chatter_frequency_hz), cutting.teeth, phase, order)
                points.append(
                    StabilityPoint(
                        spindle_speed_rpm=rpm,
                        axial_depth_m=depth,
                        chatter_frequency_hz=float(chatter_frequency_hz),
                        lobe_order=order,
                        eigenvalue_index=eigenvalue_index,
                        phase_rad=phase,
                        kappa=kappa,
                        inverse_eigenvalue_real=real,
                        inverse_eigenvalue_imag=imag,
                    )
                )

    points.sort(key=lambda point: (point.lobe_order, point.spindle_speed_rpm, point.axial_depth_m))
    return points


def nearest_point_for_speed(
    points: Iterable[StabilityPoint],
    spindle_speed_rpm_target: float,
) -> StabilityPoint | None:
    """Return the candidate point closest to a requested spindle speed."""

    if spindle_speed_rpm_target <= 0.0:
        raise ValueError("spindle_speed_rpm_target must be positive")

    best_point: StabilityPoint | None = None
    best_error = math.inf
    for point in points:
        error = abs(point.spindle_speed_rpm - spindle_speed_rpm_target) / spindle_speed_rpm_target
        if error < best_error:
            best_error = error
            best_point = point
    return best_point


def points_to_array(points: Iterable[StabilityPoint]) -> np.ndarray:
    """Convert lobe points to a numeric array for downstream plotting or checks."""

    rows = [
        (
            point.spindle_speed_rpm,
            point.axial_depth_m,
            point.chatter_frequency_hz,
            point.lobe_order,
            point.eigenvalue_index,
        )
        for point in points
    ]
    return np.asarray(rows, dtype=float)
