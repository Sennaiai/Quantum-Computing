import numpy as np

from pyqsp.poly import PolyCosineTX, PolySineTX
from pyqsp.angle_sequence import QuantumSignalProcessingPhases


class QSVTHelper:
    """
    Generate Low-Chuang QSVT phase sequences for Hamiltonian simulation.

    Wraps the validated PennyLane / pyQSP synthesis pipeline

        lambda
            ↓
        tau = lambda * physical_time
            ↓
        Chebyshev polynomial synthesis
            ↓
        Wx QSP phase synthesis
            ↓
        Low-Chuang phase conversion

    The resulting phase sequences can be passed directly to
    QSPGenerator.
    """

    def __init__(
        self,
        validator,
        physical_time=1.0,
        epsilon=1e-6,
    ):
        self.validator = validator

        self.lambda_value = float(
            validator.lambda_value
        )

        self.physical_time = float(
            physical_time
        )

        self.epsilon = float(
            epsilon
        )

        self.tau = (
            self.lambda_value
            * self.physical_time
        )

        self.validate()

        self._generate_polynomials()

        self._generate_phase_sequences()

    def validate(self):

        if self.lambda_value <= 0:
            raise ValueError(
                "Lambda must be positive."
            )

        if self.physical_time <= 0:
            raise ValueError(
                "Physical time must be positive."
            )

        if self.epsilon <= 0:
            raise ValueError(
                "Epsilon must be positive."
            )

        if self.epsilon >= 1:
            raise ValueError(
                "Epsilon must be smaller than 1."
            )

    def _generate_polynomials(self):

        cos_generator = PolyCosineTX()

        sin_generator = PolySineTX()

        (
            self.cos_coeffs,
            self.cos_scale,
        ) = cos_generator.generate(
            tau=self.tau,
            epsilon=self.epsilon,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=True,
        )

        (
            self.sin_coeffs,
            self.sin_scale,
        ) = sin_generator.generate(
            tau=self.tau,
            epsilon=self.epsilon,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=True,
        )

        self.cos_coeffs = np.asarray(
            self.cos_coeffs,
            dtype=float,
        )

        self.sin_coeffs = np.asarray(
            self.sin_coeffs,
            dtype=float,
        )

    def _generate_phase_sequences(self):

        self.cos_wx_phases = np.asarray(
            QuantumSignalProcessingPhases(
                self.cos_coeffs,
                signal_operator="Wx",
                measurement="x",
                method="laurent",
                eps=self.epsilon,
                suc=0.99,
            ),
            dtype=float,
        )

        self.sin_wx_phases = np.asarray(
            QuantumSignalProcessingPhases(
                self.sin_coeffs,
                signal_operator="Wx",
                measurement="x",
                method="laurent",
                eps=self.epsilon,
                suc=0.99,
            ),
            dtype=float,
        )

        self.cos_lc_phases = self.wx_to_low_chuang(
            self.cos_wx_phases
        )

        self.sin_lc_phases = self.wx_to_low_chuang(
            self.sin_wx_phases
        )

    @staticmethod
    def wx_to_low_chuang(
        wx_phases,
    ):
        """
        Convert pyQSP Wx-convention phases into
        Low-Chuang reflection-convention phases.

        Reference
        ---------
        PRX Quantum 2, 040203 (2021),
        Appendix A.2.
        """

        lc = np.asarray(
            wx_phases,
            dtype=float,
        ).copy()

        d = len(lc) - 1

        lc[0] += (
            (2 * d - 1)
            * np.pi
            / 4.0
        )

        if d > 1:
            lc[1:d] -= np.pi / 2.0

        lc[d] -= np.pi / 4.0

        return lc

    def cosine_phases(self):
        """
        Return the Low-Chuang cosine
        Hamiltonian-simulation phase sequence.
        """

        return self.cos_lc_phases.copy()

    def sine_phases(self):
        """
        Return the Low-Chuang sine
        Hamiltonian-simulation phase sequence.
        """

        return self.sin_lc_phases.copy()

    @property
    def cosine_degree(self):
        return len(self.cos_coeffs) - 1

    @property
    def sine_degree(self):
        return len(self.sin_coeffs) - 1

    def report(self):

        print("===== QSVT Helper =====")
        print(f"Lambda                  : {self.lambda_value:.6f}")
        print(f"Physical time           : {self.physical_time}")
        print(f"Tau = λt                : {self.tau:.6f}")
        print(f"Epsilon                 : {self.epsilon}")

        print()

        print(f"Cosine polynomial degree: {self.cosine_degree}")
        print(f"Sine polynomial degree  : {self.sine_degree}")

        print()

        print(f"Cosine scale            : {self.cos_scale}")
        print(f"Sine scale              : {self.sin_scale}")

        print()

        print(f"Cosine Wx phases        : {len(self.cos_wx_phases)}")
        print(f"Sine Wx phases          : {len(self.sin_wx_phases)}")

        print()

        print(f"Cosine LC phases        : {len(self.cos_lc_phases)}")
        print(f"Sine LC phases          : {len(self.sin_lc_phases)}")