import math


class HamiltonianValidation:
    """Validate a Counter-Diabatic Hamiltonian and derive LCU metadata."""

    def __init__(self, hamiltonian):

        self.hamiltonian = hamiltonian

        #
        # Metadata
        #
        self.name = hamiltonian.get(
            "hamiltonian_type",
            "counterdiabatic",
        )

        #
        # Hamiltonian structure
        #
        self.selected_features = list(
            hamiltonian["selected_features"]
        )

        self.pauli_terms = hamiltonian["pauli_terms"]

        #
        # Coefficients
        #
        self.coefficients = []

        for term in self.pauli_terms:

            coefficient = term["coefficient"]

            if isinstance(coefficient, complex):

                assert abs(coefficient.imag) < 1e-12, (
                    "Hamiltonian contains non-zero imaginary coefficients."
                )

                coefficient = coefficient.real

            self.coefficients.append(float(coefficient))

        #
        # Pauli strings
        #
        self.pauli_strings = [
            term["pauli_string"]
            for term in self.pauli_terms
        ]

        #
        # Derived quantities
        #
        self.num_terms = len(self.pauli_terms)

        self.num_system_qubits = len(
            self.selected_features
        )

        self.num_index_qubits = (
            math.ceil(math.log2(self.num_terms))
            if self.num_terms > 1 else 1
        )

        self.lambda_value = sum(
            abs(c) for c in self.coefficients
        )

        if self.lambda_value > 0:

            self.weights = [
                abs(c) / self.lambda_value
                for c in self.coefficients
            ]

        else:

            self.weights = [0.0] * self.num_terms

    def validate(self):

        assert self.num_terms > 0, (
            "Hamiltonian has no terms."
        )

        assert len(self.coefficients) == self.num_terms, (
            "Coefficient count mismatch."
        )

        assert len(self.pauli_strings) == self.num_terms, (
            "Pauli-string count mismatch."
        )

        assert len(self.selected_features) == self.num_system_qubits, (
            "Selected-feature count mismatch."
        )

        assert (
            2 ** self.num_index_qubits
            >= self.num_terms
        ), (
            "Index register too small."
        )

        assert (
            abs(sum(self.weights) - 1.0) < 1e-10
            or self.lambda_value == 0
        ), (
            "LCU weights do not sum to 1."
        )

        return True

    def report(self):

        return {
            "name": self.name,
            "num_terms": self.num_terms,
            "num_system_qubits": self.num_system_qubits,
            "num_index_qubits": self.num_index_qubits,
            "lambda_value": self.lambda_value,
            "weights_sum": sum(self.weights),
            "weights": self.weights,
        }