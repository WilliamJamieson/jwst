import numpy as np

from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.units import Quantity


TWOPI = 2 * np.pi


class FourierSeries1D(Fittable1DModel):
    _param_names = ()

    def __init__(self, n_terms,
                 n_models=None, model_set_axis=None, name=None, meta=None):
        self._n_terms = n_terms

        self._param_names = self._generate_param_names()

        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(param_name, default=np.zeros(()))

        super().__init__(
            n_models=n_models, model_set_axis=model_set_axis, name=name,
            meta=meta)

    @property
    def param_names(self):
        return self._param_names

    @property
    def n_terms(self):
        return self._n_terms

    def _generate_param_names(self):
        names = []

        for index in range(self.n_terms):
            names.append(f"omega_{index}")
            names.append(f"a_{index}")
            names.append(f"b_{index}")

        return tuple(names)

    def _get_parameters(self, *params):
        parameters = list(params)

        term_parameters = []
        for _ in range(self.n_terms):
            f = parameters.pop(0)
            a = parameters.pop(0)
            b = parameters.pop(0)

            term_parameters.append((f, a, b))

        return term_parameters

    @staticmethod
    def _terms(x, freq):
        argument = TWOPI * (freq * x)
        if isinstance(argument, Quantity):
            argument = argument.value

        return np.sin(argument), np.cos(argument)

    def _evaluate_term(self, x, freq, coeff_a, coeff_b):

        sin, cos = self._terms(x, freq)

        return coeff_a * sin + coeff_b * cos

    def evaluate(self, x, *params):
        term_parameters = self._get_parameters(*params)

        value = 0
        for index in range(self.n_terms):

            value += self._evaluate_term(x, *term_parameters[index])

        return value

    def fit_deriv(self, x, *params):
        term_parameters = self._get_parameters(*params)

        deriv = []
        for index in range(self.n_terms):
            deriv.extend(list(self._terms(x, term_parameters[index][0])))

        return deriv

    def _jacobian_term(self, x, freq, coeff_a, coeff_b):
        sin, cos = self._terms(x, freq)

        coeff = TWOPI * freq
        if isinstance(coeff, Quantity):
            coeff = coeff.value

        d_freq = coeff * (coeff_a * cos - coeff_b * sin)

        return [sin, cos, d_freq]

    def jacobian(self, x, *params):
        term_parameters = self._get_parameters(*params)

        deriv = []
        for index in range(self.n_terms):
            deriv.extend(self._jacobian_term(x, *term_parameters[index]))

        return deriv

    def fix_frequency(self, value, term=0):
        name = self.param_names[term * 3]
        parameter = getattr(self, name)

        parameter.value = value
        parameter.fixed = True

    def unfix_frequency(self, term=0):
        name = self.param_names[term * 3]
        parameter = getattr(self, name)

        if parameter.fixed:
            parameter.fixed = False
        else:
            raise ValueError(f"Term: {term} frequency is not fixed!")
