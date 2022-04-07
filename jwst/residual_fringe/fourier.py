import numpy as np

from astropy.modeling.core import Fittable1DModel
from astropy.modeling.models import Cosine1D, Sine1D
from astropy.modeling.parameters import Parameter


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
            names.append(f"a_{index}")
            names.append(f"b_{index}")
            names.append(f"omega_{index}")

        return tuple(names)

    def _get_parameters(self, *params):
        parameters = list(params)

        term_parameters = []
        for _ in range(self.n_terms):
            a = parameters.pop(0)
            b = parameters.pop(0)
            f = parameters.pop(0)

            term_parameters.append((a, b, f))

        return term_parameters

    @staticmethod
    def _terms(a_amplitude, b_amplitude, frequency):
        return Sine1D(a_amplitude, frequency), Cosine1D(b_amplitude, frequency)

    def _evaluate_term(self, x, a_amplitude, b_amplitude, frequency):
        sin, cos = self._terms(a_amplitude, b_amplitude, frequency)

        return sin(x) + cos(x)

    def evaluate(self, x, *params):
        parameters = self._get_parameters(*params)

        value = 0
        for param in parameters:
            value += self._evaluate_term(x, *param)

        return value

    def _fit_deriv_term(self, x, a_amplitude, b_amplitude, frequency):
        sin, cos = self._terms(a_amplitude, b_amplitude, frequency)

        d_sin = sin.fit_deriv(x, a_amplitude, frequency, 0)
        d_cos = cos.fit_deriv(x, b_amplitude, frequency, 0)

        d_a_amplitude = d_sin[0]
        d_b_amplitude = d_cos[0]
        d_frequency = d_sin[1] + d_cos[1]

        return [d_a_amplitude, d_b_amplitude, d_frequency]

    def fit_deriv(self, x, *params):
        parameters = self._get_parameters(*params)

        deriv = []
        for param in parameters:
            deriv.extend(self._fit_deriv_term(x, *param))

        return deriv

    def jacobian(self, x, *params):
        return self.fit_deriv(x, *params)

    def fix_frequency(self, value, term=0):
        name = self.param_names[3 * term + 2]
        parameter = getattr(self, name)

        parameter.value = value
        parameter.fixed = True

    def unfix_frequency(self, term=0):
        name = self.param_names[3 * term + 2]
        parameter = getattr(self, name)

        if parameter.fixed:
            parameter.fixed = False
        else:
            raise ValueError(f"Term: {term} frequency is not fixed!")
