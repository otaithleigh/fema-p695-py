from __future__ import annotations

import importlib.resources
import warnings

import asce7_16
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import fsolve

__version__ = importlib.resources.read_text(__name__, '__version__')

__all__ = [
    'acmrxx',
    'beta_total',
    'design_response_spectrum',
    'fundamental_period',
    'mapped_value',
    'seismic_response_coeff',
    'sf1',
    'smt',
    'ssf',
]


#------------------------------------------------------------------------
# Collapse margin ratio
#------------------------------------------------------------------------
def acmrxx(beta_total, collapse_prob, xin=0.622):
    """Compute the acceptable value of the adjusted collapse margin ratio (ACMR).

    Parameters
    ----------
    beta_total:
        Total uncertainty present in the system
    collapse_prob:
        Collapse probability being checked (e.g. 0.20 for ACMR20)
    xin = 0.622:
        Starting value for the nonlinear solution. Tweak this if there are
        convergence issues.

    Ref: FEMA P695 Section 7.4
    """

    # Solve lognorm.cdf as a substitute for MATLAB's logninv function
    def f(x):
        return lognorm.cdf(x, beta_total) - collapse_prob

    X = fsolve(f, xin)

    return 1 / X[0]


# Uncertainty values for each rating
_rating_values = {
    'A': 0.10,
    'B': 0.20,
    'C': 0.35,
    'D': 0.50,
}


def beta_total(rating_DR: str,
               rating_TD: str,
               rating_MDL: str,
               mu_T: float = 3.0) -> float:
    """Compute the total uncertainty present in th system.

    Parameters
    ----------
    rating_DR:
        Rating of the design requirements for the system
    rating_TD:
        Rating of the test data for the system
    rating_MDL:
        Rating of the model's representation of the system
    mu_T:
        Period-based ductility

    Ref: FEMA P695 Section 7.3.1
    """
    beta_DR = _rating_values[rating_DR.upper()]
    beta_TD = _rating_values[rating_TD.upper()]
    beta_MDL = _rating_values[rating_MDL.upper()]
    beta_RTR = min((0.1 + 0.1 * mu_T, 0.4))
    beta = np.sqrt(beta_RTR**2 + beta_DR**2 + beta_TD**2 + beta_MDL**2)

    return round(beta * 40) / 40


#------------------------------------------------------------------------
# Mapped hazards
#------------------------------------------------------------------------
_mapped_value_dict = {
    "dmax": {
        "ss": 1.5,
        # Actually 0.60 but "should be taken as less than 0.60" *eyeroll*
        "s1": 0.59999999999,
        "fa": 1.0,
        "fv": 1.50,
        "sms": 1.50,
        "sm1": 0.90,
        "sds": 1.0,
        "sd1": 0.60,
        "ts": 0.60,
    },
    "dmin": {
        "ss": 0.55,
        "s1": 0.132,
        "fa": 1.36,
        "fv": 2.28,
        "sms": 0.75,
        "sm1": 0.30,
        "sds": 0.50,
        "sd1": 0.20,
        "ts": 0.40,
    },
    "cmin": {
        "ss": 0.33,
        "s1": 0.083,
        "fa": 1.53,
        "fv": 2.4,
        "sms": 0.50,
        "sm1": 0.20,
        "sds": 0.33,
        "sd1": 0.133,
        "ts": 0.40,
    },
    "bmin": {
        "ss": 0.156,
        "s1": 0.042,
        "fa": 1.6,
        "fv": 2.4,
        "sms": 0.25,
        "sm1": 0.10,
        "sds": 0.167,
        "sd1": 0.067,
        "ts": 0.40,
    }
}
_mapped_value_dict["cmax"] = _mapped_value_dict["dmin"]
_mapped_value_dict["bmax"] = _mapped_value_dict["cmin"]


def mapped_value(value: str, sdc: str):
    """Retrieve the mapped seismic parameter for a given design category.

    Parameters
    ----------
    value : {'SS', 'S1', 'Fa', 'Fv', 'SMS', 'SM1', 'SDS', 'SD1', 'Ts'}
        Mapped parameter to retrieve 
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category
    """
    return _mapped_value_dict[sdc.lower()][value.lower()]


#------------------------------------------------------------------------
# Ground motion scaling
#------------------------------------------------------------------------
# Table A-3
_T_INTERP = np.array([
    0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6,
    1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
])
_SNRT_INTERP = {
    'farfield':
    np.array([
        0.779, 0.775, 0.761, 0.748, 0.749, 0.736, 0.602, 0.537, 0.449, 0.399,
        0.348, 0.301, 0.256, 0.208, 0.168, 0.148, 0.133, 0.118, 0.106, 0.091,
        0.080, 0.063, 0.052, 0.046, 0.041
    ]),
    'nearfield':
    np.array([
        0.936, 1.020, 0.939, 0.901, 0.886, 0.855, 0.833, 0.805, 0.739, 0.633,
        0.571, 0.476, 0.404, 0.356, 0.319, 0.284, 0.258, 0.230, 0.210, 0.190,
        0.172, 0.132, 0.104, 0.086, 0.072
    ])
}


def snrt(T, record_set: str = 'farfield'):
    """Retrieve the median 5%-damped spectral pseudo-acceleration of the
    normalized record set, SNRT, at a given period T.

    Parameters
    ----------
    T : float
        Period of the structure [seconds]
    record_set : {'farfield', 'nearfield'}
        Ground motion set (default: 'farfield')
    """
    if T <= _T_INTERP[0] or T >= _T_INTERP[-1]:
        raise ValueError(f"Period is out of range: T = {T}")

    try:
        snrt_interp = _SNRT_INTERP[record_set]
    except KeyError as exc:
        valid_sets = set(_SNRT_INTERP.keys())
        raise ValueError(f'Unrecognized record set {record_set!r}; '
                         f'must be one of {valid_sets!r}') from exc
    
    return np.interp(T, _T_INTERP, snrt_interp)


def sf1(T, sdc, record_set: str = 'farfield'):
    """Calculate scale factor 1, which scales normalized ground motions to the
    MCE spectral demand.

    FEMA P695 calls this the "scaling factor for anchoring record set to MCE
    spectral demand" and doesn't provide a symbol for it; here we use SF1.

    Parameters
    ----------
    T : float
        Period of the structure [seconds]
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category
    record_set : {'farfield', 'nearfield'}
        Ground motion set (default: 'farfield')

    Ref: FEMA P695 Section A.8, Paragraph "Scaling of Record Sets"
    """
    SNRT = snrt(T, record_set)
    SMT = smt(T, sdc)

    return SMT / SNRT


def smt(T, sdc):
    """Calculate the MCE demand, SMT.

    Parameters
    ----------
    T : float
        Fundamental period of the structure [seconds]
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category
    """
    SM1 = mapped_value("SM1", sdc)
    SMS = mapped_value("SMS", sdc)
    if T <= SM1 / SMS:
        return SMS
    else:
        return SM1 / T


#------------------------------------------------------------------------
# Spectral shape factor
#------------------------------------------------------------------------
# Figure B-3 | Equation B-1
def _ε_records_farfield(T) -> np.ndarray:
    T = np.asarray(T)
    return np.piecewise(
        T,
        [
            T < 0.5,
            (0.5 <= T) & (T < 1.5),
        ],
        [
            0.6,
            lambda T: 0.6 * (1.5 - T),
            0.0,
        ],
    )


# Figure B-9 | Equation B-6
def _ε_records_nearfield(T) -> np.ndarray:
    T = np.asarray(T)
    return np.piecewise(
        T,
        [
            T < 1.5,
            (1.5 <= T) & (T < 2.5),
        ],
        [
            0.0,
            lambda T: 0.2 * (T - 1.5),
            0.2,
        ],
    )


_ε_records_dispatch = {
    'farfield': _ε_records_farfield,
    'nearfield': _ε_records_nearfield,
    'nearfield_pulse': _ε_records_nearfield,
    'nearfield_no_pulse': _ε_records_nearfield,
}


def _get_ε_records(record_set):
    try:
        ε_records = _ε_records_dispatch[record_set]
    except KeyError as exc:
        record_sets = set(_ε_records_dispatch.keys())
        raise ValueError(f'record_set {record_set!r} '
                         f'not in {record_sets!r}') from exc

    return ε_records


def ssf(T, μT, sdc, record_set: str = 'farfield'):
    """Compute the spectral shape factor (SSF).

    Parameters
    ----------
    T : float, array_like
        Fundamental period of the structure [seconds].
    μT : float, array_like
        Period-based ductility [unitless]. See Equation 6-6.
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category (case-insensitive)
    record_set : {'farfield', 'nearfield'}
        Ground motion set (default: 'farfield')

    Ref: FEMA P695 Appendix B
    """
    μT = np.asarray(μT)
    if np.any(μT < 1.0):
        raise ValueError('μT must be >= 1.0')

    # Equation B-3
    β1: np.ndarray = np.piecewise(
        μT,
        [μT <= 8.0],
        [
            lambda μT: 0.14 * (μT - 1)**0.42,
            0.32,
        ],
    )

    # Section B.3.2
    εo = {
        'bmin': 1.0,
        'bmax': 1.0,
        'cmin': 1.0,
        'cmax': 1.0,
        'dmin': 1.5,
        'dmax': 1.5,
    }[sdc.lower()]

    ε_records = _get_ε_records(record_set)

    # Equation B-4
    return np.exp(β1 * (εo - ε_records(T)))


#------------------------------------------------------------------------
# Structure parameters
#------------------------------------------------------------------------
def fundamental_period(hn, Ct, x, sdc):
    """Calculate the fundamental period of the structure.

    Uses the following equation:

    .. math:: T = \\max(C_u C_t h_n^x, 0.25)

    Where :math:`C_u` is the coefficient on the upper limit of the calculated
    period from ASCE 7.

    Parameters
    ----------
    hn : float
        Height of the structure (ft)
    Ct : float
        Building period coefficient.
    x : float
        Exponent on building height.
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category.

    Returns
    -------
    T : float
        Fundamental period in seconds.
    """
    SD1 = mapped_value('SD1', sdc)
    Ta = asce7_16.seismic.approximate_period(hn, Ct, x)
    Cu = asce7_16.seismic.period_upper_limit_coeff(SD1)

    return np.maximum(Cu * Ta, 0.25)


def seismic_response_coeff(R, T, sdc, level: str = 'design'):
    """Calculate the seismic response coefficient, C_s.

    Parameters
    ----------
    R : float
        Response modification factor.
    T : float
        Fundamental period (s).
    sdc : {'dmax', 'dmin', 'cmax', 'cmin', 'bmax', 'bmin'}
        Seismic design category (Dmax, Dmin, etc.).
    level : {'design', 'mce'}, optional
        Hazard level to get the response coefficient for. (default: 'design')

    Note that this function follows the assumptions and restrictions enforced by
    FEMA P695; namely, it is used only with mapped hazard values from the
    ``mapped_values`` function, and for structures with periods of 4.0 s or
    lower. For a more general function, see ``asce7_16.seismic_response_coeff``.
    """
    if T > 4.0:
        warnings.warn('seismic_response_coeff: Given period '
                      f'(T = {T} s) is out of FEMA P695 range')

    Ts = mapped_value('Ts', sdc)

    _level = level.casefold()
    if _level == 'design':
        S1 = mapped_value('SD1', sdc)
        SS = mapped_value('SDS', sdc)
    elif _level in ['mce', 'maximum']:
        S1 = mapped_value('SM1', sdc)
        SS = mapped_value('SMS', sdc)
    else:
        raise ValueError(f"level {level!r} must be one of {{'design', 'mce'}}")

    if T <= Ts:
        Cs = SS / R
    else:
        Cs = max(S1 / (T * R), 0.044 * SS)

    Cs = max(Cs, 0.01)

    return Cs


def design_response_spectrum(sdc, tl=4.0):
    """Generate the design response spectrum for the given seismic design
    category.

    Parameters
    ----------
    sdc : {'Dmax', 'Dmin', 'Cmax', 'Cmin', 'Bmax', 'Bmin'}
        Seismic design category.
    tl : float, optional
        Transition period between constant velocity and constant displacement
        response domains. Note that FEMA P695 does not include values for T_L
        since it restricts archetypes to having fundamental periods of 4 sec or
        less, and thus T_L should not apply to most FEMA P695 studies.
        (default: 4.0)
    """
    sds = mapped_value('SDS', sdc)
    sd1 = mapped_value('SD1', sdc)
    return asce7_16.seismic.design_response_spectrum(sds, sd1, tl)


def mce_response_spectrum(sdc, tl=4.0):
    """Generate the MCE-level response spectrum for the given seismic design
    category.

    Parameters
    ----------
    sdc : {'Dmax', 'Dmin', 'Cmax', 'Cmin', 'Bmax', 'Bmin'}
        Seismic design category.
    tl : float, optional
        Transition period between constant velocity and constant displacement
        response domains. Note that FEMA P695 does not include values for T_L
        since it restricts archetypes to having fundamental periods of 4 sec or
        less, and thus T_L should not apply to most FEMA P695 studies.
        (default: 4.0)
    """
    sms = mapped_value('SMS', sdc)
    sm1 = mapped_value('SM1', sdc)
    return asce7_16.seismic.design_response_spectrum(sms, sm1, tl)
