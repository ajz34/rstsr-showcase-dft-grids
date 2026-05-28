def num_nvar(den_type: str) -> int:
    """Number of density variables for a given density type.

    Parameters
    ----------
    den_type : str
        The density type. One of "RHO", "SIGMA", "TAU", "LAPL".

    Returns
    -------
    int
        The number of density variables.
    """
    mapping = {
        "RHO": 1,
        "SIGMA": 4,
        "TAU": 5,
        "LAPL": 6,
    }
    if den_type not in mapping:
        raise ValueError(f"Unsupported density type: {den_type}")
    return mapping[den_type]


def num_ao_comp(den_type: str) -> int:
    """Number of AO components for a given density type.

    Parameters
    ----------
    den_type : str
        The density type. One of "RHO", "SIGMA", "TAU", "LAPL".

    Returns
    -------
    int
        The number of AO components.
    """
    mapping = {
        "RHO": 1,
        "SIGMA": 4,
        "TAU": 4,
        "LAPL": 10,
    }
    if den_type not in mapping:
        raise ValueError(f"Unsupported density type: {den_type}")
    return mapping[den_type]


def num_ao_deriv(den_type: str) -> int:
    """AO derivative order for a given density type.

    Parameters
    ----------
    den_type : str
        The density type. One of "RHO", "SIGMA", "TAU", "LAPL".

    Returns
    -------
    int
        The AO derivative order.
    """
    mapping = {
        "RHO": 0,
        "SIGMA": 1,
        "TAU": 1,
        "LAPL": 2,
    }
    if den_type not in mapping:
        raise ValueError(f"Unsupported density type: {den_type}")
    return mapping[den_type]
