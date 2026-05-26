def num_nvar(den_type: str) -> int:
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
    mapping = {
        "RHO": 0,
        "SIGMA": 1,
        "TAU": 1,
        "LAPL": 2,
    }
    if den_type not in mapping:
        raise ValueError(f"Unsupported density type: {den_type}")
    return mapping[den_type]
