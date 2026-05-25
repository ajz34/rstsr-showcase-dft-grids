# Parameter

- `xc_func`: The LibXC functional to evaluate.

    The user should make sure to initialize the LibXC functional with **proper spin-polarization**, and possibly **proper omega/parameter settings**.

- `rho`: The density tensor. The shape must be

    - Spin Unpolarized: `[ngrids, nvar]`
    - Spin Polarized: `[ngrids, nvar, 2]`

    Where `nvar` is the number of variables:

    - RHO: 1 ($\rho$)
    - SIGMA: 4 ($\rho$, $\rho_x$, $\rho_y$, $\rho_z$)
    - TAU: 5 ($\rho$, $\rho_x$, $\rho_y$, $\rho_z$, $\tau$)

    Note we slightly differ to the PySCF's convention. RHO's dimension `nvar=1` cannot be squeezed out. This is to make the code more consistent and easier to implement.

- `deriv`: The maximum derivative order to evaluate. Note the smaller derivates will also be evaluated and returned.

- `par`: How to parallelize the evaluation.

    - true/false: Whether to parallelize or not. If true, it will use the default parallelization strategy.
    - usize: The chunk size to be parallelized. If None, it will use the default chunk size.
  
    The default chunksize depends on density type and spin:

    - RHO, Unpolarized: 16384
    - RHO, Polarized: 6144
    - SIGMA: 384
    - TAU: 256

# Input/Output Shapes

Please note the output will contain all derivatives up to `deriv` user specified. For example, if `deriv = 1` for GGA (SIGMA) and unpolarized, the output will contain two tensors, first of shape `[ngrids]`, second of shape `[ngrids, 3]`. 

| deriv | Output `xc_eff`<br>Unpolarized | Output `xc_eff`<br>Polarized |
|--|--|--|
| 0 | `[ngrids]` | `[ngrids]` |
| 1 | `[ngrids, nvar]` | `[ngrids, nvar, 2]` |
| 2 | `[ngrids, nvar, nvar]` | `[ngrids, nvar, 2, nvar, 2]` |
| 3 | `[ngrids, nvar, nvar, nvar]` | `[ngrids, nvar, 2, nvar, 2, nvar, 2]` |
