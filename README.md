# j_johnson_M
Use moments to estimate parameters of a Johnson distribution.

## Usage

```python
from j_johnson_M import j_johnson_M

coef, _type, err = j_johnson_M(mu, sd, skew, kurt)

gamma, delta, xi, lamda = coef
```

## References

    Ported from original MATLAB ToolBox "Johnson Curve Toolbox"
    Dave (2021). Johnson Curve Toolbox 
    (https://www.mathworks.com/matlabcentral/fileexchange/46123-johnson-curve-toolbox), 
    MATLAB Central File Exchange. Retrieved April 29, 2021.

    ######################################################################
    # Coded in Python by MAX PIERINI © 2021 EpiData.it (info@epidata.it) #
    ######################################################################
