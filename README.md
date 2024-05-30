# quadrat
Sounds and moving pictures from quadratic difference equations.

In 1993, J. C. Sprott published [Automatic Generation of Strange Attractors](https://sprott.physics.wisc.edu/pubs/PAPER203.HTM), where chaotic maps are produced by iterating a reccurrence relation given by a pair of coupled quadratic difference equations:

$$
x_{n+1} = a_{1} + a_{2}x_{n} + a_{3}x_{n}^{2} + a_{4}x_{n}y_{n} + a_{5}y_{n} + a_{6}y_{n}^{2}
$$
$$
y_{n+1} = a_{7} + a_{8}x_{n} + a_{9}x_{n}^{2} + a_{10}x_{n}y_{n} + a_{11}y_{n} + a_{12}y_{n}^{2}
$$

However, one would prefer to consider this as the vectrorized operation:

$$
\Large{P_{n+1} = B + CV_{n}}
$$
$$
\Large{P_{n+1} = \begin{bmatrix} x_{n}+1 \\ y_{n}+1 \end{bmatrix}, B =  \begin{bmatrix} a_{1} \\ a_{7} \end{bmatrix}}, C =  \left[ \begin{array}{ccccccc} a_{2} & a_{3} & a_{4} & a_{5} & a_{6} \\
a_{8} & a_{9} & a_{10} & a_{11} & a_{12} \end{array} \right]
$$
$$
\Large{V_{n} = \begin{bmatrix} x_{n} \\ x_{n}^{2} \\ x_{n}y_{n} \\ y_{n} \\ y_{n}^{2} \\  \end{bmatrix}}
$$

In this way, attractors for many sets of coefficients can be iterated at once, using multi-dimensional numpy arrays
or torch tensors. If a secquence of points doesn't diverge, it is considered aesthetic if its
Lyapunov exponent and Correlation dimension lie within a certain range. Random searches of coefficients are performed,
where values from -1.2 to 1.2 in increments of 0.1 are assigned a letter from `A` to `Y`. The point sequences are
made into images by taking a histogram, so that points visited more frequently are brighter. Thus, a `SDUUCQUXQXDK` is:

![SDUUCQUXQXDK](img/SDUUCQUXQXDK.png)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/augeas/quadrat/main?urlpath=https%3A%2F%2Fgithub.com%2Faugeas%2Fquadrat%2Fblob%2Fmain%2Fvideos.ipynb)
