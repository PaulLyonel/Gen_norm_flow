# Generalized Normalizing Flow

This repository belongs to the tutorial paper [1] "Generalized Normalizing Flows" (https://arxiv.org/abs/2111.12506) 
and reproduces the mixture and scatterometry example. 
The code is based on the repo of the paper [2]
"Stochastic Normalizing Flows for Inverse Problems: a Markov Chains viewpoint" 
(available at https://github.com/PaulLyonel/conditionalSNF).

This repository implements normalizing flows (via the FrEIA repository https://github.com/VLL-HD/FrEIA), 
MCMC methods (MALA, Gaussian proposal), 
Langevin methods and VAE layers. We optimize
them via a unified loss function, which acts on the paths of a forward and backward process.

The forward model for the scatterometry example (Section 7.2 in [1] and script Main_scatterometry.py) 
was kindly provided by Sebastian Heidenreich from PTB (Physikalisch-Technische Bundesanstalt). 
For more information on this particular inverse problem, we refer the reader to the papers [3] and [4]. 
Furthermore, if you use the forward model of the scatterometry inverse problem, please cite the papers [3] and [4].

If you have any questions or bug reports, feel free to contact Paul Hagemann (hagemann(at)math.tu-berlin.de) or Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

[1] P. Hagemann, J. Hertrich and G. Steidl.  
Generalized Normalizing Flows via Markov Chains.  
Elements in Non-local Data Interactions: Foundations and Applications.  
Cambridge University Press, 2023.

[2] P. Hagemann, J. Hertrich and G. Steidl.  
Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint.  
SIAM/ASA Journal on Uncertainty Quantification, vol. 10 (3), pp. 1162-1190, 2022.

[3] S. Heidenreich, H. Gross, and M. Bär.  
Bayesian approach to the statistical inverse problem of scatterometry: Comparison of three surrogate models.  
International Journal for Uncertainty Quantification, 5(6), 2015.

[4] S. Heidenreich, H. Gross, and M. Bär.  
Bayesian approach to determine critical dimensions from scatterometric measurements.  
Metrologia, 55(6):S201, Dec. 2018.
