# ML_Peierls_dynamics
This repository includes codes, trained model samples and data samples to successfully run the machine learning force field model. This contains the simulation code and the training code to generate the results in the paper https://arxiv.org/abs/2510.20659. Sub-directories in this repo are:

1. Training_data_generation:
   Phase ordering dynamics code under thermal quench. Microscopic Hamiltonian is modified SSH model under semi-classical approximation. More details can be found in the paper.
   
1. Training_data_sample :
   training data samples, from 50x50 real lattice exact-diagonalization simulations.

2. Descriptor_sample :
   We have adopted the descriptor to preserve the lattice symmetry. Ex

3. Training code : 
   This is the training code using the descriptor.

4. Data_example :
   This folder includes the simulation result that is used to plot Fig. 3, Fig. 4, Fig. 6

If you have more questions, please contact sgv2ew@virginia.edu for mor information.
