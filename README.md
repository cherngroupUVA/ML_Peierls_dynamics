# ML_Peierls_dynamics
# Machine Learning Force Field Model

This repository includes codes, trained model samples, and data samples required to run the machine learning force field (MLFF) model described in  
[arXiv:2510.20659](https://arxiv.org/abs/2510.20659).

It contains both the simulation code and the training code used to generate the results presented in the paper.

---

## Repository Structure

### 1. Training_data_generation
Code for the phase-ordering dynamics under a thermal quench.  
The microscopic Hamiltonian is a modified SSH model treated within the semi-classical approximation.  
More details are provided in the paper.

### 2. Training_data_sample
Training data samples obtained from 50×50 real-lattice exact-diagonalization simulations.

### 3. Descriptor_sample
Descriptor examples used to preserve lattice symmetry in the ML model.

### 4. Training_code
Training scripts implementing the machine learning force field model using the symmetry-preserving descriptor.

### 5. Data_example
Simulation results corresponding to the figures in the paper (Fig. 3, Fig. 4, and Fig. 6).

---

For further information, please contact:  
**sgv2ew@virginia.edu**
