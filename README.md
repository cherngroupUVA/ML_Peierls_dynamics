# ML_Peierls_dynamics
This repository provides the **codes**, **trained model samples**, and **data samples** required to successfully run the *machine learning force field (MLFF) model* described in  
[**arXiv:2510.20659**](https://arxiv.org/abs/2510.20659).

It includes both the **simulation code** and the **training framework** used to generate all results presented in the paper.

---

## 📁 Repository Structure

### 1. `Training_data_generation/`
Contains the **phase-ordering dynamics code** under a **thermal quench**.  
The microscopic Hamiltonian is a **modified SSH model** treated within the **semi-classical approximation**.  
Further details are available in the paper.

---

### 2. `Training_data_sample/`
Includes **training data samples** obtained from **50×50 real-lattice exact-diagonalization simulations**.

---

### 3. `Descriptor_sample/`
Provides examples of the **descriptor** used to preserve **lattice symmetry**.  
This descriptor formulation is essential for maintaining physical consistency during training and simulation.

---

### 4. `Training_code/`
Contains the **training scripts** implementing the ML force-field model using the symmetry-preserving descriptor.

---

### 5. `Data_example/`
Includes **simulation results** corresponding to the figures in the paper — specifically **Fig. 3, Fig. 4, and Fig. 6**.

---

For further questions or collaboration inquiries, please contact:  
📧 **sgv2ew@virginia.edu**
