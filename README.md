# 🧬 Q-AMP: Quantum Antimicrobial Peptide Designer

The rise of drug resistant bacteria poses a major global health challenge, requiring new computational methods to design effective treatments quickly. In this project, we develop Q-AMP, a Python based software tool designed to model and optimize peptide sequences. Instead of testing thousands of combinations manually on regular computers, the tool uses a physics based mathematical approach combined with quantum optimization algorithms (QAOA) to evaluate molecular trade offs simultaneously.
​By taking the protein chain length as an input, the program calculates the optimal arrangement of building block maximizing bacterial membrane destruction while minimizing toxicity to human cells. The software processes these constraints through an energy minimization model and generates a clear structural profile alongside visual graphs. This demonstrates how quantum inspired optimization can assist researchers in rapidly designing targeted molecular structures for medical applications.

---

## 🚀 Features

* **Quantum Optimization Core**: Leverages a matrix-based QAOA simulation with $p=2$ circuit layers and classical COBYLA optimization to find ground-state energy minimums.
* **Biophysical Constraint Modeling**: Balances hydrophobic (Leucine) and hydrophilic (Lysine) residue arrangements to prevent red blood cell toxicity.
* **Automated Sequence Decoding**: Translates optimal binary bitstrings into clean biological amino acid sequence chains.
* **Structural Profile Visualization**: Dynamically generates and saves Matplotlib structural graphs and probability distribution outputs.
* **Modular Architecture**: Cleanly separated script modules allowing multi-developer GitHub collaboration across a 7-member team.

---

## 🛠️ Tech Stack

- **Core Language**: Python 3
- **Mathematical & Optimization Libraries**: NumPy, SciPy (`scipy.optimize.minimize`)
- **Data Visualization**: Matplotlib
- **Algorithm Framework**: Quantum Approximate Optimization Algorithm (QAOA), Ising Spin Hamiltonians

---

## 📂 Project Structure

    Q-AMP/
    │
    ├── main.py                  # Main execution script & orchestration
    ├── qubo_matrix.py           # Quadratic Unconstrained Binary Optimization matrix builder
    ├── ising_mapper.py          # Ising Hamiltonian conversion & Pauli-Z mapping
    ├── qaoa_circuit.py          # QAOA statevector evolution circuit (p=2 layers)
    ├── optimizer.py             # Classical COBYLA optimization loop
    ├── decoder.py               # Optimal bitstring to amino acid sequence decoder
    ├── visualization.py         # Matplotlib structural profile graphing engine
    └── requirements.txt         # Required Python dependencies

---

## ⚙️ Installation & Setup

Follow these steps to run the simulation locally on your machine:

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YourUsername/Q-AMP.git](https://github.com/YourUsername/Q-AMP.git)
   cd Q-AMP

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the Simulation Script**
   ```bash
   python main.py

4. **View Outputs**
   The script will print the optimal ground-state sequence (Leu-Lys-Leu-Lys-Leu-Lys) and energy value (-4.50 eV), and save the structural profile plot as qaoa_amp_results.png

---

## 👥 Project Team

* Anushreya Satish 
* Drishti Jhunjhunwala 
* Gurshan Singh
* Shibasis Chatterjee
* Srijan Chattoraj
* Satyajit Chakraborty 
* Disha Verma

---

## 📜 License

This project is open-source and available under the MIT License. Feel free to fork, modify, and use it!


---

<p align="center">
  🧬 <b>Built with passion by the Q-AMP Team</b><br>
  <sub>Exploring Quantum Horizons & Computational Biology</sub>
</p>
