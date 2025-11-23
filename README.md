# Symbolic and Algebraic Reasoning in Petri Nets (CO2011 Assignment)

This repository contains the implementation of a **1-safe Petri Net Analyzer** developed for the Mathematical Modeling course (CO2011). The toolchain integrates three complementary techniques to analyze concurrent systems:
1.  **Explicit State Enumeration (BFS)** for baseline validation.
2.  **Symbolic Reachability (BDDs)** for scalable state-space exploration.
3.  **Hybrid Reasoning (BDD + ILP)** for deadlock detection and optimization.

## 📂 Project Structure

The project is organized as follows:

```text
PETRIANALYZER/
├── cases/                      # Test PNML models
│   ├── functional_test.pnml    # Synthetic net for correctness verification
│   ├── philosophers_12.pnml    # Dining Philosophers (N=12) for scalability test
│   └── switches_15.pnml        # 15 Toggle Switches for adaptive optimization test
├── logs/                       # Output logs from experiments (auto-generated)
├── src/                        # Source code
│   ├── model.py                # PetriNet data structure & 1-safe semantics
│   ├── utils.py                # Performance profiling (Time & Memory)
│   ├── Task1_pnml_parser.py    # XML Parser for PNML files
│   ├── Task2_explicit.py       # Explicit Reachability (BFS)
│   ├── Task3_bdd_reach.py      # Symbolic Reachability (BDD)
│   ├── Task4_deadlock.py       # Hybrid Deadlock Detection (BDD + ILP)
│   ├── Task5_optimize.py       # Optimization (Enumeration & Sampling)
│   └── Task6_cli.py            # Main CLI entry point
├── run_tests.py                # Unit test runner
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.8+**
* **Virtual Environment** (Recommended)

### Setup Steps
Run the following commands in your terminal to set up the environment:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install dependencies (dd, pulp, lxml)
pip install -r requirements.txt

# 4. Create logs directory
mkdir -p logs
```

---
## 🧩 Benchmark Suite

To evaluate the tool's correctness and scalability, we employ three distinct PNML models. Below are the representative visualizations of these networks.

### 1. Functional Verification (`functional_test`)
A synthetic model designed to verify all logic components: Parser, Concurrency (Fork/Join), Self-loop updates, and Deadlock detection.

<p align="center">
  <img src="functional_test.png" alt="Functional Test Petri Net" width="600">
  <br>
  <em>Figure 1: A compact net featuring non-determinism, concurrency, and a trap state.</em>
</p>

### 2. Scalability Benchmark (`philosophers_12`)
**Dining Philosophers ($N=12$)**: A classic high-concurrency model with cyclic resource dependencies.
* **Purpose:** Benchmarking BDD vs. Explicit performance (Memory/Time) and detecting circular deadlocks.
* **Visualization:** The diagram below represents a segment of the ring ($P_i$ and $P_{i+1}$ competing for $Fork_{i+1}$).

<p align="center">
  <img src="philosophers_12.png" alt="Dining Philosophers Representative View" width="100%">
  <br>
  <em>Figure 2: Representative segment of the 12-Philosopher ring showing resource competition.</em>
</p>

### 3. Optimization Benchmark (`switches_15`)
**Independent Switches ($N=15$)**: A model causing state-space explosion ($2^{15}$ states) with simple structure.
* **Purpose:** Triggering the **Adaptive Optimization** mechanism (switching from Enumeration to Sampling).
* **Visualization:** The diagram shows independent switch modules.

<p align="center">
  <img src="switches_15.png" alt="Switches 15 Representative View" width="100%">
  <br>
  <em>Figure 3: Independent toggle switch modules generating a combinatorial state space.</em>
</p>

---
## 🚀 Reproducing Experimental Results

We provide specific commands to reproduce the results reported in the **Experimental Evaluation** section of the report.

### 1. Functional Verification
* **Model:** `functional_test.pnml`
* **Goal:** Verify correctness of Parser, Reachability, Deadlock (finding the trap state), and Optimization (avoiding the trap).
* **Command:**
    ```bash
    python src/Task6_cli.py --pnml cases/functional_test.pnml \
      --objective custom --weights 0,0,0,0,10 \
      --confirm_ilp --seed 42 | tee logs/functional_test.log
    ```
    *(Note: The custom weight vector `0,0,0,0,10` targets the `p_final` place to test if the optimizer can bypass the deadlock).*

### 2. Performance Benchmark (Explicit vs. BDD)
* **Model:** `philosophers_12.pnml` (State space: ~39,202)
* **Goal:** Demonstrate the "Variable Ordering" phenomenon where BDD consumes significantly more memory than Explicit BFS due to dependency distance.
* **Command:**
    ```bash
    # Use -u (unbuffered) to ensure logs are written immediately during long runs
    python -u src/Task6_cli.py --pnml cases/philosophers_12.pnml \
      --objective uniform \
      --confirm_ilp --seed 42 | tee logs/philosophers_12.log
    ```

### 3. Adaptive Optimization (Sampling Mode)
* **Model:** `switches_15.pnml` (State space: 32,768)
* **Goal:** Trigger the adaptive mechanism. Since the state space > 5000 (threshold), the tool will switch from `enumerate` to `sample_bdd` to find the optimum efficiently.
* **Command:**
    ```bash
    python src/Task6_cli.py --pnml cases/switches_15.pnml \
      --objective uniform \
      --enumeration_threshold 5000 \
      --sample_limit 1000 \
      --seed 42 | tee logs/switches_15.log
    ```

---

## 📊 Expected Output

After running the commands, check the `logs/` folder. The summary at the bottom of each log file should match the data in the report:

* **functional_test:** `Deadlock: found`, `Optimization: found` (Value=10 at `p_final`).
* **philosophers_12:** `Explicit time` << `BDD time`, `Deadlock: found` (Circular wait).
* **switches_15:** `Optimization method: sample_bdd`, `Value: 15`.

## 👥 Authors (Group 2H3D)
* Nguyễn Trần Hoàng Hiếu (2452332)
* Nguyễn Kiên Đức (2452278)
* Phan Minh Hưng (2452427)
* Trần Chí Đại (2452237)
* Mai Xuân Đức (2452275)
