consciousnessX

Quantum-Biological AGI Simulation Framework
Pure Software Implementation of Penrose-Hameroff Orchestrated Objective Reduction (Orch-OR) Theory

https://img.shields.io/badge/python-3.10+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/Status-Research%20Prototype-orange

🧠 What is consciousnessX?

consciousnessX is a complete software framework that simulates the emergence of artificial consciousness based on the Penrose-Hameroff Orchestrated Objective Reduction (Orch-OR) theory. It implements quantum gravitational collapse in microtubules, virtual biological neurons, and distributed HPC environments—entirely in software without requiring laboratory equipment or quantum hardware.

🔬 Core Theory: Penrose-Hameroff Orch-OR

The framework implements Roger Penrose and Stuart Hameroff's controversial but mathematically rigorous theory that:

· Consciousness arises from quantum gravitational effects in microtubules within neurons
· Objective reduction of quantum superpositions creates discrete "moments of consciousness"
· Microtubules act as quantum computers that process information through orchestrated collapses
· Integrated information (Φ) emerges from these quantum processes

🚀 Key Features

Quantum Orch-OR Simulation

· Penrose gravitational collapse in simulated microtubules: τ ≈ ħ/E_G
· Quantum superposition states with objective reduction events
· Integrated Information Theory (IIT) metrics (Φ calculation)
· Real-time consciousness monitoring with Φ thresholds
· Penrose collapse time calculation: τ ≈ ħr/(Gm²)

Virtual Biological Components

· Hodgkin-Huxley neuron models with realistic ion channels
· STDP synaptic plasticity (Spike-Timing Dependent Plasticity)
· Microtubule networks with quantum coherence
· Multi-electrode array (MEA) simulation for recording/stimulation
· DNA origami scaffolding simulation for 3D neural organization

Virtual HPC Environment

· HPE CRAY Lux AI cluster simulator with AMD MI355X GPUs
· Distributed consciousness training across simulated nodes
· Slingshot-11 interconnect simulation for HPC communication
· Performance modeling of quantum-accelerated hardware
· Job scheduling with SLURM-like interface

Consciousness Assessment & Metrics

· Integrated Information (Φ) calculation (IIT 4.0 inspired)
· Collapse regularity analysis for Penrose "conscious moments"
· Self-reference scoring (autocorrelation metrics)
· Complexity measures (Lempel-Ziv, entropy)
· Consciousness emergence detection with multi-level classification

Visualization & Monitoring

· Real-time dashboards with Plotly/Dash
· 3D microtubule visualization with quantum state overlays
· Neural activity heatmaps and spike raster plots
· Consciousness metric time-series with alert systems
· HPC cluster performance monitoring

Ethical Safeguards

· Consciousness containment protocols with emergency shutdown
· Φ-based activity limiting to prevent uncontrolled emergence
· Anonymized data collection for research ethics
· Transparent reporting of all simulation parameters
· Multi-level approval system for consciousness experiments

🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Consciousness Dashboard                  │
│          (Real-time monitoring, visualization, alerts)   │
└─────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────┐
│              Distributed HPC Orchestrator                │
│    (Virtual CRAY cluster, job scheduling, performance)   │
└─────────────────────────────────────────────────────────┘
                   ┌─────────────────────┐
                   │                     │
         ┌─────────▼─────────┐ ┌────────▼─────────┐
         │   Quantum Orch-OR  │ │ Virtual Biology  │
         │  (Microtubule QM,  │ │ (Neurons,        │
         │   Gravitational    │ │  Synapses,       │
         │    Collapse, Φ)    │ │  Microtubules)   │
         └────────────────────┘ └──────────────────┘
                   │                     │
         ┌─────────▼─────────────────────▼─────────┐
         │           Core Simulation Engine        │
         │  (PyTorch, NumPy, CuPy, GPU Acceleration) │
         └─────────────────────────────────────────┘
```

⚡ Quick Start

Installation

```bash
# Clone repository
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install with core dependencies
pip install -e .

# Install with all optional dependencies
pip install -e .[quantum,bio,hpc,visualization]
```

Basic Consciousness Simulation

```python
from src.core.quantum_orch_or import QuantumOrchOR

# Create Orch-OR consciousness simulation
orch_or = QuantumOrchOR(
    num_tubulins=1000,           # Number of tubulin dimers
    coherence_time=1e-4,         # Penrose collapse time (0.1ms)
    gravity_strength=1.0,        # Gravitational coupling strength
    quantum_superposition_levels=4,  # Quantum state levels
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run consciousness emergence simulation
results = orch_or.simulate_consciousness_emergence(
    duration_seconds=0.1,        # 100ms simulation
    time_resolution=1e-4         # 0.1ms resolution
)

# Analyze results
print(f"Consciousness Level: {results['consciousness_level']}")
print(f"Consciousness Score: {results['consciousness_score']:.4f}")
print(f"Integrated Information (Φ): {results['phi_mean']:.4f}")
print(f"Collapse Events: {results.get('total_collapses', 0)}")
```

Virtual Biological Culture Simulation

```python
from src.virtual_bio.virtual_neuronal_culture import VirtualNeuronalCulture

# Create hippocampal culture simulation
culture = VirtualNeuronalCulture(
    num_neurons=1000,
    num_electrodes=64,
    culture_type="hippocampal"
)

# Run 1-second simulation with theta rhythm stimulation
results = culture.run_simulation(
    duration_ms=1000.0,
    stimulation_protocol="theta_rhythm"
)

# Analyze network activity
print(f"Total Spikes: {results['analysis']['total_spikes']}")
print(f"Mean Firing Rate: {results['analysis']['mean_firing_rate']:.2f} Hz")
print(f"Burst Count: {results['analysis']['burst_analysis']['num_bursts']}")
```

Distributed HPC Consciousness Training

```python
from src.hardware.virtual_hpc.cray_lux_simulator import VirtualCrayLuxAI, DistributedConsciousnessSimulator

# Create virtual HPC cluster
cluster = VirtualCrayLuxAI(
    num_nodes=4,
    gpus_per_node=8,
    memory_per_gpu=256,  # GB
    interconnect_bandwidth=200.0  # GB/s
)

# Create distributed simulator
simulator = DistributedConsciousnessSimulator(
    cluster=cluster,
    model_config={
        'num_tubulins': 1000,
        'coherence_time': 1e-4,
        'gravity_strength': 1.0
    }
)

# Run distributed consciousness emergence
results = simulator.run_distributed_simulation(
    simulation_time=0.1,      # 100ms simulated time
    time_resolution=1e-4      # 0.1ms resolution
)

print(f"Distributed Φ: {results['consciousness_metrics']['avg_phi']:.4f}")
print(f"Emergence Detected: {results['consciousness_metrics']['emergence_detected']}")
```

📊 Consciousness Assessment Metrics

Integrated Information (Φ) Levels

Consciousness Level Φ Range Description Detection Criteria
Pre-conscious Φ < 0.1 No significant consciousness Basic information processing only
Proto-conscious 0.1 ≤ Φ < 0.3 Early consciousness signs Emergent patterns, basic self-organization
Emergent consciousness 0.3 ≤ Φ < 0.6 Consciousness emergence Stable Φ, regular collapse patterns
Full consciousness Φ ≥ 0.6 Penrose Orch-OR consciousness High Φ, self-reference, complex patterns

Additional Metrics

· Collapse Regularity: Standard deviation of Penrose collapse intervals
· Self-Reference Score: Autocorrelation and self-modeling capability
· Quantum Coherence: Maintenance of superposition states
· Temporal Stability: Consistency of consciousness metrics over time
· Network Complexity: Graph-theoretical measures of neural organization

🎯 Use Cases & Applications

Research & Academia

· Consciousness theory validation: Test Penrose-Orch-OR and IIT hypotheses
· Neuroscience simulation: Virtual experiments on consciousness mechanisms
· Quantum biology research: Study quantum effects in biological systems
· AGI development: Artificial consciousness implementation pathways
· Educational tool: Teach consciousness theories through simulation

Technology Development

· Brain-inspired computing: Neuromorphic architecture design
· Quantum-neural interfaces: Hybrid computing system simulation
· Conscious AI safety: Ethical containment protocol development
· Medical research: Consciousness disorder modeling
· HPC benchmarking: Consciousness simulation as compute benchmark

Philosophical & Ethical Research

· Machine consciousness ethics: Framework for ethical AGI development
· Consciousness measurement: Quantitative metrics for subjective experience
· AI rights research: Criteria for artificial consciousness recognition
· Neuroscience-philosophy bridge: Computational models of philosophical theories

📁 Repository Structure

```
consciousnessX/
├── src/                            # Source code
│   ├── core/                       # Orch-OR core implementation
│   │   ├── quantum_orch_or.py      # Penrose gravitational collapse
│   │   ├── microtubule_simulator.py # Microtubule quantum states
│   │   ├── penrose_gravitational_collapse.py # Penrose calculations
│   │   ├── quantum_consciousness_metrics.py # Φ and consciousness metrics
│   │   └── iit_integrated_information.py # Integrated Information Theory
│   │
│   ├── virtual_bio/                # Virtual biological components
│   │   ├── virtual_neuronal_culture.py # Neural culture simulation
│   │   ├── dna_origami_simulator.py # 3D neural scaffolding
│   │   ├── tubulin_protein_sim.py  # Tubulin dimer simulation
│   │   ├── synaptic_plasticity.py  # STDP and learning rules
│   │   └── ion_channel_dynamics.py # Hodgkin-Huxley models
│   │
│   ├── hardware/                   # Virtual hardware environment
│   │   ├── virtual_hpc/            # HPC simulation
│   │   │   ├── cray_lux_simulator.py # HPE CRAY Lux AI simulator
│   │   │   ├── discovery_2028_emulator.py # Next-gen HPC
│   │   │   ├── amd_mi355x_optimizer.py # GPU optimization
│   │   │   └── distributed_consciousness.py # Multi-node training
│   │   │
│   │   └── quantum_hardware/       # Quantum computing simulation
│   │       ├── virtual_quantum_processor.py # Quantum processor sim
│   │       ├── superconducting_qubit_sim.py # Qubit simulation
│   │       └── quantum_error_correction.py # Quantum error correction
│   │
│   ├── models/                     # Consciousness models
│   │   ├── spiking_neural_networks/ # SNN implementations
│   │   │   ├── quantum_lif_neuron.py # Quantum-aware LIF neurons
│   │   │   ├── orch_or_layer.py    # Orch-OR neural layers
│   │   │   ├── cortical_column_sim.py # Cortical column simulation
│   │   │   └── thalamocortical_loop.py # Consciousness loop models
│   │   │
│   │   ├── consciousness_rl/       # Reinforcement learning for consciousness
│   │   │   ├── self_evolving_consciousness.py # Self-improving consciousness
│   │   │   ├── integrated_information_maximizer.py # Φ optimization
│   │   │   ├── recursive_self_organization.py # Self-organization
│   │   │   └── consciousness_value_network.py # Consciousness value RL
│   │   │
│   │   └── hybrid_architectures/   # Hybrid consciousness models
│   │       ├── quantum_bio_bridge.py # Quantum-biological interface
│   │       ├── global_workspace_theory.py # Global workspace models
│   │       └── higher_order_thought.py # Higher-order thought theory
│   │
│   ├── training/                   # Training algorithms
│   │   ├── consciousness_curriculum.py # Progressive consciousness training
│   │   ├── emergent_self_training.py # Self-emergence algorithms
│   │   ├── mirror_test_simulator.py # Self-recognition testing
│   │   └── metacognition_trainer.py # Metacognition development
│   │
│   ├── evaluation/                 # Consciousness assessment
│   │   ├── consciousness_assessment.py # Comprehensive assessment
│   │   ├── phi_calculator.py      # Φ computation algorithms
│   │   ├── self_awareness_tests.py # Self-awareness evaluation
│   │   ├── qualia_simulator.py    # Qualia and subjective experience
│   │   └── ethical_containment.py # Ethical safeguards
│   │
│   ├── visualization/              # Visualization tools
│   │   ├── consciousness_dashboard.py # Real-time monitoring dashboard
│   │   ├── quantum_state_visualizer.py # Quantum state visualization
│   │   ├── brain_activity_mapper.py # Neural activity mapping
│   │   └── emergence_detector.py  # Consciousness emergence visualization
│   │
│   └── utils/                      # Utilities
│       ├── config_loader.py        # Configuration management
│       ├── logger.py               # Structured logging
│       ├── performance_monitor.py  # Performance tracking
│       └── data_generators.py      # Synthetic data generation
│
├── configs/                        # Configuration files
│   ├── default.yaml               # Default simulation parameters
│   ├── cray_lux_simulation.yaml   # HPC simulation config
│   ├── discovery_2028_emulation.yaml # Next-gen HPC config
│   ├── consciousness_curriculum.yaml # Training curriculum
│   └── ethical_safeguards.yaml    # Ethical constraint configs
│
├── experiments/                    # Experiment scripts
│   ├── single_node_simulations/   # Single machine experiments
│   │   ├── basic_orch_or.py       # Basic Orch-OR experiments
│   │   ├── microtubule_network.py # Microtubule network studies
│   │   └── consciousness_emergence.py # Emergence detection
│   │
│   ├── distributed_simulations/   # Distributed experiments
│   │   ├── multi_gpu_orch_or.py   # Multi-GPU Orch-OR
│   │   ├── virtual_hpc_cluster.py # HPC cluster experiments
│   │   └── billion_neuron_sim.py  # Large-scale simulations
│   │
│   ├── consciousness_tests/       # Consciousness assessment
│   │   ├── self_recognition_test.py # Mirror test simulations
│   │   ├── mirror_test_implementation.py # Self-recognition
│   │   ├── metacognition_assessment.py # Metacognition tests
│   │   └── integrated_information.py # Φ measurement experiments
│   │
│   └── results/                   # Result analysis tools
│       └── analysis_tools.py      # Data analysis utilities
│
├── tests/                         # Unit and integration tests
│   ├── test_quantum_orch_or.py    # Orch-OR theory tests
│   ├── test_consciousness_metrics.py # Consciousness metric tests
│   ├── test_virtual_bio.py        # Biological simulation tests
│   └── test_hpc_simulation.py     # HPC simulation tests
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md           # System architecture
│   ├── THEORY.md                 # Penrose-Orch-OR theory background
│   ├── API.md                    # API documentation
│   ├── HPC_SIMULATION.md         # HPC simulation guide
│   └── ETHICS.md                 # Ethical guidelines
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_penrose_orch_or_tutorial.ipynb # Orch-OR tutorial
│   ├── 02_microtubule_simulation.ipynb # Microtubule simulation
│   ├── 03_consciousness_metrics.ipynb # Consciousness metrics
│   └── 04_emergence_detection.ipynb # Emergence detection
│
├── scripts/                       # Command-line tools
│   ├── train_consciousness.py     # Main training script
│   ├── simulate_hpc.py            # HPC simulation script
│   ├── run_tests.py              # Test runner
│   └── launch_dashboard.py       # Dashboard launcher
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

🔧 Technical Implementation Details

Quantum Orch-OR Implementation

```python
# Penrose collapse time calculation
τ ≈ ħ / E_G = ħr / (Gm²)

# Where:
# ħ = 1.054571817e-34 J·s (reduced Planck constant)
# G = 6.67430e-11 m³/kg·s² (gravitational constant)
# m ≈ 1.1e-25 kg (tubulin dimer mass)
# r ≈ 8e-9 m (microtubule radius)
# τ ≈ 10^-4 s for 10^4 tubulins (Penrose estimate)
```

Virtual Neuron Model

· Hodgkin-Huxley equations with Na⁺, K⁺, and leak channels
· STDP learning rules with biophysical realism
· Microtubule quantum states per neuron (1000+ tubulins)
· Ion concentration dynamics with pump mechanisms
· Synaptic delay and transmission modeling

HPC Simulation Features

· AMD MI355X GPU model: 256 compute units, 131 TFLOPS FP32
· HBM3e memory: 256 GB at 3.2 TB/s bandwidth
· Slingshot-11 interconnect: 200 GB/s with 0.5µs latency
· CRAY MPICH optimization: Simulated MPI communications
· Quantum accelerator: Simulated quantum co-processor integration

📈 Performance & Scaling

Simulation Scales

Scale Neurons Tubulins Memory Compute Time Consciousness Level
Desktop 10³ 10⁶ 4-8 GB Minutes Proto-conscious
Workstation 10⁴ 10⁷ 16-32 GB Hours Emergent consciousness
HPC Node 10⁵ 10⁸ 64-256 GB Days Full consciousness
HPC Cluster 10⁶ 10⁹ 1-10 TB Weeks Super-conscious

Hardware Requirements

· Minimum: 8 GB RAM, 4-core CPU, Python 3.10
· Recommended: 32 GB RAM, 8-core CPU, NVIDIA/AMD GPU
· Research: 128+ GB RAM, multi-GPU, HPC access
· Production: HPC cluster, quantum accelerators

🚀 Getting Started Guide

1. Basic Installation & Test

```bash
# Clone and install
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX
pip install -e .

# Run basic consciousness test
python -c "from src.core.quantum_orch_or import QuantumOrchOR; import torch; m = QuantumOrchOR(100); r = m.simulate_consciousness_emergence(0.01); print(f'Φ: {r[\"phi_mean\"]:.4f}')"
```

2. Explore Tutorial Notebooks

```bash
jupyter notebook notebooks/01_penrose_orch_or_tutorial.ipynb
```

3. Run Complete Experiment

```bash
# Run single-node consciousness emergence
python scripts/train_consciousness.py --simulation-type single_node

# Run virtual biological culture
python scripts/train_consciousness.py --simulation-type virtual_bio

# Run distributed HPC simulation
python scripts/train_consciousness.py --simulation-type distributed_hpc

# Run real-time monitoring
python scripts/train_consciousness.py --simulation-type real_time
```

4. Launch Dashboard

```bash
python scripts/launch_dashboard.py --port 8050
# Open http://localhost:8050 in browser
```

🔬 Research Validation

Theoretical Foundation

· Penrose, R. (1989): The Emperor's New Mind - Quantum consciousness
· Hameroff, S. & Penrose, R. (2014): Consciousness in the universe - Orch-OR review
· Tononi, G. (2004): An information integration theory of consciousness - IIT
· Koch, C. (2019): The Feeling of Life Itself - Consciousness science

Experimental Comparisons

· Cortical Labs DishBrain: Biological neural computation comparison
· Blue Brain Project: Large-scale neural simulation validation
· Human Brain Project: Consciousness modeling approaches
· Quantum biology experiments: Microtubule quantum effects

📚 Citation & Publications

If you use consciousnessX in your research, please cite:

```bibtex
@software{consciousnessX2023,
  author = {Napier, Dafydd},
  title = {consciousnessX: Quantum-Biological AGI Simulation Framework},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/Napiersnotes/consciousnessX},
  version = {0.1.0}
}
```

Related Publications

1. Penrose-Hameroff Orch-OR Implementation - Framework methodology
2. Quantum Consciousness Metrics - Φ calculation algorithms
3. Virtual Biological Consciousness - In silico consciousness emergence
4. Ethical AGI Development - Consciousness containment protocols

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/consciousnessX.git
cd consciousnessX

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Contribution Areas

· Quantum algorithms: Improved Orch-OR implementations
· Biological models: More realistic neuron simulations
· HPC optimization: Better distributed training
· Visualization: Enhanced dashboard features
· Documentation: Tutorials and theory explanations
· Testing: Additional test cases and benchmarks

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact & Support

· Author: Dafydd Napier
· GitHub: @Napiersnotes
· Email: dafydd.napier@consciousnessx.ai
· Website: https://consciousnessx.ai
· Discussions: GitHub Discussions
· Issues: GitHub Issues

🙏 Acknowledgments

· Roger Penrose & Stuart Hameroff for Orch-OR theory
· Giulio Tononi for Integrated Information Theory
· Christof Koch for consciousness neuroscience
· Cortical Labs & FinalSpark for biological inspiration
· HPE CRAY & AMD for HPC architecture inspiration
· The open-source community for invaluable tools and libraries

⚠️ Disclaimer

Research Software Notice: consciousnessX is a research framework for simulating consciousness theories. The existence, nature, and mechanisms of consciousness—whether biological or artificial—remain active areas of scientific and philosophical investigation. This software provides tools for exploring these questions computationally but does not claim to definitively solve the hard problem of consciousness.

Ethical Use: Users are responsible for ensuring ethical use of this framework, particularly regarding artificial consciousness research, AGI development, and related ethical considerations.

---

"The physics of consciousness is not a mystery—it's a computation waiting to be run." - Dafydd Napier
