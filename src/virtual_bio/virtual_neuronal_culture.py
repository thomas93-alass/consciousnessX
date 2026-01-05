"""
Virtuelle neuronale Kulturen für Bewusstseins-Simulation
Simuliert biologische Neuronen ohne Labor-Zugang
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class VirtualNeuronalCulture:
    """
    Simuliert eine biologische neuronale Kultur in Silico
    Mit realistischen Ionenkanälen, Aktionspotentialen und Synapsen
    """
    
    def __init__(self, 
                 num_neurons: int = 1000,
                 num_electrodes: int = 64,
                 culture_type: str = "hippocampal",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.num_neurons = num_neurons
        self.num_electrodes = num_electrodes
        self.culture_type = culture_type
        self.device = device
        
        # Neuronale Typen basierend auf Kultur-Typ
        self.neuron_types = self.initialize_neuron_types()
        
        # Ionenkanal-Parameter (Hodgkin-Huxley Modell)
        self.ion_channels = self.initialize_ion_channels()
        
        # Synaptische Verbindungen
        self.synapses = self.initialize_synapses()
        
        # Mikrotubuli-Netzwerk (für Orch-OR)
        self.microtubules = self.initialize_microtubules()
        
        # Elektroden-Array für Recording/Stimulation
        self.electrodes = self.initialize_electrodes()
        
        # Zeitliche Zustände
        self.membrane_potentials = torch.zeros(num_neurons, device=device)
        self.ion_concentrations = self.initialize_ion_concentrations()
        self.spike_history = []
        self.activity_history = []
        
        # Biophysikalische Konstanten
        self.C_m = 1.0  # Membrankapazität (µF/cm²)
        self.dt = 0.01  # Zeitschritt (ms)
        self.temperature = 36.5  # °C
        
        print(f"Virtuelle {culture_type} Kultur initialisiert:")
        print(f"  Neuronen: {num_neurons}")
        print(f"  Elektroden: {num_electrodes}")
        print(f"  Synapsen: {len(self.synapses['connections'])}")
        print(f"  Mikrotubuli: {self.microtubules['num_tubulins']}")
    
    def initialize_neuron_types(self) -> Dict:
        """Initialisiere verschiedene Neuronentypen"""
        
        neuron_distribution = {
            'excitatory': 0.8,  # 80% exzitatorisch
            'inhibitory': 0.15, # 15% inhibitorisch
            'modulatory': 0.05   # 5% modulatorisch
        }
        
        # Weise jedem Neuron einen Typ zu
        neuron_types = []
        for i in range(self.num_neurons):
            rand = random.random()
            if rand < neuron_distribution['excitatory']:
                neuron_types.append('excitatory')
            elif rand < neuron_distribution['excitatory'] + neuron_distribution['inhibitory']:
                neuron_types.append('inhibitory')
            else:
                neuron_types.append('modulatory')
        
        # Typ-spezifische Parameter
        type_params = {
            'excitatory': {
                'resting_potential': -65.0,  # mV
                'threshold': -50.0,           # mV
                'refractory_period': 2.0,     # ms
                'max_firing_rate': 100.0,     # Hz
                'neurotransmitter': 'glutamate'
            },
            'inhibitory': {
                'resting_potential': -60.0,
                'threshold': -45.0,
                'refractory_period': 1.0,
                'max_firing_rate': 200.0,
                'neurotransmitter': 'GABA'
            },
            'modulatory': {
                'resting_potential': -55.0,
                'threshold': -40.0,
                'refractory_period': 5.0,
                'max_firing_rate': 50.0,
                'neurotransmitter': 'acetylcholine'
            }
        }
        
        return {
            'types': neuron_types,
            'params': type_params,
            'refractory_counters': torch.zeros(self.num_neurons, device=self.device)
        }
    
    def initialize_ion_channels(self) -> Dict:
        """Initialisiere Ionenkanäle für Hodgkin-Huxley Modell"""
        
        # Natrium-Kanäle (Na⁺)
        na_channels = {
            'g_Na': 120.0,  # maximale Leitfähigkeit (mS/cm²)
            'E_Na': 50.0,   # Nernst-Potential (mV)
            'm': 0.05,      # Aktivierungs-Gating Variable
            'h': 0.6,       # Inaktivierungs-Gating Variable
            'alpha_m': lambda V: 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0)),
            'beta_m': lambda V: 4.0 * torch.exp(-(V + 65.0) / 18.0),
            'alpha_h': lambda V: 0.07 * torch.exp(-(V + 65.0) / 20.0),
            'beta_h': lambda V: 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
        }
        
        # Kalium-Kanäle (K⁺)
        k_channels = {
            'g_K': 36.0,
            'E_K': -77.0,
            'n': 0.32,
            'alpha_n': lambda V: 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0)),
            'beta_n': lambda V: 0.125 * torch.exp(-(V + 65.0) / 80.0)
        }
        
        # Leck-Kanäle
        leak_channels = {
            'g_L': 0.3,
            'E_L': -54.387
        }
        
        return {
            'Na': na_channels,
            'K': k_channels,
            'Leak': leak_channels
        }
    
    def initialize_synapses(self) -> Dict:
        """Initialisiere synaptische Verbindungen"""
        
        # Erzeuge zufälliges konnektom (small-world Netzwerk)
        connections = []
        weights = []
        types = []
        delays = []
        
        # Verbindungs-Wahrscheinlichkeit (basierend auf Abstand)
        positions = torch.randn(self.num_neurons, 3, device=self.device)
        distance_matrix = torch.cdist(positions, positions)
        
        connection_prob = 0.1 * torch.exp(-distance_matrix / 2.0)
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue
                
                # Verbindung basierend auf Wahrscheinlichkeit
                if random.random() < connection_prob[i, j].item():
                    connections.append((i, j))
                    
                    # Bestimme synaptischen Typ basierend auf Neuronentypen
                    pre_type = self.neuron_types['types'][i]
                    post_type = self.neuron_types['types'][j]
                    
                    # Synaptisches Gewicht
                    if pre_type == 'excitatory':
                        weight = random.uniform(0.1, 0.5)
                        syn_type = 'excitatory'
                    elif pre_type == 'inhibitory':
                        weight = random.uniform(-0.3, -0.1)
                        syn_type = 'inhibitory'
                    else:  # modulatory
                        weight = random.uniform(-0.1, 0.3)
                        syn_type = 'modulatory'
                    
                    weights.append(weight)
                    types.append(syn_type)
                    
                    # Synaptische Verzögerung (1-5ms)
                    delay = random.uniform(1.0, 5.0)
                    delays.append(delay)
        
        # Synaptische Plastizität (STDP Parameter)
        stdp_params = {
            'tau_plus': 20.0,   # ms
            'tau_minus': 20.0,  # ms
            'A_plus': 0.01,
            'A_minus': 0.0105,
            'w_max': 1.0,
            'w_min': 0.0
        }
        
        return {
            'connections': connections,
            'weights': torch.tensor(weights, device=self.device),
            'types': types,
            'delays': torch.tensor(delays, device=self.device),
            'stdp_params': stdp_params,
            'spike_times': [[] for _ in range(len(connections))]
        }
    
    def initialize_microtubules(self) -> Dict:
        """Initialisiere Mikrotubuli für Orch-OR Simulation"""
        
        # Anzahl der Tubulin-Dimere pro Neuron
        tubulins_per_neuron = 1000
        
        # Gesamtzahl der Tubuline
        total_tubulins = self.num_neurons * tubulins_per_neuron
        
        # Tubulin-Zustände (quantenmechanische Superposition)
        tubulin_states = torch.randn(total_tubulins, 2, dtype=torch.complex64, device=self.device)
        tubulin_states = tubulin_states / torch.norm(tubulin_states, dim=1, keepdim=True)
        
        # Mikrotubuli-Organisation (13 Protofilamente)
        microtubule_organization = []
        for n in range(self.num_neurons):
            for mt in range(13):  # 13 Protofilamente
                start_idx = n * tubulins_per_neuron + mt * (tubulins_per_neuron // 13)
                end_idx = start_idx + (tubulins_per_neuron // 13)
                microtubule_organization.append({
                    'neuron_index': n,
                    'protofilament': mt,
                    'tubulin_indices': list(range(start_idx, end_idx)),
                    'length': (end_idx - start_idx) * 8e-9  # 8nm pro Tubulin
                })
        
        return {
            'num_tubulins': total_tubulins,
            'states': tubulin_states,
            'organization': microtubule_organization,
            'coherence_times': torch.ones(total_tubulins, device=self.device) * 1e-4
        }
    
    def initialize_electrodes(self) -> Dict:
        """Initialisiere MEA (Multi-Electrode Array) Elektroden"""
        
        # Elektroden-Positionen auf einer virtuellen Ebene
        electrode_positions = torch.zeros(self.num_electrodes, 3, device=self.device)
        
        # Grid-Anordnung (8x8 bei 64 Elektroden)
        grid_size = int(np.sqrt(self.num_electrodes))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < self.num_electrodes:
                    electrode_positions[idx, 0] = i * 100e-6  # 100µm Abstand
                    electrode_positions[idx, 1] = j * 100e-6
                    electrode_positions[idx, 2] = 0  # Auf Oberfläche
        
        # Elektroden-Impedanzen
        impedances = torch.ones(self.num_electrodes, device=self.device) * 100e3  # 100kΩ
        
        # Recording-Rauschen
        noise_levels = torch.ones(self.num_electrodes, device=self.device) * 5e-6  # 5µV RMS
        
        return {
            'positions': electrode_positions,
            'impedances': impedances,
            'noise_levels': noise_levels,
            'recordings': [],
            'stimulation_patterns': []
        }
    
    def initialize_ion_concentrations(self) -> Dict:
        """Initialisiere Ionenkonzentrationen"""
        
        # Intrazelluläre Konzentrationen (mM)
        concentrations = {
            'Na_intra': torch.ones(self.num_neurons, device=self.device) * 15.0,
            'K_intra': torch.ones(self.num_neurons, device=self.device) * 140.0,
            'Ca_intra': torch.ones(self.num_neurons, device=self.device) * 0.0001,
            'Cl_intra': torch.ones(self.num_neurons, device=self.device) * 10.0,
            
            'Na_extra': torch.ones(self.num_neurons, device=self.device) * 145.0,
            'K_extra': torch.ones(self.num_neurons, device=self.device) * 4.0,
            'Ca_extra': torch.ones(self.num_neurons, device=self.device) * 2.0,
            'Cl_extra': torch.ones(self.num_neurons, device=self.device) * 120.0
        }
        
        # Ionenpumpen (Na⁺/K⁺-ATPase)
        concentrations['pump_rate'] = torch.ones(self.num_neurons, device=self.device) * 0.1
        
        return concentrations
    
    def update_membrane_potential(self, external_current: Optional[torch.Tensor] = None):
        """
        Aktualisiere Membranpotentiale mit Hodgkin-Huxley Modell
        
        Args:
            external_current: Externer Input-Strom (nA)
        """
        
        if external_current is None:
            external_current = torch.zeros(self.num_neurons, device=self.device)
        
        # Aktuelle Membranpotentiale
        V = self.membrane_potentials
        
        # Ionenkanal-Leitfähigkeiten berechnen
        g_Na = self.ion_channels['Na']['g_Na']
        g_K = self.ion_channels['K']['g_K']
        g_L = self.ion_channels['Leak']['g_L']
        
        E_Na = self.ion_channels['Na']['E_Na']
        E_K = self.ion_channels['K']['E_K']
        E_L = self.ion_channels['Leak']['E_L']
        
        # Gating-Variablen aktualisieren
        m = self.ion_channels['Na']['m']
        h = self.ion_channels['Na']['h']
        n = self.ion_channels['K']['n']
        
        # Alpha/Beta Raten
        alpha_m = self.ion_channels['Na']['alpha_m'](V)
        beta_m = self.ion_channels['Na']['beta_m'](V)
        alpha_h = self.ion_channels['Na']['alpha_h'](V)
        beta_h = self.ion_channels['Na']['beta_h'](V)
        alpha_n = self.ion_channels['K']['alpha_n'](V)
        beta_n = self.ion_channels['K']['beta_n'](V)
        
        # Differentialgleichungen für Gating-Variablen
        dm_dt = alpha_m * (1 - m) - beta_m * m
        dh_dt = alpha_h * (1 - h) - beta_h * h
        dn_dt = alpha_n * (1 - n) - beta_n * n
        
        # Update Gating-Variablen
        m = m + dm_dt * self.dt
        h = h + dh_dt * self.dt
        n = n + dn_dt * self.dt
        
        # Ionenströme
        I_Na = g_Na * (m**3) * h * (V - E_Na)
        I_K = g_K * (n**4) * (V - E_K)
        I_L = g_L * (V - E_L)
        
        # Membranpotential-Differentialgleichung
        dV_dt = (-I_Na - I_K - I_L + external_current) / self.C_m
        
        # Update Membranpotentiale
        new_V = V + dV_dt * self.dt
        
        # Refraktärperioden anwenden
        refractory_mask = self.neuron_types['refractory_counters'] > 0
        new_V[refractory_mask] = self.get_resting_potential(refractory_mask)
        
        # Refraktärzähler verringern
        self.neuron_types['refractory_counters'] = torch.maximum(
            self.neuron_types['refractory_counters'] - self.dt,
            torch.tensor(0.0, device=self.device)
        )
        
        # Spike-Erkennung
        spike_thresholds = self.get_spike_thresholds()
        spikes = new_V > spike_thresholds
        
        if spikes.any():
            # Spike-Handling
            spike_indices = torch.where(spikes)[0]
            self.handle_spikes(spike_indices, new_V)
        
        # Aktualisiere Membranpotentiale
        self.membrane_potentials = new_V
        
        # Aktualisiere Ionenkonzentrationen
        self.update_ion_concentrations(I_Na, I_K)
        
        # Speichere Aktivität
        self.activity_history.append({
            'time': len(self.activity_history) * self.dt,
            'membrane_potentials': self.membrane_potentials.clone().cpu(),
            'spikes': spikes.cpu() if isinstance(spikes, torch.Tensor) else spikes,
            'external_current': external_current.clone().cpu() if external_current is not None else None
        })
        
        return {
            'membrane_potentials': self.membrane_potentials,
            'spikes': spikes,
            'I_Na': I_Na,
            'I_K': I_K,
            'I_L': I_L
        }
    
    def get_resting_potential(self, neuron_mask: torch.Tensor) -> torch.Tensor:
        """Hole Ruhepotentiale für spezifische Neuronen"""
        
        resting_potentials = torch.zeros_like(neuron_mask, dtype=torch.float32, device=self.device)
        
        for i, is_active in enumerate(neuron_mask):
            if is_active:
                neuron_type = self.neuron_types['types'][i]
                resting_potentials[i] = self.neuron_types['params'][neuron_type]['resting_potential']
        
        return resting_potentials
    
    def get_spike_thresholds(self) -> torch.Tensor:
        """Hole Spike-Schwellenwerte für alle Neuronen"""
        
        thresholds = torch.zeros(self.num_neurons, device=self.device)
        
        for i, neuron_type in enumerate(self.neuron_types['types']):
            thresholds[i] = self.neuron_types['params'][neuron_type]['threshold']
        
        return thresholds
    
    def handle_spikes(self, spike_indices: torch.Tensor, membrane_potentials: torch.Tensor):
        """Behandle Spike-Ereignisse"""
        
        for idx in spike_indices:
            # Setze auf Überschwing-Potential
            membrane_potentials[idx] = 40.0  # Spike-Peak
            
            # Setze Refraktärzähler
            neuron_type = self.neuron_types['types'][idx]
            refractory_period = self.neuron_types['params'][neuron_type]['refractory_period']
            self.neuron_types['refractory_counters'][idx] = refractory_period
            
            # Speichere Spike
            spike_time = len(self.activity_history) * self.dt
            self.spike_history.append({
                'neuron_index': idx.item(),
                'time': spike_time,
                'type': neuron_type
            })
            
            # Propagiere synaptische Inputs
            self.propagate_synaptic_inputs(idx, spike_time)
    
    def propagate_synaptic_inputs(self, pre_neuron_idx: int, spike_time: float):
        """Propagiere synaptische Inputs zu postsynaptischen Neuronen"""
        
        # Finde alle Synapsen von diesem präsynaptischen Neuron
        for i, (pre, post) in enumerate(self.synapses['connections']):
            if pre == pre_neuron_idx:
                # Füge Spike-Zeit zur Synapse hinzu
                self.synapses['spike_times'][i].append(spike_time)
                
                # STDP-Lernen (spike-timing dependent plasticity)
                self.apply_stdp(i, pre_neuron_idx, post, spike_time)
    
    def apply_stdp(self, synapse_idx: int, pre_idx: int, post_idx: int, spike_time: float):
        """Wende STDP-Lernen auf Synapse an"""
        
        params = self.synapses['stdp_params']
        
        # Finde postsynaptische Spike-Zeiten
        post_synapses = [i for i, (pre, post) in enumerate(self.synapses['connections']) 
                        if post == post_idx]
        
        post_spike_times = []
        for syn_idx in post_synapses:
            post_spike_times.extend(self.synapses['spike_times'][syn_idx])
        
        # Berechne STDP für jeden postsynaptischen Spike
        for post_time in post_spike_times:
            delta_t = spike_time - post_time  # prä vor post ist positiv
            
            if delta_t > 0:  # LTP (Long-Term Potentiation)
                weight_change = params['A_plus'] * torch.exp(-delta_t / params['tau_plus'])
            else:  # LTD (Long-Term Depression)
                weight_change = -params['A_minus'] * torch.exp(delta_t / params['tau_minus'])
            
            # Update synaptisches Gewicht
            new_weight = self.synapses['weights'][synapse_idx] + weight_change
            new_weight = torch.clamp(new_weight, params['w_min'], params['w_max'])
            self.synapses['weights'][synapse_idx] = new_weight
    
    def update_ion_concentrations(self, I_Na: torch.Tensor, I_K: torch.Tensor):
        """Aktualisiere Ionenkonzentrationen basierend auf Strömen"""
        
        # Umrechnung von Strom zu Ionenfluss
        # I = z * F * φ (φ = Ionenfluss)
        F = 96485.0  # Faraday-Konstante (C/mol)
        
        # Volumen der Neuronen (angenommen)
        cell_volume = 1e-9  # 1µm³ in cm³
        
        # Natrium-Fluss (positiver Strom = Na⁺-Einstrom)
        phi_Na = I_Na / (1.0 * F)  # Na⁺ hat Ladung +1
        delta_Na = phi_Na * self.dt / cell_volume * 1000  # in mM
        
        # Kalium-Fluss (positiver Strom = K⁺-Ausstrom)
        phi_K = I_K / (1.0 * F)  # K⁺ hat Ladung +1
        delta_K = phi_K * self.dt / cell_volume * 1000  # in mM
        
        # Aktualisiere intrazelluläre Konzentrationen
        self.ion_concentrations['Na_intra'] += delta_Na
        self.ion_concentrations['K_intra'] -= delta_K  # K⁺ verlässt Zelle
        
        # Na⁺/K⁺-Pumpe (3 Na⁺ raus, 2 K⁺ rein)
        pump_rate = self.ion_concentrations['pump_rate']
        self.ion_concentrations['Na_intra'] -= 3 * pump_rate * self.dt
        self.ion_concentrations['K_intra'] += 2 * pump_rate * self.dt
        
        # Extrazelluläre Konzentrationen (vereinfacht)
        self.ion_concentrations['Na_extra'] -= delta_Na * 0.1  # Nur kleiner Teil
        self.ion_concentrations['K_extra'] += delta_K * 0.1
    
    def record_electrode_signals(self) -> torch.Tensor:
        """Record elektrische Signale an den Elektroden"""
        
        # Neuron-Positionen (zufällig verteilt)
        neuron_positions = torch.randn(self.num_neurons, 3, device=self.device) * 100e-6
        
        # Elektroden-Signale berechnen (vereinfachtes Modell)
        signals = torch.zeros(self.num_electrodes, device=self.device)
        
        for e_idx in range(self.num_electrodes):
            electrode_pos = self.electrodes['positions'][e_idx]
            
            # Abstand zu allen Neuronen
            distances = torch.norm(neuron_positions - electrode_pos, dim=1)
            
            # Extrazelluläres Potential (vereinfacht)
            # V_ext = ρ * I / (4π * σ * r)  (punktförmige Quelle)
            sigma = 0.3  # Leitfähigkeit des Gewebes (S/m)
            
            # Ströme von aktiven Neuronen
            active_currents = torch.abs(self.membrane_potentials - 
                                       self.get_resting_potential(torch.ones(self.num_neurons, 
                                                                           device=self.device, 
                                                                           dtype=torch.bool)))
            
            # Vermeide Division durch Null
            distances = torch.clamp(distances, min=10e-6)  # Mindestabstand 10µm
            
            # Berechne Potential
            potentials = active_currents / (4 * torch.pi * sigma * distances)
            
            # Summiere Beiträge aller Neuronen
            signals[e_idx] = torch.sum(potentials)
            
            # Füge Rauschen hinzu
            noise = torch.randn(1, device=self.device) * self.electrodes['noise_levels'][e_idx]
            signals[e_idx] += noise
        
        # Speichere Recording
        self.electrodes['recordings'].append({
            'time': len(self.electrodes['recordings']) * self.dt,
            'signals': signals.clone().cpu()
        })
        
        return signals
    
    def stimulate_electrodes(self, stimulation_pattern: torch.Tensor):
        """
        Wende elektrische Stimulation über Elektroden an
        
        Args:
            stimulation_pattern: Stimulationsströme für jede Elektrode (nA)
        """
        
        if len(stimulation_pattern) != self.num_electrodes:
            raise ValueError(f"Stimulation pattern must have length {self.num_electrodes}")
        
        # Konvertiere Elektroden-Stimulation zu neuronalen Strömen
        neuron_currents = torch.zeros(self.num_neurons, device=self.device)
        
        # Neuron-Positionen
        neuron_positions = torch.randn(self.num_neurons, 3, device=self.device) * 100e-6
        
        for e_idx in range(self.num_electrodes):
            if stimulation_pattern[e_idx] != 0:
                electrode_pos = self.electrodes['positions'][e_idx]
                
                # Abstand zu allen Neuronen
                distances = torch.norm(neuron_positions - electrode_pos, dim=1)
                
                # Strom nimmt mit Abstand ab (exponentiell)
                decay = torch.exp(-distances / 50e-6)  # 50µm Decay-Konstante
                
                # Beiträge zum neuronalen Strom
                neuron_currents += stimulation_pattern[e_idx] * decay
        
        # Speichere Stimulationsmuster
        self.electrodes['stimulation_patterns'].append({
            'time': len(self.electrodes['stimulation_patterns']) * self.dt,
            'pattern': stimulation_pattern.clone().cpu(),
            'neuron_currents': neuron_currents.clone().cpu()
        })
        
        return neuron_currents
    
    def simulate_time_step(self, 
                          external_stimulation: Optional[torch.Tensor] = None,
                          record_signals: bool = True) -> Dict:
        """
        Simuliere einen kompletten Zeitschritt
        
        Args:
            external_stimulation: Externe Stimulation über Elektroden
            record_signals: Ob Elektrodensignale aufgezeichnet werden sollen
            
        Returns:
            Dictionary mit Simulationsergebnissen
        """
        
        # Wende externe Stimulation an
        if external_stimulation is not None:
            neuron_currents = self.stimulate_electrodes(external_stimulation)
        else:
            neuron_currents = torch.zeros(self.num_neurons, device=self.device)
        
        # Synaptische Inputs berechnen
        synaptic_currents = self.calculate_synaptic_currents()
        
        # Gesamtstrom = externe Stimulation + synaptische Inputs
        total_current = neuron_currents + synaptic_currents
        
        # Aktualisiere Membranpotentiale
        membrane_results = self.update_membrane_potential(total_current)
        
        # Record Elektrodensignale
        if record_signals:
            electrode_signals = self.record_electrode_signals()
        else:
            electrode_signals = None
        
        # Update Mikrotubuli-Quantenzustände (Orch-OR)
        microtubule_results = self.update_microtubule_states(membrane_results['spikes'])
        
        return {
            'membrane_potentials': membrane_results['membrane_potentials'],
            'spikes': membrane_results['spikes'],
            'ion_currents': {
                'I_Na': membrane_results['I_Na'],
                'I_K': membrane_results['I_K'],
                'I_L': membrane_results['I_L']
            },
            'total_current': total_current,
            'electrode_signals': electrode_signals,
            'microtubule_states': microtubule_results,
            'time': len(self.activity_history) * self.dt,
            'spike_count': len(self.spike_history)
        }
    
    def calculate_synaptic_currents(self) -> torch.Tensor:
        """Berechne synaptische Ströme basierend auf Spike-Historie"""
        
        synaptic_currents = torch.zeros(self.num_neurons, device=self.device)
        current_time = len(self.activity_history) * self.dt
        
        # Durchlaufe alle Synapsen
        for i, (pre, post) in enumerate(self.synapses['connections']):
            weight = self.synapses['weights'][i]
            delay = self.synapses['delays'][i]
            syn_type = self.synapses['types'][i]
            
            # Finde relevante Spikes (innerhalb der synaptischen Zeitkonstante)
            relevant_spikes = [t for t in self.synapses['spike_times'][i] 
                             if current_time - t - delay <= 20.0]  # 20ms Zeitkonstante
            
            if relevant_spikes:
                # Berechne synaptischen Strom (exponentieller Decay)
                latest_spike = max(relevant_spikes)
                time_since_spike = current_time - latest_spike - delay
                
                if time_since_spike >= 0:
                    # Exponentieller Decay
                    tau_syn = 5.0 if syn_type == 'excitatory' else 10.0  # ms
                    current_amplitude = weight * torch.exp(-time_since_spike / tau_syn)
                    
                    # Typ-spezifische Umkehrpotentiale
                    if syn_type == 'excitatory':
                        E_syn = 0.0  # mV
                    elif syn_type == 'inhibitory':
                        E_syn = -80.0  # mV
                    else:  # modulatory
                        E_syn = -20.0  # mV
                    
                    # Synaptischer Strom
                    V_post = self.membrane_potentials[post]
                    I_syn = current_amplitude * (V_post - E_syn)
                    
                    synaptic_currents[post] += I_syn
        
        return synaptic_currents
    
    def update_microtubule_states(self, spikes: torch.Tensor) -> Dict:
        """
        Update Mikrotubuli-Quantenzustände basierend auf neuronaler Aktivität
        
        Orch-OR: Neuronal spikes verursachen orchestrierte Reduktion
        """
        
        # Finde aktive Neuronen
        active_neurons = torch.where(spikes)[0]
        
        results = {
            'orchestrated_collapses': 0,
            'coherence_changes': [],
            'tubulin_updates': 0
        }
        
        if len(active_neurons) > 0:
            # Durchlaufe alle aktiven Neuronen
            for neuron_idx in active_neurons:
                # Finde zugehörige Mikrotubuli
                neuron_microtubules = [mt for mt in self.microtubules['organization']
                                     if mt['neuron_index'] == neuron_idx.item()]
                
                for mt in neuron_microtubules:
                    tubulin_indices = mt['tubulin_indices']
                    
                    # Erhöhe Kollaps-Wahrscheinlichkeit für diese Tubuline
                    # (Orchestrierte Reduktion durch neuronale Aktivität)
                    coherence_reduction = 0.1  # Stärke der Dekohärenz
                    
                    for idx in tubulin_indices:
                        # Reduziere Kohärenzzeit
                        self.microtubules['coherence_times'][idx] *= (1 - coherence_reduction)
                        
                        # Quantenzustand stören (Dekohärenz)
                        noise = torch.randn(2, dtype=torch.complex64, device=self.device) * 0.01
                        self.microtubules['states'][idx] += noise
                        self.microtubules['states'][idx] = (
                            self.microtubules['states'][idx] / 
                            torch.norm(self.microtubules['states'][idx])
                        )
                        
                        results['tubulin_updates'] += 1
                    
                    results['coherence_changes'].append({
                        'neuron': neuron_idx.item(),
                        'protofilament': mt['protofilament'],
                        'coherence_reduction': coherence_reduction,
                        'num_tubulins': len(tubulin_indices)
                    })
                    
                    # Penrose Kollaps überprüfen
                    avg_coherence = torch.mean(self.microtubules['coherence_times'][tubulin_indices])
                    if avg_coherence < 1e-5:  # Kollaps-Schwelle
                        results['orchestrated_collapses'] += 1
                        
                        # Kollabiere Quantenzustände
                        for idx in tubulin_indices:
                            # Kollaps zu einem Zustand
                            probabilities = torch.abs(self.microtubules['states'][idx])**2
                            collapsed_state = torch.multinomial(probabilities, 1).item()
                            
                            # Setze auf kollabierten Zustand
                            new_state = torch.zeros_like(self.microtubules['states'][idx])
                            new_state[collapsed_state] = 1.0
                            self.microtubules['states'][idx] = new_state
                            
                            # Setze Kohärenzzeit zurück
                            self.microtubules['coherence_times'][idx] = 1e-4
        
        return results
    
    def run_simulation(self, 
                      duration_ms: float = 1000.0,
                      stimulation_protocol: Optional[callable] = None) -> Dict:
        """
        Führe eine komplette Simulation über eine bestimmte Dauer durch
        
        Args:
            duration_ms: Simulationsdauer in Millisekunden
            stimulation_protocol: Funktion, die Stimulationsmuster generiert
            
        Returns:
            Vollständige Simulationsergebnisse
        """
        
        num_steps = int(duration_ms / self.dt)
        
        print(f"Starte Simulation für {duration_ms} ms ({num_steps} Schritte)")
        print(f"Kultur-Typ: {self.culture_type}")
        print(f"Neuronen: {self.num_neurons}")
        
        all_results = []
        
        for step in range(num_steps):
            # Generiere Stimulation (falls vorhanden)
            if stimulation_protocol is not None:
                stimulation = stimulation_protocol(step, self)
            else:
                stimulation = None
            
            # Simuliere Zeitschritt
            step_result = self.simulate_time_step(stimulation, record_signals=True)
            all_results.append(step_result)
            
            # Fortschritts-Anzeige
            if step % 1000 == 0 and step > 0:
                elapsed_ms = step * self.dt
                spikes_this_batch = sum(r['spikes'].sum().item() for r in all_results[-1000:])
                print(f"  {elapsed_ms:6.1f} ms: {spikes_this_batch} Spikes")
        
        # Analysiere Ergebnisse
        analysis = self.analyze_simulation_results(all_results)
        
        print(f"\nSimulation abgeschlossen!")
        print(f"Gesamt-Spikes: {len(self.spike_history)}")
        print(f"Durchschnittliche Feuerrate: {analysis['mean_firing_rate']:.2f} Hz")
        print(f"Synchronitäts-Index: {analysis['synchrony_index']:.3f}")
        
        return {
            'step_results': all_results,
            'spike_history': self.spike_history,
            'activity_history': self.activity_history,
            'electrode_recordings': self.electrodes['recordings'],
            'analysis': analysis,
            'culture_parameters': {
                'num_neurons': self.num_neurons,
                'culture_type': self.culture_type,
                'duration_ms': duration_ms,
                'time_step': self.dt
            }
        }
    
    def analyze_simulation_results(self, step_results: List[Dict]) -> Dict:
        """Analysiere Simulationsergebnisse"""
        
        # Extrahiere Zeitreihen
        spike_counts = [r['spikes'].sum().item() for r in step_results]
        membrane_potentials = [r['membrane_potentials'].cpu().numpy() for r in step_results]
        
        # Durchschnittliche Feuerrate
        total_spikes = sum(spike_counts)
        duration_steps = len(step_results)
        duration_ms = duration_steps * self.dt
        mean_firing_rate = total_spikes / (self.num_neurons * duration_ms / 1000.0)  # Hz
        
        # Synchronitäts-Index (Cross-Correlation)
        if len(membrane_potentials) > 1:
            # Korrelation zwischen verschiedenen Neuronen
            correlation_matrix = np.corrcoef(np.array(membrane_potentials).T)
            synchrony_index = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        else:
            synchrony_index = 0.0
        
        # Burst-Erkennung
        burst_analysis = self.detect_bursts(spike_counts)
        
        # Netzwerk-Metriken
        network_metrics = self.calculate_network_metrics()
        
        return {
            'mean_firing_rate': mean_firing_rate,
            'synchrony_index': synchrony_index,
            'total_spikes': total_spikes,
            'spike_times': [s['time'] for s in self.spike_history],
            'burst_analysis': burst_analysis,
            'network_metrics': network_metrics,
            'simulation_duration_ms': duration_ms,
            'time_resolution_ms': self.dt
        }
    
    def detect_bursts(self, spike_counts: List[int]) -> Dict:
        """Erkenne Burst-Aktivität in Spike-Zeitreihe"""
        
        spike_array = np.array(spike_counts)
        
        # Glätte Spike-Zeitreihe
        window_size = 10  # 0.1ms bei dt=0.01ms
        smoothed = np.convolve(spike_array, np.ones(window_size)/window_size, mode='same')
        
        # Finde Peaks (Bursts)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smoothed, height=np.mean(spike_array) * 2, distance=50)
        
        # Analysiere Bursts
        bursts = []
        for peak in peaks:
            burst_start = max(0, peak - 25)
            burst_end = min(len(spike_array), peak + 25)
            
            burst_spikes = sum(spike_array[burst_start:burst_end])
            burst_duration = (burst_end - burst_start) * self.dt
            
            bursts.append({
                'peak_time': peak * self.dt,
                'start_time': burst_start * self.dt,
                'end_time': burst_end * self.dt,
                'duration_ms': burst_duration,
                'total_spikes': burst_spikes,
                'peak_firing_rate': spike_array[peak] / (self.num_neurons * self.dt / 1000.0)
            })
        
        return {
            'num_bursts': len(bursts),
            'bursts': bursts,
            'mean_burst_duration': np.mean([b['duration_ms'] for b in bursts]) if bursts else 0,
            'mean_burst_frequency': len(bursts) / (len(spike_array) * self.dt / 1000.0) if len(spike_array) > 0 else 0
        }
    
    def calculate_network_metrics(self) -> Dict:
        """Berechne Netzwerk-Metriken aus Konnektom"""
        
        # Erstelle Adjazenzmatrix
        adj_matrix = torch.zeros(self.num_neurons, self.num_neurons, device=self.device)
        
        for (pre, post), weight in zip(self.synapses['connections'], self.synapses['weights']):
            adj_matrix[pre, post] = weight
        
        # Grad-Verteilung
        in_degrees = torch.sum(adj_matrix != 0, dim=0).float()
        out_degrees = torch.sum(adj_matrix != 0, dim=1).float()
        
        # Clustering-Koeffizient (vereinfacht)
        clustering_sum = 0
        for i in range(self.num_neurons):
            neighbors = torch.where(adj_matrix[i] != 0)[0]
            num_neighbors = len(neighbors)
            
            if num_neighbors > 1:
                # Zähle Verbindungen zwischen Nachbarn
                neighbor_connections = 0
                for j in range(num_neighbors):
                    for k in range(j + 1, num_neighbors):
                        if adj_matrix[neighbors[j], neighbors[k]] != 0:
                            neighbor_connections += 1
                
                clustering_sum += neighbor_connections / (num_neighbors * (num_neighbors - 1) / 2)
        
        avg_clustering = clustering_sum / self.num_neurons if self.num_neurons > 0 else 0
        
        # Small-Worldness (vereinfacht)
        # Annahme: Zufallsnetzwerk hätte ähnliche Grad-Verteilung aber niedrigeren Clustering
        
        return {
            'mean_in_degree': torch.mean(in_degrees).item(),
            'mean_out_degree': torch.mean(out_degrees).item(),
            'clustering_coefficient': avg_clustering,
            'connection_density': len(self.synapses['connections']) / (self.num_neurons ** 2),
            'excitatory_inhibitory_ratio': (
                sum(1 for t in self.synapses['types'] if t == 'excitatory') /
                max(1, sum(1 for t in self.synapses['types'] if t == 'inhibitory'))
            )
        }

# Beispiel-Nutzung
if __name__ == "__main__":
    # Erstelle virtuelle neuronale Kultur
    print("Creating virtual neuronal culture...")
    
    culture = VirtualNeuronalCulture(
        num_neurons=500,
        num_electrodes=32,
        culture_type="hippocampal"
    )
    
    # Definiere Stimulations-Protokoll
    def stimulation_protocol(step, culture):
        """Generiere periodische Stimulation"""
        
        # Periodische Stimulation alle 100ms
        if step % 10000 == 0:  # 100ms bei dt=0.01ms
            # Stimuliere zufällige Elektroden
            stimulation = torch.zeros(culture.num_electrodes, device=culture.device)
            for _ in range(5):  # 5 Elektroden gleichzeitig
                electrode = random.randint(0, culture.num_electrodes - 1)
                stimulation[electrode] = random.uniform(-100, 100)  # nA
            
            return stimulation
        
        return None
    
    # Führe Simulation durch
    results = culture.run_simulation(
        duration_ms=1000.0,  # 1 Sekunde
        stimulation_protocol=stimulation_protocol
    )
    
    print(f"\nSimulation Results:")
    print(f"  Total spikes: {results['analysis']['total_spikes']}")
    print(f"  Mean firing rate: {results['analysis']['mean_firing_rate']:.2f} Hz")
    print(f"  Synchrony index: {results['analysis']['synchrony_index']:.3f}")
    print(f"  Number of bursts: {results['analysis']['burst_analysis']['num_bursts']}")
    
    # Speichere Ergebnisse
    import pickle
    with open('virtual_culture_simulation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved to virtual_culture_simulation.pkl")
