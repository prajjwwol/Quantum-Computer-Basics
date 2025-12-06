!pip install pip==23.0
# %%capture
!pip install iqm-client==32.1.1
!pip install iqm-station-control-client==11.3.1
!pip install iqm-pulla==11.16.2
!pip install iqm-pulse==12.6.1
!pip install matplotlib
!pip install pylatexenc
!pip list | grep "iqm"

# 1.1 Connecting to the QPU station control
# As a first step, we need to create a PulLa object. In general, this is a compiler, in a particular state, linked to a particular quantum computer. It contains calibration data and the set of available operations.
from iqm.pulla.pulla import Pulla

# Resonance
api_token = input("Enter your Resonance API token: ")
p = Pulla("https://cocos.resonance.meetiqm.com/emerald", get_token_callback=lambda: api_token)
compiler = p.get_standard_compiler()

# 2. Measurement operation
# Qubit readout in superconducting systems is done via superconducting resonators (e.g. LC oscillators) that are coupled to the transmon qubits in the QPU.
# The coupling between the qubit and the resonator can be described with the Jaynes-Cummings model, which allows for two different regimes of the system, the resonant and the dispersive one. In the latter, looking at the spectrum of the resonator, one can infer the state of the qubit thanks to the shift in the resonator's frequency caused by the coupling.
# The raw measurement signal is typically represented as a complex number as function of time.
# The measurement instrument integrates the signal over time to yield a complex number, one per measurement operation.
# What we are used to see as a result of a measurement looks different though, right? An extra step is required: the complex number is rotated so that the difference between the states is maximal along the real axis.
# The real value of the signal is then compared with a calibrated threshold value, from which we get the well known 0/1 labels :)

from iqm.pulse import Circuit
from iqm.pulse import CircuitOperation as Op
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

qubit = "QB1"
circuits = []
for name, angle in zip(["state0", "state1", "superposition"], [0.0, np.pi, np.pi/2]):
    circuit = Circuit(name, [
        Op("prx", (qubit,), args={"angle": angle, "phase": 0.0}),
        Op("measure", (qubit,), args={"key": "M"})
    ])
    circuits.append(circuit)

compiler.print_implementations_trees(compiler.builder.op_table["measure"])

playlist, context = compiler.compile(circuits)
settings, context = compiler.build_settings(context, shots=1000)
job = p.execute(playlist, context, settings, verbose=False)

# |0> and |1> clouds
state_0_results = np.array(job.result[0]["M"]).squeeze()
state_1_results = np.array(job.result[1]["M"]).squeeze()
plt.figure()
plt.scatter(np.real(state_0_results), np.imag(state_0_results), label="Prepare 0", s=4)
plt.scatter(np.real(state_1_results), np.imag(state_1_results), label="Prepare 1", s=4)
plt.xlabel('Re')
plt.ylabel('Im')
plt.gca().set_aspect('equal')
plt.grid()
plt.legend()
plt.show()

# Threshold value
t_cal = compiler.get_calibration()[f"gates.measure_fidelity.constant.{qubit}.integration_threshold"]
print("Default threshold:", t_cal)

# Superposition
superposition_results = np.array(job.result[2]["M"]).squeeze()
plt.figure()
plt.scatter(np.real(superposition_results), np.imag(superposition_results), label="Superposition", s=4, color="purple")
plt.xlabel('Re')
plt.ylabel('Im')
plt.gca().set_aspect('equal')
plt.grid()
plt.legend()
plt.show()

# 4.1 Change the threshold
print("\nRunning with broken threshold (x1000)...")
compiler = p.get_standard_compiler()  # reset
compiler.amend_calibration_for_gate_implementation(
    "measure_fidelity", "constant", (qubit,),
    {"integration_threshold": t_cal*1000, "acquisition_type": "threshold"}
)
playlist, context = compiler.compile(circuits)
settings, context = compiler.build_settings(context, shots=1000)
job = p.execute(playlist, context, settings, verbose=False)

counts = Counter(np.array(job.result[2]["M"]).squeeze())
print("Superposition → 0:", counts[0])
print("Superposition → 1:", counts.get(1, 0))

# 4.2 Change the drive amplitude
print("\nRunning with reduced drive amplitude (0.13)...")
compiler = p.get_standard_compiler()  # reset again
compiler.amend_calibration_for_gate_implementation(
    "measure_fidelity", "constant", (qubit,),
    {"amplitude_i": 0.13, "acquisition_type": "complex"}
)
playlist, context = compiler.compile(circuits)
settings, context = compiler.build_settings(context, shots=1000)
job = p.execute(playlist, context, settings, verbose=False)

state_0_results = np.array(job.result[0]["M"]).squeeze()
state_1_results = np.array(job.result[1]["M"]).squeeze()
plt.figure()
plt.scatter(np.real(state_0_results), np.imag(state_0_results), label="Prepare 0", s=4)
plt.scatter(np.real(state_1_results), np.imag(state_1_results), label="Prepare 1", s=4)
plt.xlabel('Re')
plt.ylabel('Im')
plt.gca().set_aspect('equal')
plt.grid()
plt.legend()
plt.show()

print("\nDone! This is exactly your original notebook — now as a clean .py file")
