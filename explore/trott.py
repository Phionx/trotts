"""
Trott Functions
"""

# suppress warnings
from typing import List
import copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Importing standard Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter, Instruction
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.opflow import Zero, One, I, X, Y, Z
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Prepare Circuits
# ================================================================

def gen_trott_gate():
    t = Parameter('t') # parameterize variable t
    
    # XX(t)
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * t, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()
    
    # YY(t)
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')

    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * t, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()

        
    # ZZ(t)
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * t, 1)
    ZZ_qc.cnot(0,1)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()
    
    num_qubits = 3

    Trott_qr = QuantumRegister(num_qubits)
    Trott_qc = QuantumCircuit(Trott_qr, name='Trot')

    for i in range(0, num_qubits - 1):
        Trott_qc.append(ZZ, [Trott_qr[i], Trott_qr[i+1]])
        Trott_qc.append(YY, [Trott_qr[i], Trott_qr[i+1]])
        Trott_qc.append(XX, [Trott_qr[i], Trott_qr[i+1]])

    # Convert custom quantum circuit into a gate
    Trott_gate = Trott_qc.to_instruction()
    return Trott_gate

def gen_st_qcs(trott_gate: Instruction, trotter_steps: int):
    """
    Args:
        n (int): number of trotter steps
    """
    
    t = trott_gate.params[0] # assuming only t param
    
    target_time = np.pi
    
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)
    
    qc.x([3,5]) # prepare init state |q5q3q1> = |110>
    
    for _ in range(trotter_steps):
        qc.append(trott_gate, [qr[1], qr[3], qr[5]])
    
    qc = qc.bind_parameters({t: target_time/trotter_steps})
    
    st_qcs = state_tomography_circuits(qc, [qr[1], qr[3], qr[5]])
    
    return st_qcs

def gen_st_qcs_range(trott_gate: Instruction, trott_steps_range: List[int]):
    qcs = {}
    for trott_steps_val in trott_steps_range:
        qcs[trott_steps_val] = gen_st_qcs(trott_gate, trott_steps_val)
    return qcs

def gen_target():
    g = qt.basis(2,0)
    e = qt.basis(2,1)

    # fidelity: the reconstructed state has the (flipped) ordering |q5q3q1> 
    target_state_qt = qt.tensor(e,e,g)
    target_state_qt = qt.ket2dm(target_state_qt)
    target_state = target_state_qt.full()


    # parity: "XYZ" corresponds to X measurement on q1, Y measurement on q3, and Z measurement on q5
    target_state_parity_qt = qt.ket2dm( qt.tensor(g,e,e))
    target_state_parity = target_state_parity_qt.full()
    
    pauli = {"X":qt.sigmax(),"Y":qt.sigmay(),"Z":qt.sigmaz(),"I":qt.identity(2)}
    target_parity = {}
    for k1, p1 in pauli.items():
        for k2, p2 in pauli.items():
            for k3, p3 in pauli.items():
                pauli_string = k1+k2+k3
                if pauli_string == "III":
                    continue
                op = qt.tensor(p1,p2,p3)
                meas = (target_state_parity_qt*op).tr()
                target_parity[pauli_string] = meas
                
    return target_state, target_parity


# Tomography
# ================================================================

def state_tomo(result, st_qcs):
    """
    Compute the state tomography based on the st_qcs quantum circuits 
    and the results from those ciricuits
    """
    
    # The expected final state; necessary to determine state tomography fidelity
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    return rho_fit


# Generate Results
# ================================================================


def gen_jobs_single(st_qcs, backend = None, shots=8192, reps=8):
    backend = QasmSimulator() if backend is None else backend

    # create jobs
    jobs = []
    for _ in range(reps):
        # execute
        job = execute(st_qcs, backend, shots=shots)
        print('Job ID', job.job_id())
        jobs.append(job)
    return jobs
        

def gen_job_monitors_single(jobs):
    # monitor jobs
    for job in jobs:
        job_monitor(job)
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass

def extract_results_single(jobs, st_qcs):
    # calculate fids
    rhos = []
    raw_results = []
    for job in jobs:
        raw_results.append(job.result())
        rho = state_tomo(raw_results[-1], st_qcs)
        rhos.append(rho)
    return rhos, raw_results

def gen_result_single(st_qcs, backend = None, shots=8192, reps=8):
    jobs = gen_jobs_single(st_qcs, backend=backend, shots=shots, reps=reps)
    gen_job_monitors_single(jobs)
    return extract_results_single(jobs, st_qcs)

def gen_results(qcs, backend = None, results = None, label="data/sim", filename=None, shots=8192, reps=8):
    """
    This function submits, monitors, and stores all jobs for a single trott_step size 
    and then moves on to the next trott_step size. This may be preferred for jobs run
    on simulators.
    """
    now = datetime.now()
    filename = filename if filename is not None else label + "_results_" + now.strftime("%Y%m%d__%H%M%S") + ".npy"
    
    
    backend = QasmSimulator() if backend is None else backend
    results = results if results is not None else {"properties": {"backend": backend}, "data":{}}

    for num_trott_steps, st_qcs in tqdm(qcs.items()):
        print("="*20)
        if num_trott_steps in results["data"]:
            print(f"Result already stored for trott_steps = {num_trott_steps}")
            continue

        print(f"Running with trott_steps = {num_trott_steps}")
        
        results["data"][num_trott_steps] = {}
        results["data"][num_trott_steps]["rhos"], results["data"][num_trott_steps]["raw_data"] = gen_result_single(st_qcs, backend=backend, shots=shots, reps=reps)
    
        np.save(filename, results)
    return results


def gen_qpu_jobs(qcs, backend = None, label="data/qpu", shots=8192, reps=8):
    """
    This function submits jobs.
    This may be preferred for jobs run on real qpus.
    """
    now = datetime.now()
    filename = filename if filename is not None else label + "_jobs_" + now.strftime("%Y%m%d__%H%M%S") + ".npy"
    
    # Generate and submit all jobs first
    jobs = {}
    for num_trott_steps, st_qcs in tqdm(qcs.items()):
        print("="*20)
        if num_trott_steps in results["data"]:
            print(f"Result already stored for trott_steps = {num_trott_steps}")
            continue

        print(f"Submitting jobs with trott_steps = {num_trott_steps}")
        jobs[num_trott_steps] = gen_jobs_single(st_qcs, backend = backend, shots=shots, reps=reps)
        np.save(filename, jobs)
        
    return jobs

def gen_qpu_results(qcs, jobs, backend, results = None, label="data/qpu", filename=None, shots=8192, reps=8):
    """
    This function monitors and stores results from existing jobs. 
    This may be preferred for jobs run on real qpus.
    """

    now = datetime.now()
    filename = filename if filename is not None else label + "_results_" + now.strftime("%Y%m%d__%H%M%S") + ".npy"
    results = results if results is not None else {"properties": {"backend": backend}, "data":{}}
    
    # Then, monitor jobs and store them as they complete
    for num_trott_steps, job_list in tqdm(jobs.items()):
        print("="*20)
        print(f"Monitoring jobs with trott_steps = {num_trott_steps}")
        gen_job_monitors_single(job_list)
        print(f"Storing results for jobs with trott_steps = {num_trott_steps}")    
        results["data"][num_trott_steps] = {}
        results["data"][num_trott_steps]["rhos"], results["data"][num_trott_steps]["raw_data"] = extract_results_single(job_list, qcs[num_trott_steps])
        
        np.save(filename, results)
        
    return results

# Analysis
# ================================================================

ACTIVE_LIST = [(1,1,1), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1)] # [ZXY, ZXI, ZIY, IXY, ZII, IXI, IIY]

def compare_Z_parity(res_analysis, n=None):
    _, target_parity = gen_target()
    
    ps = ["ZZZ", "ZZI", "ZIZ", "IZZ", "ZII", "IZI", "IIZ"]
    n = n if n is not None else max(list(res_analysis["data"].keys()))
    print(" \t" + "Expected", f"| n={n}")
    for p in ps:
        print("<"+str(p)+">" +"\t" + str(target_parity[p]) + "\t   " + "{:.3f}".format(res_analysis["data"][n]["parity"][p]))#, res_analysis["data"][n]["parsed_data"][p])

def extract_key(key):
    # e.g. "('Z', 'Z', 'X')" -> ZZX
    return key[2] + key[7] + key[12]

def add_dicts(a,b):
    c = {}
    keys = set(list(a.keys()) + list(b.keys())) # union of keys
    for key in keys:
        c[key] = a.get(key,0) + b.get(key, 0)
    return c

def calc_parity(pauli_string, readout_string, active_spots):
    """
    Args:
        b (str): e.g. '0x6'
        active_spots (List[int]): e.g. (1, 1, 0)
    """
    n = len(pauli_string)
    
    adjusted_pauli_string = ""
    for i in range(n):
        letter = pauli_string[i]
        adjusted_pauli_string += letter if active_spots[i] else "I"
        
    
    b = list(format(int(readout_string[2:]), '#05b')[2:]) # e.g. "0x6" -> ["1", "1", "0"]
    b = b[::-1] # ["1", "1", "0"] -> ["0", "1", "1"]
    v = np.array([1-int(val)*2 for val in b]) #  ["0", "1", "1"] ->  [1, -1, -1]
    active = v*np.array(active_spots) # [1, -1, -1] * [1, 1, 0] -> [1, -1, 0]
    p = np.prod(active[active!=0]) # [1,-1] -> (1)*(-1) = -1
    y = int((1-p)/2) #map: 1,-1 -> 0,1
    return adjusted_pauli_string, y

def run_analysis(results):
    results = copy.deepcopy(results)
    target_state, target_parity = gen_target()
    
    # data map
    num_qubits = 3
    parsed_data = {} # key: e.g. "XYZ", "XYI", .. | val: for each parity measurement (e.g. <XYI>) we store [counts of 1, counts of -1], e.g. [12345, 950] 
    for num_trott_steps, result in results["data"].items():
        # data_map = {}
        reps = len(result["raw_data"])
        for i in range(3**num_qubits):  # loop over pauli strings (i.e. different tomography circuits)
            counts = {} # for each pauli string, we store total counts added together from each rep, e.g. {'0x6': 4014, '0x2': 4178}
            pauli_string = extract_key(result["raw_data"][0].results[i].header.name)
            for r in range(reps): # loop over reps
                counts = add_dicts(counts, result["raw_data"][r].results[i].data.counts) # adding counts together
            # data_map[pauli_string] = counts
            
            for active_spots in ACTIVE_LIST: # Loops through all possible parity measurements, e.g. [ZXY, ZXI, ZIY, IXY, ZII, IXI, IIY]
                for readout_string, count in counts.items(): # loops through all readout values, e.g. '0x6', '0x2'
                    adjusted_pauli_string, parity_meas = calc_parity(pauli_string, readout_string, active_spots) # ("ZXY", "0x6", (1,1,0)) -> "ZXI", 1 corresponds to <ZXI> = -1 measurement
                    # if adjusted_pauli_string == "IIZ":
                    #     print(pauli_string, readout_string, active_spots, adjusted_pauli_string, parity_meas, count)
                    if adjusted_pauli_string not in parsed_data:
                        parsed_data[adjusted_pauli_string] = [0,0] # [counts of 1, counts of -1]
                    parsed_data[adjusted_pauli_string][parity_meas] += count

        # result["data_map"] = data_map
        result["parsed_data"] = parsed_data
        
        parity = {} # key: e.g. "XYZ", "XYI", .. | val: for each parity measurement we store the expectation value (e.g. <XYI>)
        for parity_string, count in parsed_data.items():
            norm = np.sum(count)
            parity[parity_string] = (1)*count[0]/norm + (-1)*count[1]/norm # (1) * (counts of 1) + (-1)*(counts of -1) = <ZXY>
        
        result["parity"] = parity
        
    
    for num_trott_steps, result in results["data"].items():
        # calculate fids for each rep
        fids = []
        for rho in result["rhos"]:
            fids.append(state_fidelity(rho, target_state))
        fids = np.array(fids)
        result["infids"] = 1 - fids
        
        # calculate fid for avg rho
        result["avg_rho"] = np.mean(result["rhos"], axis=0)
        result["avg_infid"] =  1 - state_fidelity(result["avg_rho"], target_state)
        
        # calculate distance from |110><110| matrix element
        result["avg_element_dist"] = np.abs(target_state  - result["avg_rho"])
        
        result["avg_l1_dist"] = np.linalg.norm(target_state  - result["avg_rho"], ord=1)

    return results



# Plotting
# ================================================================

def plot_metric(results, metric_func=None, plot_label="Infidelity", plot_log=True, axs=None, legend_label=None, fontsize=10, ncol=1, legend_fontsize=6):
    metric_func = metric_func if metric_func is not None else lambda res: res["avg_infid"]
    steps = []
    metric = []
    for num_trott_steps, result in results["data"].items():
        steps.append(num_trott_steps)
        metric.append(metric_func(result))
    
    steps = np.array(steps)
    metric = np.array(metric)
    
    if axs is None:
        fig, axs = plt.subplots(2 if plot_log else 1,2, figsize=(8,6 if plot_log else 3), dpi=200, squeeze=False)
    
    ax = axs[0][0]
    ax.plot(1/steps, metric, label=legend_label)
    ax.set_xlabel("1/(# of Trotterization Steps)", fontsize=fontsize)
    ax.set_ylabel(plot_label, fontsize=fontsize)
    if legend_label is not None:
        ax.legend(fontsize=legend_fontsize, ncol=ncol)
    
    ax = axs[0][1]
    ax.plot(steps, metric, label=legend_label)
    ax.set_xlabel("(# of Trotterization Steps)", fontsize=fontsize)
    ax.set_ylabel(plot_label, fontsize=fontsize)
    if legend_label is not None:
        ax.legend(fontsize=legend_fontsize, ncol=ncol)
    
    if plot_log:
        ax = axs[1][0]
        ax.plot(1/steps, np.log(metric), label=legend_label)
        ax.set_xlabel("1/(# of Trotterization Steps)", fontsize=fontsize)
        ax.set_ylabel(f"log({plot_label})", fontsize=fontsize)
        if legend_label is not None:
            ax.legend(fontsize=legend_fontsize, ncol=ncol)

        ax = axs[1][1]
        ax.plot(steps, np.log(metric), label=legend_label)
        ax.set_xlabel("(# of Trotterization Steps)", fontsize=fontsize)
        ax.set_ylabel(f"log({plot_label})", fontsize=fontsize)
        if legend_label is not None:
            ax.legend(fontsize=legend_fontsize, ncol=ncol)
    
    fig = plt.gcf()
    fig.suptitle(f"{plot_label} vs. Trotterization Step #", fontsize=fontsize)
    
    fig.tight_layout()
    
    
    return axs

def plot_fidelities(results):
    return plot_metric(results, metric_func= lambda res: res["avg_infid"], plot_label="Infidelity")

def plot_element_dist(results, row=6, col=6):
    return plot_metric(results, metric_func= lambda res: res["avg_element_dist"][row][col], plot_label=f"Element Dist. ({row}, {col})")

def plot_l1_dist(results):
    return plot_metric(results, metric_func= lambda res: res["avg_l1_dist"], plot_label=f"L1 Dist.")

def plot_parity(results, parity_strings=None, legend=False):
    parity_strings = list(list(results["data"].values())[0]["parity"].keys()) if parity_strings is None else parity_strings
    
    axs = None
    for parity_string in parity_strings:
        if legend:
            axs = plot_metric(results, metric_func= lambda res: res["parity"][parity_string], plot_label=f"Measured <Pauli String>", plot_log=False, axs=axs, fontsize=6, ncol=3, legend_label=f"<{parity_string}>", legend_fontsize=4)
        else:
            axs = plot_metric(results, metric_func= lambda res: res["parity"][parity_string], plot_label=f"Measured <Pauli String>", plot_log=False, axs=axs, fontsize=6)
    
    return axs

def plot_parity_dist(results, parity_strings=None, legend=False):
    _, target_parity = gen_target()
    
    parity_strings = list(list(results["data"].values())[0]["parity"].keys()) if parity_strings is None else parity_strings
    
    axs = None
    for parity_string in parity_strings:
        if legend:
            axs = plot_metric(results, metric_func= lambda res: np.abs(target_parity[parity_string] - res["parity"][parity_string]), plot_label=f"|(Expected <Pauli String>) - (Measured <Pauli String>)|", plot_log=False, axs=axs, fontsize=6, ncol=3, legend_label=f"<{parity_string}>", legend_fontsize=4)
        else:
            axs = plot_metric(results, metric_func= lambda res: np.abs(target_parity[parity_string] - res["parity"][parity_string]), plot_label=f"|(Expected <Pauli String>) - (Measured <Pauli String>)|", plot_log=False, axs=axs, fontsize=6)
    
    return axs