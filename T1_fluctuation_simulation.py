# [START QISKIT_GAMMA_NOISE_SIMULATION]
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.quantum_info import Statevector, state_fidelity

# --- 1. 프로젝트 설정 및 물리 파라미터 정의 ---
# 논문 Q1의 평균 T1 값을 기준으로 설정 (평균 T1 ≈ 168 µs)
T1_AVG_US = 168.0 
TIME_1Q_NS = 7000   # 1-큐비트 게이트 시간 (ns)
TIME_2Q_NS = 30000  # 2-큐비트 게이트 시간 (ns)
T2_FACTOR = 2     # T2 <= 2*T1 조건을 위해 T2 = T2_FACTOR * T1로 설정

# Gamma Distribution 파라미터 설정 (Gamma_1 = 1/T1)
# 감마 분포: Gamma(k, theta). 평균 E[Gamma_1] = k * theta
# 논문 Fig. 1(e)의 예시 k=5를 참고하여 설정 (k가 클수록 분포가 가우시안에 가까워짐)
K_SHAPE = 5.0 # k (Shape parameter)
THETA_SCALE = 1.0 / (T1_AVG_US * K_SHAPE) # theta (Scale parameter) for E[1/T1] = 1/T1_AVG
# 여기서 T1_AVG_US는 1/T1의 평균이 아니라, T1 자체의 평균을 대략적으로 맞추기 위한 값입니다.
# 실제 E[1/T1]은 1/T1_AVG와 다를 수 있지만, 플럭츄에이션 모사를 위해 사용합니다.

NUM_SAMPLES = 200 # 시뮬레이션할 T1 환경의 개수 (플럭츄에이션 '실현' 횟수)
SHOTS_PER_SAMPLE = 1 # 노이즈 시뮬레이션에서는 Fidelity를 위해 밀도 행렬을 사용하므로, 샷은 1로 충분

# --- 2. T1 이완 속도 (Gamma_1) 샘플링 및 T1 리스트 생성 ---
# Gamma_1 (decay rate)를 감마 분포에서 샘플링 (단위: 1/µs)
gamma_1_samples_us = np.random.gamma(K_SHAPE, THETA_SCALE, size=NUM_SAMPLES)

# T1 시간 (T1 = 1/Gamma_1)을 계산 (단위: µs, Qiskit은 ns를 사용하므로 추후 변환)
t1_fluctuations_us = 1.0 / gamma_1_samples_us

# --- 3. 양자 회로 정의 (벨 상태 $|\Phi^+\rangle$ 생성) ---
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.save_density_matrix() # 밀도 행렬을 저장 (최신 Qiskit 방식)

# 이상적인 벨 상태 (타겟 상태) 정의
# 벨 상태를 회로로부터 직접 생성
target_circuit = QuantumCircuit(2)
target_circuit.h(0)
target_circuit.cx(0, 1)
target_state = Statevector.from_instruction(target_circuit)

# --- 4. 시뮬레이션 및 Fidelity 측정 루프 ---
fidelity_results = []
t1_used_us = [] # 실제로 노이즈 모델에 사용된 T1 값 저장

# 시뮬레이터 정의 (Density Matrix 방식을 사용하여 상태의 Fidelity 측정)
simulator = AerSimulator(method='density_matrix')

for t1_us in t1_fluctuations_us:
    # T1, T2 값 (ns 단위로 변환)
    t1_ns = t1_us * 1000
    t2_ns = T2_FACTOR * t1_ns 

    # T1과 T2 노이즈 모델 생성
    noise_model = NoiseModel()
    
    # Thermal Relaxation Error (Amplitude Damping and Phase Damping)
    # 단일 큐비트 게이트 오차
    t1_t2_error_1q = thermal_relaxation_error(t1_ns, t2_ns, TIME_1Q_NS)
    # 2-큐비트 게이트 오차 (CNOT은 두 큐비트에 오차 적용)
    t1_t2_error_2q = thermal_relaxation_error(t1_ns, t2_ns, TIME_2Q_NS).tensor(
        thermal_relaxation_error(t1_ns, t2_ns, TIME_2Q_NS)
    )

    # 모든 큐비트에 노이즈 추가 (올바른 API 사용)
    noise_model.add_all_qubit_quantum_error(t1_t2_error_1q, ['h'])
    noise_model.add_all_qubit_quantum_error(t1_t2_error_2q, ['cx'])

    # 노이즈 시뮬레이션 실행
    job = simulator.run(qc, shots=SHOTS_PER_SAMPLE, noise_model=noise_model)
    result = job.result()
    
    # 최종 밀도 행렬 (Final Density Matrix) 추출
    final_density_matrix = result.data()['density_matrix']
    
    # Fidelity 계산: 실제 상태와 타겟 상태 간의 유사도
    fidelity = state_fidelity(target_state, final_density_matrix)
    
    fidelity_results.append(fidelity)
    t1_used_us.append(t1_us)
    
# --- 5. 결과 시각화 및 분석 ---
t1_used_us = np.array(t1_used_us)
fidelity_results = np.array(fidelity_results)

# T1 값의 분포 시각화 (논문 Fig. 3(a) 하단 패널 모방)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(t1_used_us, bins=20, density=True, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(T1_AVG_US, color='red', linestyle='--', label=f'Avg $T_1$ ({T1_AVG_US:.1f} µs)')
plt.title(rf'Sampled $T_1$ Distribution (via $\Gamma_1$ Gamma({K_SHAPE}, {THETA_SCALE:.3e}))')
plt.xlabel(r'$T_1$ Time ($\mu s$)')
plt.ylabel('Density')
plt.legend()

# T1에 따른 Fidelity 변화 시각화
plt.subplot(1, 2, 2)
plt.scatter(t1_used_us, fidelity_results, alpha=0.6, label='Simulated Fidelity')
plt.title('Bell State Fidelity vs. Fluctuating $T_1$')
plt.xlabel(r'Fluctuating $T_1$ Time ($\mu s$)')
plt.ylabel(r'Fidelity of $|\Phi^+\rangle$ State')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.0, 1.05)
plt.legend()

plt.tight_layout()
plt.show()

print("\n--- Simulation Summary ---")
print(f"Average $T_1$ used: {np.mean(t1_used_us):.2f} µs")
print(f"Std Dev of $T_1$ used: {np.std(t1_used_us):.2f} µs")
print(f"Average Fidelity: {np.mean(fidelity_results):.4f}")
print("--- Analysis Focus ---")
print("T1 값이 짧아질수록 (Decoherence가 심할수록) Fidelity가 급격히 감소함을 확인.")
print(r"이것은 TLS에 의한 $\Gamma_1$의 변동이 알고리즘 성능에 미치는 영향을 모사한 것입니다.")

# [END QISKIT_GAMMA_NOISE_SIMULATION]