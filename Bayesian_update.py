import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmam, mesolve, expect, qeye, ket2dm
from scipy.special import gamma

class BayesianT1Estimator:
    def __init__(self, k_init, theta_init, alpha=0.0, beta=0.0):
        """
        논문의 Eq. (3) Gamma Distribution 초기화
        k: Shape parameter
        theta: Scale parameter (논문에서는 theta가 시간 차원을 가짐)
        alpha: P(0|1) - 1 상태인데 0으로 읽을 확률 (Readout Error)
        beta: P(1|0) - 0 상태인데 1로 읽을 확률 (Readout Error)
        """
        self.k = k_init
        self.theta = theta_init
        self.alpha = alpha
        self.beta = beta
        self.history = {'T1_est': [], 'k': [], 'theta': []}

    def _f_m(self, m, k, theta, tau):
        """
        논문의 Eq. (6a), (6b) 구현
        m: 측정 결과 (0 or 1)
        """
        # Eq (6a, 6b) 공통 항 계산
        term_k = (theta / (theta + tau)) ** k
        term_k1 = (theta / (theta + tau)) ** (k + 1)
        
        numerator_base = self.beta + (1 - self.alpha - self.beta) * term_k1
        denominator_base = self.beta + (1 - self.alpha - self.beta) * term_k
        
        # m=0 인 경우 (Ground state)
        if m == 0:
            # 논문의 식은 P(1) 기준이므로, P(0) = 1 - P(1) 형태로 변환 필요하거나
            # 논문 Eq (6a) 직접 사용:
            # f0는 분자/분모가 (1-beta - ...) 형태임
            num = 1 - self.beta - (1 - self.alpha - self.beta) * term_k1
            den = 1 - self.beta - (1 - self.alpha - self.beta) * term_k
            return (k / theta) * (num / den)
            
        # m=1 인 경우 (Excited state)
        elif m == 1:
            return (k / theta) * (numerator_base / denominator_base)

    def update(self, m, tau):
        """
        논문의 Eq. (5a), (5b)를 이용한 베이지안 업데이트 (Moment Matching)
        """
        # 현재 파라미터로 f_m 계산
        f_k = self._f_m(m, self.k, self.theta, tau)
        f_k_plus_1 = self._f_m(m, self.k + 1, self.theta, tau)

        # 수치적 안정성 보정
        eps = 1e-12
        if f_k is None or np.isnan(f_k) or np.isinf(f_k):
            f_k = eps
        if f_k_plus_1 is None or np.isnan(f_k_plus_1) or np.isinf(f_k_plus_1):
            f_k_plus_1 = eps
        f_k = max(f_k, eps)
        f_k_plus_1 = max(f_k_plus_1, eps)
        
        # Eq (5b): Update k
        # k_{i+1}^-1 = ...
        inv_k_new = (f_k_plus_1 / f_k) - 1
        # inv_k_new가 너무 작거나 비양수이면 폭주 방지
        if not np.isfinite(inv_k_new) or inv_k_new <= eps:
            inv_k_new = eps
        k_new = 1.0 / inv_k_new
        
        # Eq (5a): Update theta (논문 표기 g^-1)
        # g_{i+1}^-1 = ... (논문에서 g는 theta에 해당)
        inv_theta_new = f_k_plus_1 - f_k
        # theta는 양수여야 하므로 inv_theta_new는 양수로 클램프
        if not np.isfinite(inv_theta_new) or inv_theta_new <= eps:
            inv_theta_new = eps
        theta_new = 1.0 / inv_theta_new
        
        # 파라미터 갱신
        self.k = k_new
        self.theta = theta_new
        
        # 추정치 저장 (T1_hat = theta / k)
        # 참고: Gamma distribution mean E[Gamma_1] = k/theta
        # T1 = 1/Gamma_1 이므로 E[T1] ~ theta/k (논문 Eq 4 근처 설명 참조)
        self.history['T1_est'].append(self.theta / self.k)
        self.history['k'].append(self.k)
        self.history['theta'].append(self.theta)

    def get_adaptive_tau(self, c=1.0):
        """
        논문의 Eq. (4): 다음 waiting time 결정
        tau_{i+1} = c * T1_hat
        """
        t1_est = self.theta / self.k
        return c * t1_est

# --- 1. 실험 설정 (QuTiP Simulation) ---
true_T1 = 200.0  # 단위: us (True value)
num_shots = 50   # 총 50번의 single shot
c_adaptive = 1.0 # Adaptive time constant

# SPAM Error 설정 (논문 참조)
alpha = 0.11 # P(0|1) error
beta = 0.14  # P(1|0) error

# 초기 Prior 설정 (Gamma dist: k=3, theta=450us - 논문값 참조)
estimator = BayesianT1Estimator(k_init=3.0, theta_init=450.0, alpha=alpha, beta=beta)

# QuTiP 연산자 정의
psi_excited = basis(2, 0) # Excited state (QuTiP에서 0이 excited - sigmam()이 작용할 수 있도록)
rho_excited = ket2dm(psi_excited)  # 초기 상태를 밀도 행렬로 사용
# sigmam()이 |excited⟩ → |ground⟩ 전이를 일으키도록 basis(2,0)을 excited로 정의
# 붕괴 연산자: sqrt(1/T1) * sigma_minus
a = sigmam() 
H = 0 * qeye(2)  # Hamiltonian (Free evolution) - zero Hamiltonian as Qobj
gamma = 1.0 / true_T1  # Decay rate (1/us)
c_ops = [np.sqrt(gamma) * a]  # Collapse operator

results_tau = []
results_m = []

print(f"True T1: {true_T1} us")
print(f"Decay rate gamma: {gamma:.6f} /us")
print(f"Expected decay after T1 time: {np.exp(-1):.4f} (should be ~0.368)")
print("Starting Adaptive Bayesian Estimation...")

# --- 2. 실험 루프 ---
for i in range(num_shots):
    # 1. Controller: 다음 waiting time 결정
    if i == 0:
        tau = estimator.theta / estimator.k 
    else:
        tau = estimator.get_adaptive_tau(c=c_adaptive)
    
    # 2. Experiment (QuTiP): 큐비트 시간 발전 및 측정
    # 시간: 0 ~ tau
    print(f"Shot {i+1}: tau = {tau:.4f} us")
    tlist = np.linspace(0, tau, 100)
    # mesolve로 밀도 행렬 계산 (Master Equation)
    output = mesolve(H, rho_excited, tlist, c_ops, e_ops=[])
    final_state = output.states[-1]
    
    # Excited state population (P_1) 계산
    # QuTiP basis(2,1)이 excited라면, projection operator는 basis(2,1)*basis(2,1).dag()
    p_excited = expect(psi_excited * psi_excited.dag(), final_state)
    print(f"  -> P(excited) = {p_excited:.4f}, Expected = {np.exp(-tau/true_T1):.4f}")
    print(f"  -> Final state diagonal: [{final_state.data.to_array()[0,0]:.4f}, {final_state.data.to_array()[1,1]:.4f}]")
    
    # 3. Measurement Simulation (Single Shot with SPAM)
    # 실제 물리적 붕괴 시뮬레이션
    is_decayed = np.random.rand() > p_excited # True if decayed to Ground
    
    if is_decayed: # State is Ground |0>
        # P(1|0) error (beta) 적용
        measured_state = 1 if np.random.rand() < beta else 0
    else: # State is Excited |1>
        # P(0|1) error (alpha) 적용
        measured_state = 0 if np.random.rand() < alpha else 1
        
    results_tau.append(tau)
    results_m.append(measured_state)
    
    # 4. Controller: Bayesian Update
    estimator.update(measured_state, tau)

# --- 3. 결과 시각화 ---
t1_history = estimator.history['T1_est']
shots = np.arange(1, num_shots + 1)

plt.figure(figsize=(10, 6))

# T1 추정치 변화
plt.plot(shots, t1_history, 'o-', color='purple', label='Estimated T1')
plt.axhline(y=true_T1, color='k', linestyle='--', label='True T1')
plt.fill_between(shots, 
                 [t - t*0.2 for t in t1_history], 
                 [t + t*0.2 for t in t1_history], 
                 color='purple', alpha=0.1, label='Uncertainty (approx)')

# 측정 결과 표시 (상단/하단)
for j, (m, t) in enumerate(zip(results_m, results_tau)):
    y_pos = min(t1_history) * 0.8 if m==0 else max(t1_history) * 1.1
    color = 'red' if m==0 else 'blue'
    plt.scatter(j+1, y_pos, color=color, s=10, marker='x' if m==0 else 'o')

plt.xlabel('Experiment Number (N)')
plt.ylabel(r'$T_1$ Estimate ($\mu s$)')
plt.title(f'Real-time Adaptive $T_1$ Estimation (True $T_1$={true_T1}$\\mu s$)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"Final Estimated T1: {t1_history[-1]:.2f} us after {num_shots} shots.")