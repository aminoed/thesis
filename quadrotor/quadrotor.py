
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

g = 9.81
R = 0.1
l = 0.5
M_rotor = 0.1
M = 1
m = M + 4*M_rotor

Jx = Jy = 2/5 * M * (R**2) + 2 * M_rotor*(l**2)
Jz = 2/5*M*R**2 + 4*M_rotor*(l**2)

TARGET_HEIGHT = 1.0

def quadrotor_ode(t, state):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, t = state

    F, T_x, T_y = pd_controller(state)
    dx1 = np.cos(x8)*np.cos(x9)*x4 + (np.sin(x7)*np.sin(x8)*np.cos(x9) - np.cos(x7)*np.sin(x9))*x5 + (np.cos(x7)*np.sin(x8)*np.cos(x9) + np.sin(x7)*np.sin(x9))*x6
    dx2 = np.cos(x8)*np.sin(x9)*x4 + (np.sin(x7)*np.sin(x8)*np.sin(x9) + np.cos(x7)*np.cos(x9))*x5 + (np.cos(x7)*np.sin(x8)*np.sin(x9) - np.sin(x7)*np.cos(x9))*x6
    dx3 = np.sin(x8)*x4 - np.sin(x7)*np.cos(x8)*x5 - np.cos(x7)*np.cos(x8)*x6
    dx4 = x12*x5 - x11*x6 - g*np.sin(x8)    
    dx5 = x10*x6 - x12*x4 + g*np.cos(x8)*np.sin(x7)
    dx6 = x11*x4 - x10*x5 + g*np.cos(x8)*np.cos(x7) - F/m
    
    dx7 = x10 + (np.sin(x7)*(np.sin(x8)/np.cos(x8)))*x11 + (np.cos(x7)*(np.sin(x8)/np.cos(x8)))*x12
    dx8 = np.cos(x7)*x11 - np.sin(x7)*x12
    dx9 = (np.sin(x7)/np.cos(x8))*x11 + (np.cos(x7)/np.cos(x8))*x12
    
    dx10 = (Jy-Jz)/Jx*x11*x12 + T_x/Jx
    dx11 = (Jz-Jx)/Jy*x10*x12 + T_y/Jy
    dx12 = 0
    dt = 1.0

    return [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dt]


def pd_controller(state, target_height=TARGET_HEIGHT, target_roll=0.0, target_pitch=0.0):
    x3= state[2]
    x6 = state[5]
    x7, x8 = state[6], state[7]
    x10, x11 = state[9], state[10]

    F = m * g - 10.0 * (x3 - target_height) + 3.0 * x6
    T_x = -(x7 - target_roll) - x10
    T_y = -(x8 - target_pitch) - x11
    return F, T_x, T_y

def simulate(initial_state, total_time, time_step):
    # Integrate one trajectory over [0, 5] seconds.
    if initial_state is None:
        initial_state = np.zeros(13)
    initial_state = np.asarray(initial_state, dtype=float)
    

    sample_count = round(total_time / time_step) + 1
    output_times = np.linspace(0.0, total_time, sample_count)

    solution = solve_ivp(
        quadrotor_ode,
        t_span=(0.0, total_time),
        y0=initial_state,
        t_eval=output_times,
        method="RK45",
        max_step=time_step,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    return solution


def sample_quadrotor(n):
    initial_state = np.zeros(13)
    initial_state[:6] = rng.uniform(-0.4, 0.4, size=6)
    return initial_state

rng = np.random.default_rng(0)
initial_state = np.zeros(13)
initial_state[:6] = rng.uniform(-0.4, 0.4, size=6)

solution = simulate(initial_state, 5.0, 0.025)

time = solution.t
height = solution.y[2]

plt.plot(time, height, label="x3(t)")
plt.axhline(1.0, color="gray", linestyle="--", label="Target: 1 m")
plt.xlabel("Time t (s)")
plt.ylabel("Height x3 (m)")
plt.grid()
plt.legend()
plt.show()
plt.savefig("height.png", dpi=150)

print(f"Height at 5 s: {height[-1]:.6f} m")
print(f"Maximum sampled height: {height.max():.6f} m")