import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import Slider, TextBox, RadioButtons
import re
from scipy.integrate import odeint
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # matplotlib '-' 렌더링 오류 방지

# -------------------- 수치 해석 알고리즘 --------------------
def solve_euler(f, x0, t):
    h = t[1] - t[0]
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + h * np.array(f(x[i - 1], t[i - 1]))
    return x

def solve_rk4(f, x0, t):
    h = t[1] - t[0]
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        k1 = np.array(f(x[i - 1], t[i - 1]))
        k2 = np.array(f(x[i - 1] + 0.5 * h * k1, t[i - 1] + 0.5 * h))
        k3 = np.array(f(x[i - 1] + 0.5 * h * k2, t[i - 1] + 0.5 * h))
        k4 = np.array(f(x[i - 1] + h * k3, t[i - 1] + h))
        x[i] = x[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

# -------------------- 입력 함수 파싱 --------------------
def safe_eval(expr):
    allowed_funcs = [
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "heaviside", "piecewise", "arcsin", "arccos", "arctan"
    ]
    for func in allowed_funcs:
        expr = re.sub(rf"(?<!\w){func}(?=\()", rf"np.{func}", expr)
    return expr

def get_u_function(expr):
    expr = safe_eval(expr)
    def u(t):
        try:
            return eval(expr, {"t": t, "np": np})
        except Exception as e:
            print(f"u(t) parse error at t={t}: {e}")
            return 0
    return u

# -------------------- 시뮬레이터 클래스 --------------------
class ODESimulator:
    def __init__(self):
        self.t = np.linspace(0, 10, 1000)
        self.method = "RK4"
        self.fig_u = self.fig_x = self.fig_xdot = None
        self.line_u = self.line_x = self.line_xdot = None
        self.build_ui()

    def build_ui(self):
        self.fig = plt.figure(figsize=(8, 7))
        self.fig.subplots_adjust(left=0.35, bottom=0.5)

        self.fig.text(0.1, 0.84, 'ζ (ζ=1: Critical Damping)', fontsize=10)
        ax_zeta = plt.axes([0.1, 0.80, 0.2, 0.03])
        self.s_zeta = Slider(ax_zeta, '', 0.0, 3.0, valinit=0.5)

        self.damping_text = self.fig.text(0.45, 0.83, '', fontsize=10)

        self.boxes = {}
        positions = {
            'a': [0.05, 0.72, 0.25, 0.04],
            'b': [0.05, 0.66, 0.25, 0.04],
            'c': [0.05, 0.60, 0.25, 0.04],
            'x0': [0.05, 0.54, 0.25, 0.04],
            'x0dot': [0.05, 0.48, 0.25, 0.04],
            'u': [0.05, 0.42, 0.25, 0.04]
        }
        defaults = {'a': '1.0', 'b': '', 'c': '4.0', 'x0': '0', 'x0dot': '0', 'u': '1'}

        for key in positions:
            ax = plt.axes(positions[key])
            self.boxes[key] = TextBox(ax, key, initial=defaults[key])
            self.boxes[key].on_submit(self.update)

        radio_ax = plt.axes([0.05, 0.25, 0.25, 0.15])
        self.radio = RadioButtons(radio_ax, ('RK4', 'Euler', 'odeint'))
        self.radio.on_clicked(self.update)

        self.s_zeta.on_changed(self.on_slider_change)
        self.update(None)
        plt.show()

    def on_slider_change(self, val):
        self.boxes['b'].set_val('')
        self.update(None)

    def update(self, val):
        if hasattr(self, 'eqn_text'):
            self.eqn_text.remove()

        # 수식 문자열 안전하게 생성
        a_text = self.boxes['a'].text
        b_text_raw = self.boxes['b'].text
        c_text = self.boxes['c'].text
        u_text = self.boxes['u'].text

        try:
            a_val = float(a_text)
            c_val = float(c_text)
            zeta_val = self.s_zeta.val
            b_val_auto = 2 * zeta_val * np.sqrt(a_val * c_val)
            b_display = b_text_raw if b_text_raw else f"{b_val_auto:.2f}"
        except Exception as e:
            b_display = "?"

        equation_str = f"{a_text}·x''(t) + {b_display}·x'(t) + {c_text}·x(t) = {u_text}"
        self.eqn_text = self.fig.text(0.05, 0.95, f"Equation: {equation_str}", fontsize=10, verticalalignment='top')

        try:
            a = float(a_text)
            c = float(c_text)
            x0 = [float(self.boxes['x0'].text), float(self.boxes['x0dot'].text)]
            u_func = get_u_function(u_text.strip())

            if b_text_raw == "":
                zeta = self.s_zeta.val
                b = 2 * zeta * np.sqrt(a * c)
            else:
                b = float(b_text_raw)
                zeta = b / (2 * np.sqrt(a * c))
                self.s_zeta.set_val(zeta)

            def system(x, t):
                dx1dt = x[1]
                dx2dt = (1 / a) * (u_func(t) - b * x[1] - c * x[0])
                return [dx1dt, dx2dt]

            method = self.radio.value_selected
            if method == "RK4":
                sol = solve_rk4(system, x0, self.t)
            elif method == "Euler":
                sol = solve_euler(system, x0, self.t)
            else:
                sol = odeint(system, x0, self.t)

            u_vals = np.array([u_func(ti) for ti in self.t])
            self.update_plot('u', u_vals, 'Input u(t)', 'u(t)')
            self.update_plot('x', sol[:, 0], 'State x(t)', 'x(t)')
            self.update_plot('xdot', sol[:, 1], 'State ẋ(t)', 'ẋ(t)')

            if zeta < 1:
                damping_case = 'Underdamped'
            elif np.isclose(zeta, 1.0):
                damping_case = 'Critically damped'
            else:
                damping_case = 'Overdamped'
            self.damping_text.set_text(f"Damping condition: {damping_case}")

        except Exception as e:
            print("Error:", e)

    def update_plot(self, plot_id, y, title, ylabel):
        fig_attr = f"fig_{plot_id}"
        ax_attr = f"ax_{plot_id}"
        line_attr = f"line_{plot_id}"

        if getattr(self, fig_attr) is None:
            fig, ax = plt.subplots()
            line, = ax.plot(self.t, y, label=ylabel)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid()
            ax.legend()
            setattr(self, fig_attr, fig)
            setattr(self, ax_attr, ax)
            setattr(self, line_attr, line)
            fig.show()
        else:
            line = getattr(self, line_attr)
            ax = getattr(self, ax_attr)
            fig = getattr(self, fig_attr)
            line.set_ydata(y)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()


if __name__ == "__main__":
    ODESimulator()
