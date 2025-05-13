'''
mu - гравитационный параметр
p - фокальный параметр
e - эксцентриситет
O - долгота восходящего узла
0 - истинная аномалия
i - наклонение
w - аргумент перицентра
t_pi - время прохождения перигея
u - широта
N - возмущающее ускорение на трансверсаль
W - возмущающее ускорение на бинормаль
S - возмущающее ускорение на радиаль

p

j = 1 / (1 + (r^3 * W * sin(u) * ctg(i))/(mu * p))

B_ка = C_a * 0.5 * (F_m)/(m_ка)
N_a = -B_ка * rho v^2 (1 + e * cos(0)) / (1 + 2 * e * cos(0) + e^2) ^ 1/2
S_a = -B_ка * rho v^2 (e * sin(0)) / (1 + 2 * e * cos(0) + e^2) ^ 1/2

v = ( ((mu) / p) * (1 + 2e * cos(0) + e^2) ) ^ 1/2

du = (u * p) ^ 1/2 / (r^2 * j)
dp = (2j)/2 * r^3 * N
dO = -(r^3 * j)/(mu * p) * sin(u)/sin(i) * W
di = -(r^3 * j)/(mu * p) * cos(u) * W
de = (r^2 * j)/(mu) * ((1 + r/p) * N * cos(0) + S * sin(0) + (e * r * N)/p)
dw = (r^2 * j)/(mu * e) * ((1 + r/p) N * sin(0) + S * cos(0) + (e * r * W * sin(u) * ctg(i))/p)
dt_p = (r^2 * j) / (mu * p) ^ 1/2

e_0 0,02
Н_п км 170
i_0 20
w_0 20
O_0 70
C_a 2,1
F_m / m_ка = alpha 0,02
'''

import numpy as np
import matplotlib.pyplot as plt

C_a = 2.1
alpha = 0.02
B_ka = C_a * 0.5 * alpha
rho = 1.83e-12
mu = 398600.4415

dots_count = 300000
angle_lim = 6 * np.pi
du = angle_lim / dots_count


class sub_orbit:
    def setup(self, p, e, O, i, w, t_p):
        self.p = p
        self.e = e
        self.O = O
        self.i = i
        self.w = w
        self.t_p = t_p

def calc_orbit(orbit):
    def process_step(previous_orbit: sub_orbit, u_0):
        def runge_kutta_step(df, x_0, dt):
            k_1 = dt * df(x_0)
            k_2 = dt * df(x_0 + k_1 / 2)
            k_3 = dt * df(x_0 + k_2 / 2)
            k_4 = dt * df(x_0 + k_3)

            return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        p, e, O, i, w, t_p = previous_orbit.p, previous_orbit.e, previous_orbit.O, previous_orbit.i, previous_orbit.w, previous_orbit.t_p
        tetta = w - u_0

        r = p / (1 + e * np.cos(tetta))
        v = np.sqrt((mu / p) * (1 + 2 * e * np.cos(tetta) + (e**2)))

        N = -B_ka * rho * (v ** 2) * ((1 + e * np.cos(tetta)) / np.sqrt(1 + 2 * e * np.cos(tetta) + (e ** 2)))
        S = -B_ka * rho * (v ** 2) * ((e * np.sin(tetta)) / np.sqrt(1 + 2 * e * np.cos(tetta) + (e ** 2)))
        W = 0

        j = 1 / (1 + ((r**3) * W * np.sin(u_0) * (1 / np.tan(i))) / (mu * p))

        def dp(u): return (2 * i * (r**3) * N) / mu
        def dO(u): return -((r**3) * j * np.sin(u) * W) / (mu * p * np.sin(i))
        def di(u): return -((r**3) * j * np.cos(u) * W) / (mu * p)
        def de(u): return ((r**2) * j * (1 + r / p) * N * np.cos(tetta) + S * np.sin(tetta) + (e * r * N) / p) / mu
        def dw(u): return ((r**2) * j * (1 + r / p) * N * np.sin(tetta) - S * np.cos(tetta) + (e * r * W * np.sin(u) * (1 / np.tan(i))) / p) / (mu * e)
        def dt_p(u): return ((r**2) * j) / np.sqrt(mu * p)

        diffs = []
        values = [p, e, O, i, w, t_p]
        for d in [dp, dO, di, de, dw, dt_p]:
            diff = runge_kutta_step(d, u_0, du)
            diffs.append(diff)

        return [d + v for d, v in zip(diffs, values)]

    with open("orbit_logs.txt", "w") as fstream:
        fstream.write("p, e, 0, i, w, t_p, u\n\n")

        orbits = [base_orbit]
        dot_arg = np.linspace(orbit.w, angle_lim + orbit.w, dots_count)
        for i in range(1, dots_count):
            orbit = sub_orbit()
            values = process_step(orbits[i - 1], dot_arg[i])
            orbit.setup(*values)
            orbits.append(orbit)

            print(i)
            fstream.write(values.__str__() + str(dot_arg[i]) + "\n")

    return orbits

base_orbit = sub_orbit()
base_orbit.setup(600,0.9, np.deg2rad(63.4),np.deg2rad(63.4),np.deg2rad(-90), 0)


orbits = calc_orbit(base_orbit)
orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in orbits]

p, e, O, i, w, t_p = map(list, zip(*orbits_values))
dot_arg = np.linspace(base_orbit.w, angle_lim + base_orbit.w, dots_count)

r = [p[i] / (1 + e[i] * np.cos(w[i] - dot_arg[i])) for i in range(dots_count)]
v = [np.sqrt(2 * (mu / r[i] + e[i])) for i in range(dots_count)]

#f, axes = plt.subplots(6, 1)
# axes[0].plot(dot_arg, p, label="p")
# axes[1].plot(dot_arg, e, label="e")
# axes[2].plot(dot_arg, O, label="O")
# axes[3].plot(dot_arg, i, label="i")
# axes[4].plot(dot_arg, w, label="w")
# axes[5].plot(dot_arg, p, label="t_p")

# f, axes = plt.subplots(2, 1)
# axes[0].plot(dot_arg, r, label="r")
# axes[1].plot(dot_arg, v, label="v")

#ax = plt.subplot(2, 1, 1)
plt.axes(projection = 'polar')
plt.polar(dot_arg, r, label="r")


plt.legend()
plt.show()