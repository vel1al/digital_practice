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
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class sub_orbit:
    def setup(self, p, e, O, i, w, t_p):
        self.p = p # фокальный параметр
        self.e = e # эксцентриситет
        self.O = O # долгота восходящего узла
        self.i = i # наклонение орбиты
        self.w = w # аргумент перицентра
        self.t_p = t_p # время пересечения перицентра

# константы
C_a = 2.1 # коэффициент аэродинамического сопротивления
alpha = 0.02
B_ka = C_a * 0.5 * alpha # конструктивный баллистический коэффициент
mu = 3.98603e5 # гравитационный параметр Земли
delta = 66.07e3 # константа, обусловленная сжатием Земли
earth_r = 6371 # радиус Земли
q = 1396 # плотность потока Солнца на орбите Земли (вт/м^2)
n = 0.3 # кпд преобразования фотоэллектрических панелей
k = 0.07 # коэффициент деградации фотоэллектрических панелей (exp(процент кпд) за год)

# входные данные
a_0 = 10000 # большая полуось орбиты
i_0 = 63.4
O_0 = 32.46
w_0 = 0
e_0 = 0.03
S_sp = 10 # площадь фотоэллектрических панелей
P_load = 500 # мощность, потребляемая целевой нагрузкой
P_service = 700 # мощность, потребляемая служебной нагрузкой
max_battery_capacity = 3000 # ёмкость аккумулятора
sun_vector = np.array([0, 1, 0.3]) # радиус-вектор Солнца в геоцентрической СК
start_time = datetime(day=26,year=2025,month=5) # условное время старта (начала движения из перицентра)


p_0 = a_0 / (1 - e_0**2)

base_orbit = sub_orbit()
base_orbit.setup(p_0, e_0, np.deg2rad(O_0), np.deg2rad(i_0), w_0, 0) # заданная орбита движения

atmosphere_h_cutoff = 500 # граничная высота расчёта влияния аэродинамического торможения
solar_angle_cutoff = np.pi / 2 # предельный угол падения солнечных лучей на фотоэлектрические панели
rounds_count = 200 # количество витков для долгосрочного моделирования
dots_per_round = 10000 # количество точек итерационного вычисления на один оборот

dots_count_long = dots_per_round * rounds_count # общее количество точек на долгосрочном моделировании
dots_count_semi = dots_per_round * rounds_count // 10 # общее количество точек на краткосрочном моделировании
dots_count_short = dots_per_round # общее количество точек на одновиточном моделировании

angle_lim_long = 2 * rounds_count * np.pi # предельный угол аргумента широты при долгосрочном моделировании
angle_lim_semi = 2 * rounds_count * np.pi
angle_lim_short = 2 * np.pi

dot_arg_long = np.linspace(0, angle_lim_long, dots_count_long) # линейное пространство интегрирования по аргументу широты
dot_arg_semi = np.linspace(0, angle_lim_semi, dots_count_semi)
dot_arg_short = np.linspace(0, angle_lim_short, dots_count_short)

def air_density(h):
    h_km = h if isinstance(h, (float, int)) else np.array(h)
    h_m = h_km * 1000

    layers = [
        (0, 1.225, 8440),
        (25000, 3.899e-2, 6490),
        (30000, 1.774e-2, 6750),
        (50000, 2.541e-3, 7070),
        (70000, 2.233e-4, 7470),
        (100000, 5.297e-5, 8500),
        (150000, 2.511e-6, 12200),
        (200000, 2.076e-7, 25000),
        (500000, 1.916e-10, 45000)
    ]

    rho = np.zeros_like(h_m)
    for i in range(len(layers)):
        h_min = layers[i][0]
        h_max = layers[i + 1][0] if i < len(layers) - 1 else np.inf
        mask = (h_m >= h_min) & (h_m < h_max)
        if np.any(mask):
            rho0, H = layers[i][1], layers[i][2]
            rho[mask] = rho0 * np.exp(-(h_m[mask] - h_min) / H)

    return rho



def calc_kepler_orbit(base_orbit, dots_count, simulating_angle=2 * np.pi):
    e = base_orbit.e
    p = base_orbit.p
    w = base_orbit.w

    u = np.linspace(w, simulating_angle + w, dots_count)
    r = p / (1 + e * np.cos(u - w))

    return [r, u]


def julian_date(dt):
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3

    return dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045 + (
            dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400


def greenwich_sidereal_time(dt):
    JD = julian_date(dt)
    T = (JD - 2451545.0) / 36525.0

    GMST_deg = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000

    GMST_deg %= 360

    return np.deg2rad(GMST_deg)


def eci_to_geodetic(x, y, z, dt):
    theta = greenwich_sidereal_time(dt)

    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    z_rot = z

    r = np.sqrt(x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    lon = np.arctan2(y_rot, x_rot)
    lat = np.arcsin(z_rot / r)

    return np.degrees(lat), np.degrees(lon)


def orbit_to_eci(orbit, vec):
    w, O, i = orbit.w, orbit.O, orbit.i

    R_w = np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])

    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])

    R_O = np.array([
        [np.cos(O), -np.sin(O), 0],
        [np.sin(O), np.cos(O), 0],
        [0, 0, 1]
    ])

    r_eci = R_O @ R_i @ R_w @ vec

    return r_eci


def calc_osculating_orbits(base_orbit, logs_file_path, dots_count, u_i, b_earth_shape_disturbed=False,
                           b_atmoshpere_disturbed=False, simulating_angle=2 * np.pi, log_periodicity=1):
    du = simulating_angle / dots_count

    def process_step(previous_orbit: sub_orbit, u):
        def runge_kutta_step(df, x_0, dt):
            k_1 = dt * df(x_0)
            k_2 = dt * df(x_0 + k_1 / 2)
            k_3 = dt * df(x_0 + k_2 / 2)
            k_4 = dt * df(x_0 + k_3)

            return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        p, e, O, i, w, t_p = previous_orbit.p, previous_orbit.e, previous_orbit.O, previous_orbit.i, previous_orbit.w, previous_orbit.t_p
        u_0 = u % (2 * np.pi)
        theta = w - u_0 # угол истинной аномалии

        r = p / (1 + e * np.cos(theta))
        v = np.sqrt(p * mu * (1 + 2 * e * np.cos(theta) + e ** 2)) / p

        N_a, S_a, W_a = 0, 0, 0
        if b_atmoshpere_disturbed:
            h = r - earth_r

            if h < atmosphere_h_cutoff:
                rho = air_density(h)

                N_a = -B_ka * rho * (v ** 2) * ((1 + e * np.cos(theta)) / np.sqrt(1 + 2 * e * np.cos(theta) + (e ** 2)))
                S_a = -B_ka * rho * (v ** 2) * ((e * np.sin(theta)) / np.sqrt(1 + 2 * e * np.cos(theta) + (e ** 2)))

        N_g, S_g, W_g = 0, 0, 0
        if b_earth_shape_disturbed:
            N_g = - (mu * delta * np.sin(2 * u_0) * (np.sin(i) ** 2)) / (r ** 4)
            S_g = (mu * delta * (3 * (np.sin(u_0) ** 2) * (np.sin(i) ** 2) - 1)) / (r ** 4)
            W_g = (mu * delta * np.sin(u_0) * np.sin(2 * i)) / (r ** 4)

        N = N_a + N_g
        S = S_a + S_g
        W = W_a + W_g

        j = 1 / (1 + ((r ** 3) * W * np.sin(u_0) * (1 / np.tan(i))) / (mu * p))

        def dp(u):
            return (2 * j * (r ** 3) * N) / mu

        def dO(u):
            return -((r ** 3) * j * np.sin(u_0) * W) / (mu * p * np.sin(i))

        def di(u):
            return -((r ** 3) * j * np.cos(u_0) * W) / (mu * p)

        def de(u):
            return ((r ** 2) * j * ((1 + r / p) * N * np.cos(theta) + S * np.sin(theta) + (e * r * N) / p)) / mu

        def dw(u):
            return (((r ** 2) * j) * ((1 + r / p) * N * np.sin(theta) - S * np.cos(theta) + (e * r * W * np.sin(u_0) * (1 / np.tan(i))) / p)) / (mu * e)

        def dt_p(u):
            return ((r ** 2) * j) / np.sqrt(mu * p)


        diffs = []
        values = [p, e, O, i, w, t_p]
        for d in [dp, de, dO, di, dw, dt_p]:
            diff = runge_kutta_step(d, u_0, du)
            diffs.append(diff)

        return [d + v for d, v in zip(diffs, values)]

    orbits = [base_orbit]

    with open(logs_file_path, "w") as fstream:
        for i in range(1, dots_count):
            orbit = sub_orbit()
            values = process_step(orbits[i - 1], u_i[i])
            orbit.setup(*values)
            orbits.append(orbit)

            if i % log_periodicity == 0:
                output = values + [u_i[i]]
                fstream.write(output.__str__() + "\n")

            if i % dots_per_round == 0:
                print(f"{i // dots_per_round} - виток")

    return orbits


def calculate_solar_angle(trans_vector, radial_vector, sun_vector):
    n = np.cross(trans_vector, radial_vector)
    dot_product = np.dot(n, sun_vector)

    norm_n = np.linalg.norm(n)
    norm_sun_v = np.linalg.norm(sun_vector)

    cos_alpha = np.abs(dot_product) / (norm_n * norm_sun_v)
    alpha = np.arccos(cos_alpha)
    theta_rad = np.pi / 2 - alpha

    return theta_rad


def calculate_velocity(orbit, theta):
    p, e, w, O, i = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i
    h = np.sqrt(mu * p)

    v_x_perifocal = -(mu / h) * np.sin(theta)
    v_y_perifocal = (mu / h) * (e + np.cos(theta))
    v_z_perifocal = 0

    v_perifocal = np.array([v_x_perifocal, v_y_perifocal, v_z_perifocal])

    v_inertial = orbit_to_eci(orbit, v_perifocal)

    return v_inertial


def calculate_tangent(orbit, theta):
    v = calculate_velocity(orbit, theta)
    tangent = v / np.linalg.norm(v)

    return tangent


b_recalc_orbit = True

disturbed_orbits_long, disturbed_orbits_short, disturbed_a_orbits, disturbed_g_orbits = [base_orbit], [base_orbit], [base_orbit], [base_orbit]
if (b_recalc_orbit):
    disturbed_orbits_long = calc_osculating_orbits(base_orbit, "disturbed_orbit_long_logs.txt", dots_count_long, dot_arg_long, True, True,
                                                   simulating_angle=angle_lim_long)
    disturbed_orbits_short = calc_osculating_orbits(base_orbit, "disturbed_orbit_short_logs.txt", dots_count_short, dot_arg_short, True, True,
                                                    simulating_angle=angle_lim_short)
    disturbed_a_orbits = calc_osculating_orbits(base_orbit, "disturbed_a_orbit_logs.txt", dots_count_semi, dot_arg_semi, False, True,
                                                simulating_angle=angle_lim_semi)
    disturbed_g_orbits = calc_osculating_orbits(base_orbit, "disturbed_g_orbit_logs.txt", dots_count_semi, dot_arg_semi, True, False,
                                                simulating_angle=angle_lim_semi)

else:
    with open("disturbed_orbit_long_logs.txt", 'r') as fstream:
        while fstream:
            line = fstream.readline()
            if line:
                values = json.loads(line)

                orbit_values = values[:-1]

                orbit = sub_orbit()
                orbit.setup(*orbit_values)
                disturbed_orbits_long.append(orbit)
            else:
                break
    with open("disturbed_orbit_short_logs.txt", 'r') as fstream:
        while fstream:
            line = fstream.readline()
            if line:
                values = json.loads(line)

                orbit_values = values[:-1]

                orbit = sub_orbit()
                orbit.setup(*orbit_values)
                disturbed_orbits_short.append(orbit)
            else:
                break
    with open("disturbed_a_orbit_logs.txt", 'r') as fstream:
        while fstream:
            line = fstream.readline()
            if line:
                values = json.loads(line)

                orbit_values = values[:-1]

                orbit = sub_orbit()
                orbit.setup(*orbit_values)
                disturbed_a_orbits.append(orbit)
            else:
                break
    with open("disturbed_g_orbit_logs.txt", 'r') as fstream:
        while fstream:
            line = fstream.readline()
            if line:
                values = json.loads(line)

                orbit_values = values[:-1]

                orbit = sub_orbit()
                orbit.setup(*orbit_values)
                disturbed_g_orbits.append(orbit)
            else:
                break

def calculate_power_usage(sun_vector, T_0):
    Ns, ts, bs, angls = [], [], [], []

    t_prev = 0
    battery_capacity = max_battery_capacity
    for itter in range(1, dots_count_long, 10):
        orbit = disturbed_orbits_long[itter - 1]
        p, e, w, O, i, t_p = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i, orbit.t_p
        theta = w - dot_arg_long[itter] % (2 * np.pi)

        r = p / (1 + e * np.cos(theta))

        x_orb = r * np.cos(theta) # текущие орбитальные координаты
        y_orb = r * np.sin(theta)
        z_orb = 0

        r_orb = np.array([x_orb, y_orb, z_orb])
        r_eci = orbit_to_eci(orbit, r_orb)

        pos_vector = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.dot(sun_vector, pos_vector))
        earth_angular_radius = np.arctan(earth_r / np.linalg.norm(r_eci))

        N, solar_angle = -P_service, 0
        if angle < (np.pi / 2 - earth_angular_radius):
            r_radial = r_eci
            r_transversal = calculate_tangent(orbit, theta)

            solar_angle = calculate_solar_angle(r_transversal, r_radial, sun_vector)

            if solar_angle < solar_angle_cutoff:
                N += q * S_sp * n * np.exp(-k * ((T_0 + t_p) / 31536000)) * np.cos(solar_angle)
                N -= P_load

        battery_capacity += N * (t_p - t_prev) / 3600
        if battery_capacity <= 0:
            N = 0
            battery_capacity = 0
        elif battery_capacity >= max_battery_capacity:
            battery_capacity = max_battery_capacity

        Ns.append(N)
        ts.append(t_p)
        bs.append(battery_capacity)
        angls.append(solar_angle)

        t_prev = t_p

    plt.figure(figsize=(20, 16))
    ax = plt.subplot(411)
    ax.scatter(ts, Ns, color="black", s=1)

    for i in range(1, rounds_count, 5):
        ax.axvline(ts[(len(ts) // rounds_count) * i - 1], color="red", label=f'{i}-й виток', linestyle=':')

    ax.set_xlabel('Время с')
    ax.set_ylabel('Мощность Вт/ч')
    ax.set_title(label=f'В течение {rounds_count} витков')

    ax2 = plt.subplot(412)
    ax2.scatter(ts, angls, color="black", s=1)
    ax2.set_xlabel('Время с')
    ax2.set_ylabel('Угол падения солнечных лучей рад')
    ax2.set_title(label=f'В течение {rounds_count} витков')

    ax3 = plt.subplot(413)
    ax3.scatter(ts[:len(ts) // rounds_count], Ns[:len(ts) // rounds_count], color="black", s=1)
    ax3.set_xlabel('Время с')
    ax3.set_ylabel('Мощность Вт/ч')
    ax3.set_title(label='В течение 1 витка')

    ax4 = plt.subplot(414)
    ax4.scatter(ts[:len(ts) // rounds_count], angls[:len(ts) // rounds_count], color="black", s=1)
    ax4.set_xlabel('Время с')
    ax4.set_ylabel('Угол падения солнечных лучей рад')
    ax4.set_title(label='В течение 1 витка')

    plt.figure(figsize=(20, 16))
    ax = plt.subplot(211)
    ax.scatter(ts, bs, color="red", s=1)

    for i in range(1, rounds_count, 5):
        ax.axvline(ts[(len(ts) // rounds_count) * i - 1], color="red", label=f'{i}-й виток', linestyle=':')

    ax.set_xlabel('Время с')
    ax.set_ylabel('Запас Вт/ч')
    ax.set_title(label=f'В течение {rounds_count} витков')

    ax2 = plt.subplot(212)
    ax2.scatter(ts[:len(ts) // rounds_count], bs[:len(ts) // rounds_count], color="red", s=1)
    ax2.set_xlabel('Время с')
    ax2.set_ylabel('Запас Вт/ч')
    ax2.set_title(label='В течение 1 витка')

def estimate_orbital_disturbances():
    kepler_orbit = calc_kepler_orbit(base_orbit, dots_count_short)
    osculating_orbit = calc_osculating_orbits(base_orbit, "non_disturbed_orbit_logs.txt", dots_count_short, dot_arg_short)

    disturbed_orbits_long_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in disturbed_orbits_long]
    disturbed_orbits_short_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in osculating_orbit]
    disturbed_a_orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in
                                 disturbed_a_orbits]
    disturbed_g_orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in
                                 disturbed_g_orbits]

    r_dist = [disturbed_orbits_long_values[i][0] / (
            1 + disturbed_orbits_long_values[i][1] * np.cos(dot_arg_long[i] - disturbed_orbits_long_values[i][4])) for i in
              range(len(disturbed_orbits_long_values))]
    r_osc = [disturbed_orbits_short_values[i][0] / (
            1 + disturbed_orbits_short_values[i][1] * np.cos(dot_arg_short[i] - disturbed_orbits_short_values[i][4])) for i in
             range(len(disturbed_orbits_short_values))]

    r_2, u_2 = kepler_orbit[0], kepler_orbit[1]

    columns = list(zip(*disturbed_orbits_long_values))
    columns_a = list(zip(*disturbed_a_orbits_values))
    columns_g = list(zip(*disturbed_g_orbits_values))

    orbits_dict = ["фокальный параметр", "эксцентриситет", "долгота восходящего узла", "наклонение орбиты",
                   "аргумент перицентра"]

    #Оценка точности метода оскулирующих орбит
    plt.figure(figsize=(6, 6))
    plt.axes(projection='polar')
    plt.title('Уравнение элипса', pad=20)
    plt.plot(u_2, r_2, color='blue', label='r')
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(6, 6))
    plt.axes(projection='polar')
    plt.title("Метод оскулирующих орбит (без возмущений)")
    plt.plot(dot_arg_short, r_osc, color='red', label='r')
    plt.grid(True)
    plt.legend()

    diff = r_osc - r_2
    plt.figure(figsize=(6, 6))
    plt.axes(projection='polar')
    plt.title('Сравнение', pad=20)
    plt.plot(dot_arg_short, diff, color='red', label='dr')
    plt.grid(True)
    plt.legend()

    #Оценка вековых изменений параметров орбиты силами аэродинамического торможения
    for i in range(5):
        plt.figure(figsize=(10, 10))
        plt.title(orbits_dict[i] + " без влияния нецентральности гравитационного поля", pad=20)
        plt.scatter(dot_arg_semi, columns_a[i], label=orbits_dict[i], s=1, color='b')
        plt.axhline(columns_a[i][0], linestyle=':')

        plt.grid(True)
        plt.legend()

    #Оценка вековых изменений параметров орбиты нецентральности гравитационного поля Земли
    for i in range(5):
        plt.figure(figsize=(10, 10))
        plt.title(orbits_dict[i] + " без влияния аэродинамического торможения", pad=20)
        plt.scatter(dot_arg_semi, columns_g[i], label=orbits_dict[i], s=1, color='b')
        plt.axhline(columns_g[i][0], linestyle=':')

        plt.grid(True)
        plt.legend()

def plot_track():
    plt.figure(figsize=(15, 7))
    image = plt.imread('earth_map.jpg')
    extent = [-180, 180, -90, 90]
    plt.imshow(image, extent=extent)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel('Долгота (°)')
    plt.ylabel('Широта (°)')

    start_time = datetime(2023, 1, 1, 0, 0, 0)
    for itter in range(1, dots_count_short, 10):
        orbit = disturbed_orbits_short[itter - 1]
        p, e, w, O, i, t_p = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i, orbit.t_p
        theta = w - dot_arg_short[itter] % (2 * np.pi)

        r = p / (1 + e * np.cos(theta))

        x_orb = r * np.cos(theta) # текущие орбитальные координаты
        y_orb = r * np.sin(theta)
        z_orb = 0

        r_orb = np.array([x_orb, y_orb, z_orb])
        r_eci = orbit_to_eci(orbit, r_orb)

        t = start_time + timedelta(seconds=t_p)
        lat, lon = eci_to_geodetic(r_eci[0], r_eci[1], r_eci[2], t)

        pos_vector = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.dot(sun_vector, pos_vector))
        earth_angular_radius = np.arctan(earth_r / np.linalg.norm(r_eci))

        if angle > (np.pi / 2 - earth_angular_radius):
            plt.scatter(lon, lat, s=1, color="gray")
        else:
            plt.scatter(lon, lat, s=1, color="yellow")

def plot_orbit():
    theta = np.linspace(0, 2 * np.pi, 201)
    cth, sth, zth = [f(theta) for f in [np.cos, np.sin, np.zeros_like]]

    lon0 = earth_r * np.vstack((cth, zth, sth))
    lons = []
    for phi in np.pi / 180 * np.arange(0, 180, 15):
        cph, sph = [f(phi) for f in [np.cos, np.sin]]
        lon = np.vstack((lon0[0] * cph - lon0[1] * sph,
                         lon0[1] * cph + lon0[0] * sph,
                         lon0[2]))
        lons.append(lon)

    lats = []
    for phi in np.pi / 180 * np.arange(-75, 90, 15):
        cph, sph = [f(phi) for f in [np.cos, np.sin]]
        lat = earth_r * np.vstack((cth * cph, sth * cph, zth + sph))
        lats.append(lat)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    light_points, shadows_points, shadow_itters = [], [], []
    for itter in range(1, dots_count_short, 10):
        orbit = disturbed_orbits_short[itter - 1]
        p, e, w, O, i = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i
        theta = w - dot_arg_short[itter] % (2 * np.pi)

        r = p / (1 + e * np.cos(theta))

        x_orb = r * np.cos(theta) # текущие орбитальные координаты
        y_orb = r * np.sin(theta)
        z_orb = 0

        r_orb = np.array([x_orb, y_orb, z_orb])
        r_eci = orbit_to_eci(orbit, r_orb)

        pos_vector = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.dot(sun_vector, pos_vector))
        earth_angular_radius = np.arctan(earth_r / np.linalg.norm(r_eci))

        if angle > (np.pi / 2 - earth_angular_radius):
            shadows_points.append([r_eci[0], r_eci[1], r_eci[2]])
        else:
            light_points.append([r_eci[0], r_eci[1], r_eci[2]])

    for x, y, z in lons:
        ax.plot(x, y, z, '-k', color='blue')
    for x, y, z in lats:
        ax.plot(x, y, z, '-k', color='blue')
    for x, y, z in light_points:
        ax.scatter(x, y, z, color='yellow')
    for x, y, z in shadows_points:
        ax.scatter(x, y, z, color='black')
    ax.quiver(0, 0, 0, *(sun_vector * 3000), color='yellow')

    plt.gca().set_aspect('equal')


estimate_orbital_disturbances()
plot_orbit()
plot_track()
calculate_power_usage(sun_vector, 0)
plt.show()