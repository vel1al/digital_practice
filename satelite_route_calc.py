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

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

C_a = 2.1
alpha = 0.02
B_ka = C_a * 0.5 * alpha
mu = 3.98603e5
delta = 66.07e3
earth_r = 6371

simulating_dots = 1000000
atmosphere_h_cutoff = 500

class sub_orbit:
    def setup(self, p, e, O, i, w, t_p):
        self.p = p
        self.e = e
        self.O = O
        self.i = i
        self.w = w
        self.t_p = t_p


def density_coesa1976(h):
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

    # Среднее звёздное время в градусах
    GMST_deg = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000

    # Приведение к диапазону 0-360°
    GMST_deg %= 360
    return np.deg2rad(GMST_deg)

def eci_to_geodetic(x, y, z, dt):
    theta = greenwich_sidereal_time(dt)

    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    z_rot = z

    r = np.sqrt(x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    lon = np.atan2(y_rot, x_rot)
    lat = np.asin(z_rot / r)

    return np.degrees(lat), np.degrees(lon)

def orbit_to_eci(p, e, O, i, w, u):
    tetta = u - w

    r = p / (1 + e * np.cos(tetta))
    a = p / (1 - e ** 2)

    x_orb = r * np.cos(tetta)
    y_orb = r * np.sin(tetta)
    z_orb = 0

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

    r_orb = np.array([x_orb, y_orb, z_orb])
    r_eci = R_O @ R_i @ R_w @ r_orb

    return r_eci

def calc_osculating_orbits(base_orbit, logs_file_path, dots_count, b_earth_shape_disturbed=False, b_atmoshpere_disturbed=False, simulating_angle=2 * np.pi, log_periodicity = 1):
    du = simulating_angle / dots_count

    def process_step(previous_orbit: sub_orbit, u_0):
        def runge_kutta_step(df, x_0, dt):
            k_1 = dt * df(x_0)
            k_2 = dt * df(x_0 + k_1 / 2)
            k_3 = dt * df(x_0 + k_2 / 2)
            k_4 = dt * df(x_0 + k_3)

            return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        p, e, O, i, w, t_p = previous_orbit.p, previous_orbit.e, previous_orbit.O, previous_orbit.i, previous_orbit.w, previous_orbit.t_p
        tetta = u_0 - w

        r = p / (1 + e * np.cos(tetta))
        v = np.sqrt(p * mu * (1 + 2 * e * np.cos(tetta) + e**2)) / p

        N_a, S_a, W_a = 0, 0, 0
        if b_atmoshpere_disturbed:
            h = r - earth_r

            if h < atmosphere_h_cutoff:
                rho = density_coesa1976(h)
                N_a = -B_ka * rho * (v ** 2) * ((1 + e * np.cos(tetta)) / np.sqrt(1 + 2 * e * np.cos(tetta) + (e ** 2)))
                S_a = -B_ka * rho * (v ** 2) * ((e * np.sin(tetta)) / np.sqrt(1 + 2 * e * np.cos(tetta) + (e ** 2)))


        N_g, S_g, W_g = 0, 0, 0
        if b_earth_shape_disturbed:
            N_g = - (mu * delta * 3 * np.sin(2 * u_0) * np.sin(i) ** 2) / (r ** 4)
            S_g = (mu * delta * (3 * (np.sin(u_0) ** 2) * (np.sin(i) ** 2) - 1)) / (r ** 4)
            W_g = (mu * delta * 3 * np.sin(u_0) * np.sin(2 * i)) / (r ** 4)

        N = N_a + N_g
        S = S_a + S_g
        W = W_a + W_g

        j = 1 / (1 + ((r ** 3) * W * np.sin(u_0) * (1 / np.tan(i))) / (mu * p))

        def dp(u):
            return (2 * j * (r ** 3) * N) / mu

        def dO(u):
            return -((r ** 3) * j * np.sin(u) * W) / (mu * p * np.sin(i))

        def di(u):
            return -((r ** 3) * j * np.cos(u) * W) / (mu * p)

        def de(u):
            return ((r ** 2) * j * (1 + r / p) * N * np.cos(tetta) + S * np.sin(tetta) + (e * r * N) / p) / mu

        def dw(u):
            return ((r ** 2) * j * (1 + r / p) * N * np.sin(tetta) - S * np.cos(tetta) + (
                    e * r * W * np.sin(u) * (1 / np.tan(i))) / p) / (mu * e)

        def dt_p(u):
            return ((r ** 2) * j) / np.sqrt(mu * p)

        diffs = []
        values = [p, e, O, i, w, t_p]
        for d in [dp, dO, di, de, dw, dt_p]:
            diff = runge_kutta_step(d, u_0, du)
            diffs.append(diff)

        return [d + v for d, v in zip(diffs, values)]

    with open(logs_file_path, "w") as fstream:
        orbits = [base_orbit]
        dot_arg = np.linspace(0, base_orbit.w, dots_count)
        for i in range(1, dots_count):
            orbit = sub_orbit()
            values = process_step(orbits[i - 1], dot_arg[i])
            orbit.setup(*values)
            orbits.append(orbit)

            if i % log_periodicity == 0:
                output = values + [dot_arg[i]]
                fstream.write(output.__str__() + "\n")

    return orbits


base_orbit = sub_orbit()
base_orbit.setup(10000, 0.3, np.deg2rad(32.46), np.deg2rad(70), 0, 0)

def estimate_orbital_disturbances(dots_count):
    b_recalc_orbit = True

    disturbed_orbits, disturbed_a_orbits, disturbed_g_orbits = [base_orbit], [base_orbit], [base_orbit]
    if (b_recalc_orbit):
        disturbed_orbits = calc_osculating_orbits(base_orbit, "disturbed_orbit_logs.txt", dots_count, True, True, simulating_angle=20 * np.pi)
        #disturbed_a_orbits = calc_osculating_orbits(base_orbit, "disturbed_a_orbit_logs.txt", dots_count, False, True, simulating_angle=20 * np.pi)
        #disturbed_g_orbits = calc_osculating_orbits(base_orbit, "disturbed_g_orbit_logs.txt", dots_count, True, False, simulating_angle=20 * np.pi)
    else:
        with open("disturbed_orbit_logs.txt", 'r') as fstream:
            while fstream:
                line = fstream.readline()
                if line:
                    values = json.loads(line)
                    orbit_values = values[:-1]
                    orbit = sub_orbit()
                    orbit.setup(*orbit_values)
                    disturbed_orbits.append(orbit)
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

    kepler_orbit = calc_kepler_orbit(base_orbit, 100000)
    osculating_orbit = calc_osculating_orbits(base_orbit, "non_disturbed_orbit_logs.txt", 100000)

    disturbed_orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in disturbed_orbits]
    #disturbed_a_orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in disturbed_a_orbits]
    #disturbed_g_orbits_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in disturbed_g_orbits]
    osculating_orbit_values = [[orbit.p, orbit.e, orbit.O, orbit.i, orbit.w, orbit.t_p] for orbit in osculating_orbit]

    dot_arg_2pi = np.linspace(0, 2 * np.pi, 100000)
    dot_arg_2pi = dot_arg_2pi[:-1]
    dot_arg = np.linspace(0, 20 * np.pi, dots_count)
    dot_arg = dot_arg[:-1]

    r_dist = [disturbed_orbits_values[i][0] / (
                1 + disturbed_orbits_values[i][1] * np.cos(dot_arg[i] - disturbed_orbits_values[i][4])) for i in
         range(len(disturbed_orbits_values) - 1)]
    r_osc = [osculating_orbit_values[i][0] / (
                1 + osculating_orbit_values[i][1] * np.cos(dot_arg_2pi[i] - osculating_orbit_values[i][4])) for i in
         range(len(osculating_orbit_values) - 1)]

    r_2, u_2 = kepler_orbit[0], kepler_orbit[1]
    r_2, u_2 = r_2[:-1], u_2[:-1]

    disturbed_orbits_values = disturbed_orbits_values[:-1]
    # disturbed_a_orbits_values = disturbed_a_orbits_values[:-1]
    # disturbed_g_orbits_values = disturbed_g_orbits_values[:-1]
    # columns = list(zip(*disturbed_orbits_values))
    # columns_a = list(zip(*disturbed_a_orbits_values))
    # columns_g = list(zip(*disturbed_g_orbits_values))


    orbits_dict = ["фокальный параметр", "эксцентриситет", "долгота восходящего узла", "наклонение орбиты", "аргумент перицентра"]
    # plt.figure(figsize=(12, 8))
    # plt.title("Оценка точности метода оскулирующих орбит")
    # ax1 = plt.subplot(231, projection='polar')
    # ax1.plot(u_2, r_2, color='blue', label='r')
    # ax1.set_title('Уравнение элипса', pad=20)
    # ax1.grid(True)
    # ax1.legend()
    #
    # ax2 = plt.subplot(232, projection='polar')
    # ax2.plot(dot_arg_2pi, r_osc, color='red', label='r')
    # ax2.set_title('Метод оскулирующих орбит (без возмущений)', pad=20)
    # ax2.grid(True)
    # ax2.legend()
    #
    # diff = r_osc - r_2
    # ax3 = plt.subplot(233)
    # ax3.plot(dot_arg_2pi, diff, color='red', label='dr')
    # ax3.set_title('Сравнение', pad=20)
    # ax3.grid(True)
    # ax3.legend()
    #
    # plt.figure(figsize=(18, 20))
    # plt.title("Оценка вековых изменений параметров орбиты")
    ax4 = plt.subplot(231, projection='polar')
    ax4.plot(dot_arg, r_dist, color='red', label='r')
    # ax4.grid(True)
    # ax4.legend()
    #
    # for i in range(5):
    #     ax = plt.subplot(232 + i)
    #     ax.plot(dot_arg, columns[i], label=orbits_dict[i])
    #     ax.axhline(columns[i][0], linestyle=':')
    #
    #     for u in range(1, 10):
    #         ax.axvline(u * 2 * np.pi, color="red", label=f'{u}-й виток', linestyle=':')
    #
    #     ax.grid(True)
    #     ax.legend()
    #
    # plt.figure(figsize=(18, 20))
    # plt.title("Оценка вековых изменений параметров орбиты силами аэродинамического торможения")
    # for i in range(5):
    #     ax = plt.subplot(232 + i)
    #     ax.plot(dot_arg, columns_a[i], label=orbits_dict[i])
    #     ax.axhline(columns_a[i][0], linestyle=':')
    #
    #     for u in range(1, 10):
    #         ax.axvline(u * 2 * np.pi, color="red", label=f'{u}-й виток', linestyle=':')
    #
    #     ax.grid(True)
    #     ax.legend()
    #
    # plt.figure(figsize=(18, 20))
    # plt.title("Оценка вековых изменений параметров орбиты нецентральности гравитационного поля Земли")
    # for i in range(5):
    #     ax = plt.subplot(232 + i)
    #     ax.plot(dot_arg, columns_g[i], label=orbits_dict[i])
    #     ax.axhline(columns_g[i][0], linestyle=':')
    #
    #     for u in range(1, 10):
    #         ax.axvline(u * 2 * np.pi, color="red", label=f'{u}-й виток', linestyle=':')
    #
    #     ax.grid(True)
    #     ax.legend()

    # theta = np.linspace(0, 2 * np.pi, 201)
    # cth, sth, zth = [f(theta) for f in [np.cos, np.sin, np.zeros_like]]
    #
    # lon0 = earth_r * np.vstack((cth, zth, sth))
    # lons = []
    # for phi in np.pi / 180 * np.arange(0, 180, 15):
    #     cph, sph = [f(phi) for f in [np.cos, np.sin]]
    #     lon = np.vstack((lon0[0] * cph - lon0[1] * sph,
    #                      lon0[1] * cph + lon0[0] * sph,
    #                      lon0[2]))
    #     lons.append(lon)
    #
    # lat0 = earth_r * np.vstack((cth, sth, zth))
    # lats = []
    # for phi in np.pi / 180 * np.arange(-75, 90, 15):
    #     cph, sph = [f(phi) for f in [np.cos, np.sin]]
    #     lat = earth_r * np.vstack((cth * cph, sth * cph, zth + sph))
    #     lats.append(lat)
    #
    # fig = plt.figure(figsize=[10, 8])
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    #
    # shadows_points = []
    # for itter in range(1, 100000 - 1, 500):
    #     orbit = osculating_orbit[itter - 1]
    #     p, e, w, O, i = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i
    #     tetta = dot_arg_2pi[itter] - w
    #
    #     r = p / (1 + e * np.cos(tetta))
    #     a = p / (1 - e ** 2)
    #
    #     x_orb = r * np.cos(tetta)
    #     y_orb = r * np.sin(tetta)
    #     z_orb = 0
    #
    #     R_w = np.array([
    #         [np.cos(w), -np.sin(w), 0],
    #         [np.sin(w), np.cos(w), 0],
    #         [0, 0, 1]
    #     ])
    #
    #     R_i = np.array([
    #         [1, 0, 0],
    #         [0, np.cos(i), -np.sin(i)],
    #         [0, np.sin(i), np.cos(i)]
    #     ])
    #
    #     R_O = np.array([
    #         [np.cos(O), -np.sin(O), 0],
    #         [np.sin(O), np.cos(O), 0],
    #         [0, 0, 1]
    #     ])
    #
    #     r_orb = np.array([x_orb, y_orb, z_orb])
    #
    #     r_eci = R_O @ R_i @ R_w @ r_orb
    #
    #     sun_vector = np.array([0, 1, 0.5])
    #     pos_vector = r_eci / np.linalg.norm(r_eci)
    #
    #     angle = np.arccos(np.dot(sun_vector, pos_vector))
    #     earth_angular_radius = np.arctan(earth_r / np.linalg.norm(r_eci))
    #
    #     if angle > (np.pi / 2 - earth_angular_radius):
    #         ax.scatter(r_eci[0], r_eci[1], r_eci[2], color='k', s=10)
    #         shadows_points.append(i)
    #     else:
    #         ax.scatter(r_eci[0], r_eci[1], r_eci[2], color='y', s=10)
    #     ax.quiver(0, 0, 0, *(sun_vector*3000), color='y')
    #
    # for x, y, z in lons:
    #     ax.plot(x, y, z, '-k', color='b')
    # for x, y, z in lats:
    #     ax.plot(x, y, z, '-k', color='b')

    plt.figure(figsize=(15, 7))
    image = plt.imread('earth_map.jpg')
    extent = [-180, 180, -90, 90]
    plt.imshow(image, extent=extent)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel('Долгота (°)')
    plt.ylabel('Широта (°)')

    start_time = datetime(2023, 1, 1, 0, 0, 0)
    for itter in range(1, dots_count - 1, 100):
        orbit = disturbed_orbits[itter - 1]
        p, e, w, O, i, t_p = orbit.p, orbit.e, orbit.w, orbit.O, orbit.i, orbit.t_p
        eci = orbit_to_eci(p, e, O, i, w, dot_arg[itter])
        t = start_time + timedelta(seconds=t_p)
        lat, lon = eci_to_geodetic(eci[0], eci[1], eci[2], t)

        plt.scatter(lon, lat)

    plt.show()


def calculate_power_generation(orbit_altitude_km, panel_area_m2, efficiency, mission_duration_minutes):
    """
    Рассчитывает циклограмму генерации энергии солнечными панелями на орбите

    Параметры:
    orbit_altitude_km - высота орбиты в км
    panel_area_m2 - площадь солнечных панелей в м²
    efficiency - КПД солнечных панелей (0-1)
    mission_duration_minutes - продолжительность миссии в минутах
    """
    # Константы
    EARTH_RADIUS_KM = 6371  # Радиус Земли в км
    SUN_POWER_W_PER_M2 = 1361  # Солнечная постоянная (Вт/м²)

    # Расчет параметров орбиты
    orbit_radius_km = EARTH_RADIUS_KM + orbit_altitude_km
    orbit_period_minutes = 2 * np.pi * np.sqrt((orbit_radius_km ** 3) / (398600.4418)) / 60

    # Угол, при котором КА входит в тень Земли
    shadow_angle = np.arcsin(EARTH_RADIUS_KM / orbit_radius_km)

    # Время в тени и на свету (в минутах)
    shadow_time_minutes = (2 * shadow_angle / (2 * np.pi)) * orbit_period_minutes
    light_time_minutes = orbit_period_minutes - shadow_time_minutes

    # Генерация временной шкалы
    time_points = np.arange(0, mission_duration_minutes, 1)
    power_output = np.zeros_like(time_points, dtype=float)

    # Расчет мощности для каждого момента времени
    for i, t in enumerate(time_points):
        # Определяем положение на орбите (0 - в перигее, π - в апогее)
        orbit_phase = (t % orbit_period_minutes) / orbit_period_minutes * 2 * np.pi

        # Проверяем, находится ли КА в тени
        if np.abs(orbit_phase - np.pi) < shadow_angle:
            power_output[i] = 0  # В тени
        else:
            # На свету - полная мощность с учетом ориентации
            # Упрощение: считаем, что панели всегда ориентированы на Солнце
            power_output[i] = SUN_POWER_W_PER_M2 * panel_area_m2 * efficiency

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, power_output / 1000, label='Мощность (кВт)')

    # Разметка графика
    plt.title(f'Циклограмма генерации энергии на орбите {orbit_altitude_km} км\n'
              f'Площадь панелей: {panel_area_m2} м², КПД: {efficiency * 100:.1f}%')
    plt.xlabel('Время, мин')
    plt.ylabel('Мощность, кВт')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Добавляем отметки периодов
    for t in np.arange(0, mission_duration_minutes, orbit_period_minutes):
        plt.axvline(t, color='gray', linestyle=':', alpha=0.5)
        plt.text(t, np.max(power_output) / 1000 * 0.95, f'{t:.0f} мин',
                 ha='center', va='top', rotation=90, alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Вывод информации об орбите
    print(f"Период обращения: {orbit_period_minutes:.2f} минут")
    print(f"Время на свету: {light_time_minutes:.2f} минут ({light_time_minutes / orbit_period_minutes * 100:.1f}%)")
    print(f"Время в тени: {shadow_time_minutes:.2f} минут ({shadow_time_minutes / orbit_period_minutes * 100:.1f}%)")
    print(f"Средняя мощность: {np.mean(power_output) / 1000:.2f} кВт")


# Пример использования
orbit_altitude_km = 400  # высота орбиты в км
panel_area_m2 = 50  # площадь панелей в м²
efficiency = 0.3  # КПД панелей
mission_duration_minutes = 180  # продолжительность моделирования в минутах

#calculate_power_generation(orbit_altitude_km, panel_area_m2, efficiency, mission_duration_minutes)

estimate_orbital_disturbances(200000)