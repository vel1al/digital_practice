import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from pyproj import Geod


def plot_satellite_track(lons, lats, title="Трасса спутника на развёртке Земли"):
    """
    Визуализирует трассу спутника на плоской карте с учётом эллипсоидности Земли

    Параметры:
    lons - список/массив долгот спутника (в градусах)
    lats - список/массив широт спутника (в градусах)
    title - заголовок графика
    """
    # Определяем проекцию (равнопромежуточная цилиндрическая проекция)
    # Используем эллипсоид WGS84
    proj = Proj(proj='eqc', ellps='WGS84')

    # Преобразуем координаты в проекцию
    x, y = proj(lons, lats)

    # Создаем фигуру
    plt.figure(figsize=(15, 8))

    # Рисуем трассу

    plt.plot(x, y, 'b-', linewidth=1, label='Трасса спутника')
    plt.scatter(x[0], y[0], c='g', s=50, label='Начало трассы')
    plt.scatter(x[-1], y[-1], c='r', s=50, label='Конец трассы')

    # Добавляем сетку
    plt.grid(True, linestyle='--', alpha=0.7)

    # Настраиваем оси
    plt.xticks(np.arange(-180, 181, 30))
    plt.yticks(np.arange(-90, 91, 30))
    plt.xlabel('Долгота (градусы)')
    plt.ylabel('Широта (градусы)')

    # Добавляем заголовок и легенду
    plt.title(title)
    plt.legend()

    # Показываем график
    plt.tight_layout()
   # img = plt.imread("earth.jpg")
   # plt.imshow(img, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
    plt.show()


def calculate_geodetic_distance(lons, lats):
    """
    Вычисляет геодезическое расстояние между точками трассы (в метрах)
    с учётом эллипсоидности Земли
    """
    geod = Geod(ellps='WGS84')
    total_distance = 0.0

    for i in range(1, len(lons)):
        _, _, dist = geod.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])
        total_distance += dist

    return total_distance


    # Генерируем тестовые данные (трасса спутника)
np.random.seed(42)
num_points = 1000
lons = np.linspace(-180, 180, num_points)
lats = 45 * np.sin(np.linspace(0, 4 * np.pi, num_points)) + 10 * np.random.randn(num_points)

    # Визуализируем трассу
plot_satellite_track(lons, lats)

    # Вычисляем общее расстояние
distance = calculate_geodetic_distance(lons, lats)
print(f"Общее геодезическое расстояние трассы: {distance / 1000:.2f} км")