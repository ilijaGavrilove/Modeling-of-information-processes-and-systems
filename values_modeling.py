import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm

# Параметры модели (вариант 4)
mu = 398600  # гравитационная постоянная Земли, км³/с²
Rz = 6371  # радиус Земли, км
xn, yn = 2158.1, 5994.6  # координаты станции, км
delta_h = 2  # начальная ошибка высоты, км
sigma_D = 0.070  # СКО дальности, км (70 м)
sigma_dD = 0.0007  # СКО радиальной скорости, км/с (0.7 м/с)
dt_meas = 4  # шаг измерений, сек

# Начальные условия реального движения
x0_real = 0
y0_real = 6671  # высота 300 км (6671 - 6371)
vx0_real = 7.72989
vy0_real = 0

# Начальные условия для модели (с ошибкой)
x0_model = 0
y0_model = 6673  # +2 км ошибки
vx0_model = 7.72989
vy0_model = 0


# Функция для вычисления производных (уравнения движения)
def orbit_dynamics(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    ax = -mu * x / r ** 3
    ay = -mu * y / r ** 3
    return [vx, vy, ax, ay]


# Функция для вычисления измерений
def calculate_measurements(x, y, vx, vy):
    # Разности координат
    dx = x - xn
    dy = y - yn

    # Расстояние до станции
    D = np.sqrt(dx ** 2 + dy ** 2)

    # Радиальная скорость
    dD = (dx * vx + dy * vy) / D if D > 0 else 0

    return D, dD


# Время моделирования (один виток ~90 мин = 5400 сек)
t_span = (0, 6000)
t_eval = np.arange(0, t_span[1], 1)  # шаг 1 сек

# 1. Моделирование реального движения
sol_real = solve_ivp(orbit_dynamics, t_span,
                     [x0_real, y0_real, vx0_real, vy0_real],
                     t_eval=t_eval,
                     method='DOP853',
                     rtol=1e-10, atol=1e-10)

# 2. Моделирование движения с ошибкой начальной высоты
sol_model = solve_ivp(orbit_dynamics, t_span,
                      [x0_model, y0_model, vx0_model, vy0_model],
                      t_eval=t_eval,
                      method='DOP853',
                      rtol=1e-10, atol=1e-10)

# 3. Вычисление измерений
# Для реального движения (без шума)
D_real = []
dD_real = []
for i in range(len(sol_real.t)):
    D, dD = calculate_measurements(sol_real.y[0, i], sol_real.y[1, i],
                                   sol_real.y[2, i], sol_real.y[3, i])
    D_real.append(D)
    dD_real.append(dD)

# Для модели с шумом (только в моменты измерений)
meas_times = np.arange(0, t_span[1], dt_meas)
D_meas = []
dD_meas = []
D_clean = []  # измерения дальности без шума для сравнения
dD_clean = []  # добавлен расчет чистых значений радиальной скорости

# Найдем ближайшие точки в sol_model для моментов измерений
for t in meas_times:
    idx = np.abs(sol_model.t - t).argmin()
    x, y, vx, vy = sol_model.y[0, idx], sol_model.y[1, idx], sol_model.y[2, idx], sol_model.y[3, idx]

    D, dD = calculate_measurements(x, y, vx, vy)
    D_clean.append(D)
    dD_clean.append(dD)  # сохраняем чистую радиальную скорость

    # Добавляем шум
    D_noisy = D + norm.rvs(0, sigma_D)
    dD_noisy = dD + norm.rvs(0, sigma_dD)

    D_meas.append(D_noisy)
    dD_meas.append(dD_noisy)

# 4. Построение графиков
plt.figure(figsize=(15, 12))

# График относительной дальности
plt.subplot(2, 1, 1)
plt.plot(sol_real.t, D_real, 'b-', label='Реальное движение (без шума)', linewidth=1.5)
plt.plot(meas_times, D_clean, 'g--', label='Модель с ошибкой (без шума)', linewidth=1.5)
plt.plot(meas_times, D_meas, 'ro', markersize=4, label='Измерения с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Относительная дальность, км')
plt.title('Относительная дальность ИСЗ до станции слежения')
plt.grid(True)
plt.legend()
plt.ylim(min(D_real) * 0.95, max(D_real) * 1.05)

# График радиальной скорости
plt.subplot(2, 1, 2)
plt.plot(sol_real.t, dD_real, 'b-', label='Реальное движение (без шума)', linewidth=1.5)
plt.plot(meas_times, dD_clean, 'g--', label='Модель с ошибкой (без шума)', linewidth=1.5)  #
plt.plot(meas_times, dD_meas, 'ro', markersize=4, label='Измерения с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Радиальная скорость, км/с')
plt.title('Радиальная скорость ИСЗ относительно станции')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 5. Дополнительные графики для анализа
plt.figure(figsize=(15, 10))

# Траектории движения
plt.subplot(2, 2, 1)
plt.plot(sol_real.y[0], sol_real.y[1], 'b-', label='Реальная орбита (круговая)')
plt.plot(sol_model.y[0], sol_model.y[1], 'r--', label='Модель с ошибкой (эллиптическая)')
plt.plot(xn, yn, 'go', markersize=8, label='Станция слежения')
plt.xlabel('X, км')
plt.ylabel('Y, км')
plt.title('Траектории движения ИСЗ')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Высоты полета
h_real = np.sqrt(sol_real.y[0] ** 2 + sol_real.y[1] ** 2) - Rz
h_model = np.sqrt(sol_model.y[0] ** 2 + sol_model.y[1] ** 2) - Rz

plt.subplot(2, 2, 2)
plt.plot(sol_real.t, h_real, 'b-', label='Реальная высота')
plt.plot(sol_model.t, h_model, 'r--', label='Модель с ошибкой')
plt.xlabel('Время, сек')
plt.ylabel('Высота, км')
plt.title('Высота полета ИСЗ')
plt.grid(True)
plt.legend()

# Радиальная скорость вблизи нуля
plt.subplot(2, 2, 3)
plt.plot(sol_real.t, dD_real, 'b-', label='Реальная')
plt.plot(meas_times, dD_clean, 'g--', label='Модель (без шума)')  #
plt.plot(meas_times, dD_meas, 'ro', markersize=4, label='Измерения')
plt.xlabel('Время, сек')
plt.ylabel('Радиальная скорость, км/с')
plt.title('Радиальная скорость (увеличенный масштаб)')
plt.grid(True)
plt.ylim(-0.5, 0.5)
plt.legend()

# Гистограмма ошибок измерений
errors_D = [D_m - D_c for D_m, D_c in zip(D_meas, D_clean)]
errors_dD = [dD_m - dD_c for dD_m, dD_c in zip(dD_meas, dD_clean)] 

plt.subplot(2, 2, 4)
plt.hist(errors_D, bins=20, alpha=0.7, label=f'Ошибки D (σ={sigma_D:.4f} км)')
plt.hist(errors_dD, bins=20, alpha=0.7, label=f'Ошибки dD/dt (σ={sigma_dD:.6f} км/с)')
plt.xlabel('Ошибка измерения')
plt.ylabel('Частота')
plt.title('Распределение ошибок измерений')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()