import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
import os

# Создаем папку для сохранения графиков
output_dir = "corrected_kalman_plots"
os.makedirs(output_dir, exist_ok=True)

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
print("Моделирование реального движения...")
sol_real = solve_ivp(orbit_dynamics, t_span,
                     [x0_real, y0_real, vx0_real, vy0_real],
                     t_eval=t_eval,
                     method='DOP853',
                     rtol=1e-10, atol=1e-10)

# 2. Моделирование движения с ошибкой начальной высоты
print("Моделирование движения с ошибкой...")
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
dD_clean = []  # измерения радиальной скорости без шума

# Найдем ближайшие точки в sol_model для моментов измерений
for t in meas_times:
    idx = np.abs(sol_model.t - t).argmin()
    x, y, vx, vy = sol_model.y[0, idx], sol_model.y[1, idx], sol_model.y[2, idx], sol_model.y[3, idx]

    D, dD = calculate_measurements(x, y, vx, vy)
    D_clean.append(D)
    dD_clean.append(dD)

    # Добавляем шум
    D_noisy = D + norm.rvs(0, sigma_D)
    dD_noisy = dD + norm.rvs(0, sigma_dD)

    D_meas.append(D_noisy)
    dD_meas.append(dD_noisy)

# 4. КОРРЕКТНАЯ реализация фильтра Калмана
print("Реализация фильтра Калмана...")

# Инициализация фильтра
x_est = np.array([0, 6673, 7.72989, 0])  # начальная оценка состояния
P = np.diag([delta_h ** 2 / 9, delta_h ** 2 / 9, 25e-6, 25e-6])  # начальная ковариация
Q = np.diag([1e-8, 1e-8, 1e-10, 1e-10])  # небольшая матрица шумов процесса для устойчивости
R = np.diag([sigma_D ** 2, sigma_dD ** 2])  # матрица шумов измерений

# Массивы для сохранения результатов
h_true_arr = []  # реальная высота
h_est_arr = []  # оцененная высота
error_h_arr = []  # ошибка оценки высоты
errors_D_arr = []  # ошибки измерений дальности
errors_dD_arr = []  # ошибки измерений радиальной скорости
P_diag_pos_arr = []  # диагональные элементы P для положения
P_diag_vel_arr = []  # диагональных элементов P для скорости
time_arr = []  # время измерений

# Основной цикл обработки измерений
for i, t in enumerate(meas_times):
    # 1. Находим реальную высоту в момент измерения
    idx_real = np.abs(sol_real.t - t).argmin()
    x_real = sol_real.y[0, idx_real]
    y_real = sol_real.y[1, idx_real]
    h_true = np.sqrt(x_real ** 2 + y_real ** 2) - Rz
    h_true_arr.append(h_true)

    # 2. Прогноз состояния
    r = np.sqrt(x_est[0] ** 2 + x_est[1] ** 2)
    a = mu / r ** 3

    # Матрица перехода
    F = np.array([
        [1, 0, dt_meas, 0],
        [0, 1, 0, dt_meas],
        [-a * dt_meas, 0, 1, 0],
        [0, -a * dt_meas, 0, 1]
    ])

    # Прогноз состояния
    x_pred = F @ x_est

    # Прогноз ковариации
    P_pred = F @ P @ F.T + Q

    # 3. Подготовка к коррекции
    # Вычисляем ожидаемые измерения
    dx = x_pred[0] - xn
    dy = x_pred[1] - yn
    D_pred = np.sqrt(dx ** 2 + dy ** 2)

    # Проверка деления на ноль
    if D_pred < 1e-6:
        D_pred = 1e-6

    # ПРАВИЛЬНАЯ матрица измерений
    H = np.zeros((2, 4))

    # Производные по положению для D
    H[0, 0] = dx / D_pred
    H[0, 1] = dy / D_pred

    # Производные для dD
    Vx = x_pred[2]
    Vy = x_pred[3]
    dot_product = dx * Vx + dy * Vy
    dD_pred = dot_product / D_pred

    # Корректные производные для радиальной скорости
    H[1, 0] = (Vx * D_pred - dx * dD_pred) / D_pred ** 2
    H[1, 1] = (Vy * D_pred - dy * dD_pred) / D_pred ** 2
    H[1, 2] = dx / D_pred
    H[1, 3] = dy / D_pred

    # Вектор измерений
    z = np.array([D_meas[i], dD_meas[i]])
    z_pred = np.array([D_pred, dD_pred])

    # Невязка измерений
    innov = z - z_pred

    # 4. Коррекция
    S = H @ P_pred @ H.T + R

    # Регуляризация для устойчивости
    S_inv = np.linalg.inv(S + 1e-8 * np.eye(2))
    K = P_pred @ H.T @ S_inv

    x_est = x_pred + K @ innov

    # Ограничение: высота не может быть отрицательной
    r_est = np.sqrt(x_est[0] ** 2 + x_est[1] ** 2)
    if r_est < Rz:
        # Если оценка ниже поверхности Земли, корректируем
        scale = (Rz + 300) / r_est  # возвращаем на номинальную высоту
        x_est[0] *= scale
        x_est[1] *= scale

    # Обновление ковариации
    I = np.eye(4)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

    # 5. Сохранение результатов
    h_est = np.sqrt(x_est[0] ** 2 + x_est[1] ** 2) - Rz
    h_est_arr.append(h_est)
    error_h_arr.append(h_est - h_true)
    errors_D_arr.append(D_meas[i] - D_clean[i])
    errors_dD_arr.append(dD_meas[i] - dD_clean[i])
    P_diag_pos_arr.append(np.diag(P)[:2])  # дисперсии положения
    P_diag_vel_arr.append(np.diag(P)[2:])  # дисперсии скорости
    time_arr.append(t)

# Преобразуем в массивы NumPy для удобства
h_true_arr = np.array(h_true_arr)
h_est_arr = np.array(h_est_arr)
error_h_arr = np.array(error_h_arr)
errors_D_arr = np.array(errors_D_arr)
errors_dD_arr = np.array(errors_dD_arr)
P_diag_pos_arr = np.array(P_diag_pos_arr)
P_diag_vel_arr = np.array(P_diag_vel_arr)
time_arr = np.array(time_arr)

# 5. Построение и сохранение графиков

# График 1: Реальная и оцененная высота
plt.figure(figsize=(10, 6))
plt.plot(time_arr, h_true_arr, 'b-', linewidth=2, label='Реальная высота')
plt.plot(time_arr, h_est_arr, 'r--', linewidth=1.5, label='Оценка фильтра Калмана')
plt.xlabel('Время, сек')
plt.ylabel('Высота, км')
plt.title('Реальная и оцененная высота полета ИСЗ')
plt.legend()
plt.grid(True)
plt.ylim(290, 310)  # Фиксированный масштаб для наглядности
plt.savefig(os.path.join(output_dir, '1_height_comparison.png'), dpi=300)
plt.close()

# График 2: Ошибка оценивания высоты
plt.figure(figsize=(10, 6))
plt.plot(time_arr, error_h_arr, 'g-', linewidth=1.5)
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, км')
plt.title('Ошибка оценки высоты полета ИСЗ')
plt.grid(True)
plt.ylim(-0.1, 2.2)  # Правильный масштаб
plt.savefig(os.path.join(output_dir, '2_height_error.png'), dpi=300)
plt.close()

# График 3: Ошибки измерений дальности (ЛИНИЯ с точками)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, errors_D_arr, 'm-', linewidth=1, marker='o', markersize=3, label='Ошибки')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=3 * sigma_D, color='r', linestyle='--', alpha=0.5, label='±3σ')
plt.axhline(y=-3 * sigma_D, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, км')
plt.title('Ошибки измерений дальности')
plt.grid(True)
plt.ylim(-3 * sigma_D, 3 * sigma_D)
plt.legend()
plt.savefig(os.path.join(output_dir, '3_range_errors.png'), dpi=300)
plt.close()

# График 4: Ошибки измерений радиальной скорости (ЛИНИЯ с точками)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, errors_dD_arr, 'c-', linewidth=1, marker='o', markersize=3, label='Ошибки')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=3 * sigma_dD, color='r', linestyle='--', alpha=0.5, label='±3σ')
plt.axhline(y=-3 * sigma_dD, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, км/с')
plt.title('Ошибки измерений радиальной скорости')
plt.grid(True)
plt.ylim(-3 * sigma_dD, 3 * sigma_dD)
plt.legend()
plt.savefig(os.path.join(output_dir, '4_velocity_errors.png'), dpi=300)
plt.close()

# График 5: Дисперсии оценки положения
plt.figure(figsize=(10, 6))
plt.semilogy(time_arr, P_diag_pos_arr[:, 0], 'b-', label='Дисперсия x')
plt.semilogy(time_arr, P_diag_pos_arr[:, 1], 'r-', label='Дисперсия y')
plt.xlabel('Время, сек')
plt.ylabel('Дисперсия, км²')
plt.title('Дисперсии оценки положения')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '5_position_variances.png'), dpi=300)
plt.close()

# График 6: Дисперсии оценки скорости
plt.figure(figsize=(10, 6))
plt.semilogy(time_arr, P_diag_vel_arr[:, 0], 'b-', label='Дисперсия Vx')
plt.semilogy(time_arr, P_diag_vel_arr[:, 1], 'r-', label='Дисперсия Vy')
plt.xlabel('Время, сек')
plt.ylabel('Дисперсия, (км/с)²')
plt.title('Дисперсии оценки скорости')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '6_velocity_variances.png'), dpi=300)
plt.close()

print(f"Все 6 графиков успешно сохранены в папке '{output_dir}'")