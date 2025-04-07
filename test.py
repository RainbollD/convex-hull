import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Создаем данные вручную
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
z = [0, 1, 8, 27, 64]

# Создаем фигуру и 3D-осевую систему
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Соединяем точки и заполняем поверхности
ax.plot_trisurf(x, y, z, color='cyan', alpha=0.5)

# Рисуем точки для наглядности
ax.scatter(x, y, z, color='b', marker='o')

# Устанавливаем метки осей
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Отображаем график
plt.show()
