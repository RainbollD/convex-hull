import itertools
import os
from itertools import combinations


class PointsXYZ:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __sub__(self, other):
        if isinstance(other, PointsXYZ):
            return PointsXYZ(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, PointsXYZ):
            return PointsXYZ(
                self.y * other.z - self.z * other.y,
                -self.x * other.z + self.z * other.x,
                self.x * other.y - self.y * other.x
            )
        elif isinstance(other, (int, float)):
            return PointsXYZ(self.x * other, self.y * other, self.z * other)
        return NotImplemented


class PointsABCD:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c}, {self.d})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PointsABCD(self.a * other, self.b * other, self.c * other, self.d * other)
        return NotImplemented

    def __iter__(self):
        return iter((self.a, self.b, self.c, self.d))


def is_file(filepath):
    if not os.path.exists(filepath):
        print("Файл не найден")
        quit(1)


def read_coordinates(filepath):
    try:
        with open(filepath, 'r') as file:
            file_content = file.read()
        file_format = file_content[0]
        coordinates = file_content[4:].split('\n')
        points = []
        if file_format == 'V':
            points = [PointsXYZ(*list(map(int, line.strip().split()))) for line in coordinates if line.strip()]
        elif file_format == 'H':
            points = [PointsABCD(*list(map(int, line.strip().split()))) for line in coordinates if line.strip()]
        return file_format, points

    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        quit(1)


def is_one_line(coord):
    a, b, c = coord
    vect_ab = b - a
    vect_ac = c - a
    return vect_ab * vect_ac == PointsXYZ(0, 0, 0)


def transform_vector_normally(vect_normally, d):
    plane_coef = PointsABCD(vect_normally.x, vect_normally.y, vect_normally.z, d)

    if plane_coef.a < 0: return plane_coef * (-1), -1
    if plane_coef.b < 0: return plane_coef * (-1), -1
    if plane_coef.c < 0: return plane_coef * (-1), -1
    return plane_coef, 1


def count_plane_three_points(coord):
    a, b, c = coord

    vect_ab = b - a
    vect_ac = c - a

    vect_normally = vect_ab * vect_ac

    d = (vect_normally.x * a.x + vect_normally.y * a.y + vect_normally.z * a.z)

    plane_coef, do_change = transform_vector_normally(vect_normally, d)
    return plane_coef, do_change


def create_plane_formula(plane_coef):
    return lambda x, y, z: (plane_coef.a * x + plane_coef.b * y + plane_coef.c * z - plane_coef.d)


def is_one_side(plane, points, do_change):
    results = [plane(point.x, point.y, point.z) * do_change for point in points]
    if all(r >= 0 for r in results): return 1
    if all(r <= 0 for r in results): return -1
    return 0


def find_convex_hull(points):
    faces = []
    for three_points in combinations(points, 3):
        if not is_one_line(three_points):
            plane_coef, do_change = count_plane_three_points(three_points)
            plane_formula = create_plane_formula(plane_coef)
            sign_inequality = is_one_side(plane_formula, points, do_change)
            if sign_inequality != 0:
                sign = '>=' if sign_inequality * do_change >= 0 else '<='
                faces.append([plane_coef, sign])
    return faces


def create_view_plane(coefs):
    line = ''
    variables = ['x', 'y', 'z']

    for coef, var in zip(coefs[0], variables):
        if coef == 0:
            line += ''
        elif coef > 0:
            line += f"+ {coef}{var} " if coef != 1 else f"+ {var} "
        else:
            line += f"- {abs(coef)}{var} " if coef != -1 else f"- {var} "

    line += f"{coefs[1]} {coefs[0].d}"
    line = line[2:] if line[:2] == '+ ' else line

    return line


def create_view_print_planes(planes):
    line = []
    for plane in planes:
        line.append(create_view_plane(plane))
    return line


def print_planes(planes):
    planes_view = set(create_view_print_planes(planes))
    print(f"Number of faces: {len(planes_view)}")
    print("\n".join(sorted(planes_view, key=lambda x: x[-1])))


def count3x3(three_points):
    one, two, three = three_points
    return (one.a * (two.b * three.c - two.c * three.b) -
            one.b * (two.a * three.c - two.c * three.a) +
            one.c * (two.a * three.b - two.b * three.a))


def transform_float_to_int(top):
    if int(top) == float(top):
        return int(top)
    return top


def count_3x3_first_d(three_points, delta_a):
    one, two, three = three_points
    return transform_float_to_int((one.d * (two.b * three.c - two.c * three.b) -
                                   one.b * (two.d * three.c - two.c * three.d) +
                                   one.c * (two.d * three.b - two.b * three.d)) / delta_a)


def count_3x3_second_d(three_points, delta_a):
    one, two, three = three_points
    return transform_float_to_int((one.a * (two.d * three.c - two.c * three.d) -
                                   one.d * (two.a * three.c - two.c * three.a) +
                                   one.c * (two.a * three.d - two.d * three.a)) / delta_a)


def count_3x3_third_d(three_points, delta_a):
    one, two, three = three_points
    return transform_float_to_int((one.a * (two.b * three.d - two.d * three.b) -
                                   one.b * (two.a * three.d - two.d * three.a) +
                                   one.d * (two.a * three.b - two.b * three.a)) / delta_a)


def count_delta(three_points):
    delta_a = count3x3(three_points)
    delta_x = count_3x3_first_d(three_points, delta_a)
    delta_y = count_3x3_second_d(three_points, delta_a)
    delta_z = count_3x3_third_d(three_points, delta_a)
    return map(str, (delta_x, delta_y, delta_z))


def find_tops(points):
    tops = set()
    for three_points in combinations(points, 3):
        if count3x3(three_points) != 0:
            coords = count_delta(three_points)
            tops.add(" ".join(coords))
    return tops


def full_names_tops(tops):
    names_tops = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                  "U", "V", "W",
                  "X", "Y", "Z"]
    i = 0
    s = 1
    while len(names_tops) < len(tops):
        names_tops.append(names_tops[i] + str(s))
        if i % 25 == 0:
            s += 1
    return names_tops


def print_tops(tops):
    names_tops = full_names_tops(tops)
    print(f"Number of vertices: {len(tops)}")
    for name, top in zip(names_tops, tops):
        print(f"{name}: {''.join(top.strip())}")


def polyhedral_graph(faces, tops):
    """
    Строит полиэдральный граф на основе граней и вершин.

    Args:
        faces (list): Список граней (каждая грань — это объект PointsABCD).
        tops (set): Множество вершин в формате строк "x y z".

    Returns:
        dict: Граф в виде словаря смежности.
    """
    # Преобразуем tops в список кортежей с float-координатами
    points = [tuple(map(float, top.split())) for top in tops]

    # Создаем словарь для хранения индексов вершин
    vertex_index = {point: idx for idx, point in enumerate(points)}

    edges = set()

    for face in faces:
        a, b, c, d = face.a, face.b, face.c, face.d

        # Находим вершины, лежащие на этой грани
        vertices_on_face = []
        for point in points:
            x, y, z = point
            if abs(a * x + b * y + c * z - d) < 1e-6:
                vertices_on_face.append(vertex_index[point])

        # Соединяем вершины в цикле
        n = len(vertices_on_face)
        for j in range(n):
            v1 = vertices_on_face[j]
            v2 = vertices_on_face[(j + 1) % n]
            edges.add(frozenset({v1, v2}))

    # Строим граф
    graph = {i: [] for i in range(len(points))}
    for edge in edges:
        v1, v2 = edge
        graph[v1].append(v2)
        graph[v2].append(v1)

    return graph


def print_first_tops_graph(names_tops):
    print(f'  {'  '.join(names_tops)}')


def print_intersections_graph(name_top, graph_vertex, graph_neighbor, count_vertex):
    print(name_top, end=' ')
    graph_neighbor.sort()
    for i in range(count_vertex):
        print('1' if i != graph_vertex and i in graph_neighbor else '0', end='  ')
    print()


def print_adjacency_matrix(graph):
    print("Adjacency matrix")
    names_tops = full_names_tops(graph)[:len(graph)]
    print_first_tops_graph(names_tops)

    for name_top, graph_branch in zip(names_tops, graph.items()):
        graph_vertex, graph_neighbor = graph_branch
        print_intersections_graph(name_top, graph_vertex, graph_neighbor, len(graph))


def print_vertex_edge_graph(faces, graph):
    for face, graph_branch in zip(faces, graph.items()):
        graph_vertex, graph_neighbor = graph_branch
        graph_neighbor = [chr(ord('A') + num) for num in graph_neighbor]


        edges = itertools.combinations(graph_neighbor, 2)
        edge_strings = [''.join(edge) for edge in edges]

        print(f"Face: {create_view_plane(face)}")
        print(f"Vertices: {', '.join(graph_neighbor)}")
        print(f"Edges: {', '.join(edge_strings)}")

def control_file_format_v(points):
    faces = find_convex_hull(points)
    print_planes(faces)

    tops = find_tops([coef[0] for coef in faces])
    print_tops(tops)

    graph = polyhedral_graph([coef[0] for coef in faces], tops)
    print_adjacency_matrix(graph)
    print_vertex_edge_graph(faces, graph)


def control_file_format_h(points):
    faces = [[p, "<="] for p in points]

    tops = find_tops(points)
    print_tops(tops)

    graph = polyhedral_graph([coef[0] for coef in faces], tops)
    print_adjacency_matrix(graph)
    print_vertex_edge_graph(faces, graph)


def main():
    filepath = r'tets/test_3_octahedron.txt'
    is_file(filepath)

    file_format, points = read_coordinates(filepath)

    if file_format == 'V':
        control_file_format_v(points)
    elif file_format == 'H':
        control_file_format_h(points)


#    polyhedral_graph(faces, tops)


if __name__ == '__main__':
    main()
