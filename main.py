import os
import argparse
from itertools import combinations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

    def __eq__(self, other):
        if isinstance(other, PointsXYZ):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def to_tuple(self):
        return (self.x, self.y, self.z)


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

    def __eq__(self, other):
        if isinstance(other, PointsABCD):
            return (self.a, self.b, self.c, self.d) == (other.a, other.b, other.c, other.d)
        elif isinstance(other, (tuple, list)) and len(other) == 4:
            return (self.a, self.b, self.c, self.d) == tuple(other)
        return False

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d))

    def to_tuple(self):
        return (self.a, self.b, self.c, self.d)


def is_file(filepaths):
    """Checking for file"""
    works_filepath = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Файл {filepath} не найден.\n")
            continue
        works_filepath.append(filepath)
    return works_filepath


def read_coordinates(filepath):
    """
    Functions:
    1) Read data from a file
    2) Write points to PointXYZ if they're coordinates and to PointABCD - plane coefficients
    :param filepath: file path
    :return: file format (V, H), class PointXYZ/PointABCD with points
    """
    try:
        with open(filepath, 'r') as file:
            file_content = file.read()

        file_format = file_content[0]
        coordinates = file_content[4:].split('\n')

        if file_format == 'V':
            points = [PointsXYZ(*list(map(int, line.strip().split()))) for line in coordinates if line.strip()]
        elif file_format == 'H':
            points = [PointsABCD(*list(map(int, line.strip().split()))) for line in coordinates if line.strip()]
        else:
            print(f"Формат вида '{file_format}' не обрабатывается.\n"
                  f"Проверьте корректность данных в файле!\n"
                  f"Доступные форматы: V, H")
            quit(0)
        return file_format, points

    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        quit(0)


def is_one_line(coord):
    """Check whether the vectors line on the same line"""
    a, b, c = coord
    vect_ab = b - a
    vect_ac = c - a
    return vect_ab * vect_ac == PointsXYZ(0, 0, 0)


def count_plane_three_points(coord):
    """Сoefficient calculation for the plane"""
    a, b, c = coord

    vect_ab = b - a
    vect_ac = c - a

    vect_normally = vect_ab * vect_ac

    d = (vect_normally.x * a.x + vect_normally.y * a.y + vect_normally.z * a.z)

    return PointsABCD(vect_normally.x, vect_normally.y, vect_normally.z, d)


def create_plane_formula(plane_coef):
    return lambda x, y, z: (plane_coef.a * x + plane_coef.b * y + plane_coef.c * z - plane_coef.d)


def is_one_side(plane, points):
    """Check if the points are on the same side"""
    results = [plane(point.x, point.y, point.z) for point in points]

    if all(r >= 0 for r in results): return 1
    if all(r <= 0 for r in results): return -1

    return 0


def is_double_planes(plane_coef, faces):
    """Check if the same plane in list"""
    for face in faces:
        if plane_coef == face: return True
    return False


def find_convex_hull(points):
    """
    Finding the convex hull.
    Functions:
    1) Construct combinations of 3 points
    2) Calculate the plane
    3) Check the side for points
    :param points: points in the space
    :return: list of planes coefficients'
    """
    faces = []

    for three_points in combinations(points, 3):
        if not is_one_line(three_points):

            plane_coef = count_plane_three_points(three_points)
            plane_formula = create_plane_formula(plane_coef)
            sign_inequality = is_one_side(plane_formula, points)

            if sign_inequality == 0: continue

            if sign_inequality >= 0:
                plane_coef *= -1

            if not is_double_planes(plane_coef, faces):
                faces.append(plane_coef)

    return faces


def create_view_plane(coefs):
    """Creating a beautiful mapping for a plane of type x + y + z <= 0"""
    line = ''
    variables = ['x', 'y', 'z']

    for coef, var in zip(coefs, variables):
        if coef == 0:
            continue
        elif coef > 0:
            line += f"+ {coef}{var} " if coef != 1 else f"+ {var} "
        else:
            line += f"- {abs(coef)}{var} " if coef != -1 else f"- {var} "

    line += f"<= {coefs.d}"
    line = line[2:] if line.startswith('+ ') else line

    return line.strip()


def create_list_view_planes(planes):
    """Create list of beautiful mapping for planes"""
    line = []
    for plane in planes:
        line.append(create_view_plane(plane))
    return line


def print_planes(planes):
    planes_view = create_list_view_planes(planes)
    print(f"Number of faces: {len(planes_view)}")
    print("\n".join(sorted(planes_view, key=lambda x: x[-1])), end='\n\n')


def transform_float_to_int(top):
    if int(top) == float(top):
        return int(top)
    return top


def count3x3(three_points):
    """Calculating det for a 3X3 matrix"""
    one, two, three = three_points
    return (one.a * (two.b * three.c - two.c * three.b) -
            one.b * (two.a * three.c - two.c * three.a) +
            one.c * (two.a * three.b - two.b * three.a))


def count_3x3_first_d(three_points, delta_a):
    """Calculating det for a 3X3 matrix.
     First column is d"""
    one, two, three = three_points
    return transform_float_to_int((one.d * (two.b * three.c - two.c * three.b) -
                                   one.b * (two.d * three.c - two.c * three.d) +
                                   one.c * (two.d * three.b - two.b * three.d)) / delta_a)


def count_3x3_second_d(three_points, delta_a):
    """Calculating det for a 3X3 matrix.
     Second column is d"""
    one, two, three = three_points
    return transform_float_to_int((one.a * (two.d * three.c - two.c * three.d) -
                                   one.d * (two.a * three.c - two.c * three.a) +
                                   one.c * (two.a * three.d - two.d * three.a)) / delta_a)


def count_3x3_third_d(three_points, delta_a):
    """Calculating det for a 3X3 matrix.
     Third column is d"""
    one, two, three = three_points
    return transform_float_to_int((one.a * (two.b * three.d - two.d * three.b) -
                                   one.b * (two.a * three.d - two.d * three.a) +
                                   one.d * (two.a * three.b - two.b * three.a)) / delta_a)


def count_delta(three_points):
    """Calculate the coordinates of the vertex"""
    delta_a = count3x3(three_points)

    if delta_a == 0:
        return None

    delta_x = count_3x3_first_d(three_points, delta_a)
    delta_y = count_3x3_second_d(three_points, delta_a)
    delta_z = count_3x3_third_d(three_points, delta_a)

    return (delta_x, delta_y, delta_z)


def is_vertex_satisfies_inequalities(vertex, points):
    for plane in points:
        if plane.a * vertex.x + plane.b * vertex.y + plane.c * vertex.z > plane.d + 1e-8:
            return False
    return True


def find_vertices(points):
    """
    Finding the vertices.
    Functions:
    1) Construct combinations of 3 points
    2) Calculate the vertices
    :param points: points in the space
    :return: set of vertices'
    """
    vertices = set()

    for three_points in combinations(points, 3):
        coords = count_delta(three_points)
        if coords is not None:
            vertex = PointsXYZ(*coords)

            if is_vertex_satisfies_inequalities(vertex, points):
                vertices.add(vertex)

    return sorted(vertices, key=lambda v: (v.x, v.y, v.z))


def full_names_vertices(vertices):
    """Connect the letters to the vertex coordinates"""
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    result = []
    for i, vertex in enumerate(vertices):
        if i < 26:
            result.append((names[i], vertex))
        else:
            result.append((f"{names[i % 26]}{i // 26}", vertex))
    return result


def print_vertices(vertices):
    named_vertices = full_names_vertices(vertices)
    print(f"Number of vertices: {len(named_vertices)}")
    for name, vertex in named_vertices:
        print(f"{name}: {vertex.x} {vertex.y} {vertex.z}")
    print()
    return named_vertices


def is_vertex_on_face(vertex, face):
    """Check if the vertex is on the face"""
    return abs(face.a * vertex.x + face.b * vertex.y + face.c * vertex.z - face.d) < 1e-8


def build_adjacency_matrix(vertices, faces):
    """
    Build adjacency matrix
    Idea: if two vertices are on two the same faces, then they have an edge
    :param vertices: List format: [(vertice,(x,y,z)), ...] - name vertice and coordinates
    :param faces: coefficients of plane
    :return: binary matrix, where 1 if there's the general plane else 0
    """

    n = len(vertices)
    adj = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            shared_faces = 0
            for face in faces:
                if is_vertex_on_face(vertices[i][1], face) and is_vertex_on_face(vertices[j][1], face):
                    shared_faces += 1
                    if shared_faces > 2:
                        break
            if shared_faces == 2:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj


def print_adjacency_matrix(vertices, adj_matrix):
    print("Adjacency matrix")

    names = [v[0] for v in vertices]

    print("  " + " ".join(names))

    for i in range(len(vertices)):
        row = [str(adj_matrix[i][j]) for j in range(len(vertices))]
        print(names[i] + " " + " ".join(row))
    print()


def find_face_vertices(vertices, face):
    face_vertices = []

    for name, vertex in vertices:
        if is_vertex_on_face(vertex, face):
            face_vertices.append(name)
    return face_vertices


def find_edges(face_vertices):
    edges = []

    for i in range(len(face_vertices)):
        for j in range(i + 1, len(face_vertices)):
            edges.append(f"{face_vertices[i]}{face_vertices[j]}")

    return edges


def print_face_info(faces, vertices):
    for face in faces:
        face_vertices = find_face_vertices(vertices, face)
        edges = find_edges(face_vertices)

        print(f"Face: {create_view_plane(face)}")
        print(f"Vertices: {', '.join(face_vertices)}")
        print(f"Edges: {', '.join(edges)}")
        print()


def print_collides(collides):
    print("\nCollision detection:")
    print("The polyhedrons", "intersect" if collides else "do not intersect")


def minkowski_difference(poly1, poly2):
    collides = True

    if poly1 == None or poly2 == None: return

    all_faces = poly1 + poly2

    for face in all_faces:
        if face.a * 0 + face.b * 0 + face.c * 0 > face.d + 1e-8:
            collides = False

    print_collides(collides)


def get_face_vertices(named_vertices, adj_matrix):
    """
    Function connects vertices in one list
    :param named_vertices:
    :param adj_matrix:
    :return:
    """

    face_vertices = []

    for index_adj_matrix, row in enumerate(adj_matrix):
        for index, connection in enumerate(row):
            if connection == 1:
                face_vertices.append([named_vertices[index_adj_matrix][1], named_vertices[index][1]])
    return face_vertices


def plot_polyhedron(vertices, named_vertices, ax, adj_matrix, alpha=0):
    """Figure visualization"""
    face_verts = get_face_vertices(named_vertices, adj_matrix)

    for fv in face_verts:
        fv = [element.to_tuple() for element in fv]
        poly = Poly3DCollection([fv], alpha=alpha, linewidths=1, edgecolors='k')
        poly.set_facecolor('blue')
        ax.add_collection3d(poly)

    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    zs = [v.z for v in vertices]
    ax.scatter(xs, ys, zs, color='blue', s=25)

    for vert in named_vertices:
        label = f"{vert[0]}({vert[1].x}, {vert[1].y}, {vert[1].z})"
        ax.text(vert[1].x, vert[1].y, vert[1].z, label, color='purple', fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.5
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    mid_z = (max_z + min_z) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def preparation_plot_polyhedron(vertices, faces, named_vertices, adj_matrix):
    """Create environment for figure"""
    for vertice, face, named_vertice in zip(vertices, faces, named_vertices):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_polyhedron(vertice,  named_vertices=named_vertice, ax=ax, adj_matrix=adj_matrix)
        plt.title("3D")
        plt.show()


def process_file(filepath):
    """
    Function:
    1) Control actions for different file format
    2) Finding and printing *convex hull
                            *vertices
                            *adjacency matrix
    :param filepath:
    :return: faces, vertices <type: list>
    """

    file_format, points = read_coordinates(filepath)

    print(f"Actions with file at the path: {filepath}\n"
          f"\nFile format: {file_format}\n")

    if file_format == 'V':
        faces = find_convex_hull(points)
        print_planes(faces)
    elif file_format == 'H':
        faces = points

    vertices = find_vertices(faces)
    named_vertices = print_vertices(vertices)

    adj_matrix = build_adjacency_matrix(named_vertices, faces)
    print_adjacency_matrix(named_vertices, adj_matrix)
    print_face_info(faces, named_vertices)
    return faces, vertices, named_vertices, adj_matrix


def get_info_console():
    """Get list of files name's"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Введите файлы")
    args = parser.parse_args()
    return args.input


def main():
    """
    Main control function
    1) make action with 1 or 2 files
    2) calculating minkowski difference for 2 files
    3) Visualization figures
    :return:
    """
    string_console_data = ["tets/test_4_H_tetrahedron.txt"]  # get_info_console()

    filepaths = is_file(string_console_data)

    if len(filepaths) == 2:
        filepath = filepaths[0]
        faces1, vertices1, named_vertices1, adj_matrix = process_file(filepath)

        filepath = filepaths[1]
        faces2, vertices2, named_vertices2, adj_matrix = process_file(filepath)

        minkowski_difference(faces1, faces2)

        vertices = [vertices1, vertices2]
        faces = [faces1, faces2]
        named_vertices = [named_vertices1, named_vertices2]

    elif len(filepaths) == 1:
        filepath = filepaths[0]
        faces1, vertices1, named_vertices, adj_matrix = process_file(filepath)

        vertices = [vertices1]
        faces = [faces1]
        named_vertices = [named_vertices]

    else:
        print(f"Неправильные данные.\n"
              f"Количество путей к файлу должно быть: 1 или 2")
        quit(0)

    preparation_plot_polyhedron(vertices, faces, named_vertices, adj_matrix)


if __name__ == '__main__':
    main()
