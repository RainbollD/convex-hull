import os
from itertools import combinations

from scipy.constants import point


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
        return NotImplemented


class PointsABCD:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c}, {self.d})"


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


def count_plane_three_points(coord):
    a, b, c = coord
    vect_ab = b - a
    vect_ac = c - a
    vect_normally = vect_ab * vect_ac
    D = -(vect_normally.x * a.x + vect_normally.y * a.y + vect_normally.z * a.z)
    return lambda x, y, z: (vect_normally.x * x + vect_normally.y * y + vect_normally.z * z + D), vect_normally, D


def is_one_side(plane, points):
    results = [plane(point.x, point.y, point.z) for point in points]
    if all(r >= 0 for r in results): return 1
    if all(r <= 0 for r in results): return -1
    return 0


def find_convex_hull(points):
    faces = []
    for three_points in combinations(points, 3):
        if not is_one_line(three_points):
            plane, vect_normally, D = count_plane_three_points(three_points)
            sign_inequality = is_one_side(plane, points)
            if sign_inequality != 0:
                faces.append([plane, vect_normally, D, sign_inequality])
    return faces


def get_presentation_with_xyz(vector, var):
    if vector == 1: return var
    if vector == -1: return '-' + var
    if vector == 0: return ''
    return str(vector) + var


def do_change_signs(ax, by, cz):
    if ((ax != '' and ax[0] == '-') or (ax == '' and by != '' and by[0] == '-')
            or (ax == '' and by == '' and cz != '' and cz[0] == '-')): return True, -1
    return False, 1


def get_print_plane(data_plane):
    vect_normally, D, sign_inequality = data_plane[1:]

    ax = get_presentation_with_xyz(vect_normally.x, 'x')
    by = get_presentation_with_xyz(vect_normally.y, 'y')
    cz = get_presentation_with_xyz(vect_normally.z, 'z')

    line = []
    change, int_change = do_change_signs(ax, by, cz)

    signs_math_expression = ['- ', '+ '] if change else ['+ ', '- ']

    add_sign = lambda expr: signs_math_expression[0] + expr if coefs[0] != '-' else signs_math_expression[1] + expr[1:]

    for coefs in (ax, by, cz):
        if coefs == '': continue
        if len(line):
            coefs = add_sign(coefs)
        else:
            coefs = coefs[-1]
        line.append(coefs)

    line.append(">=" if sign_inequality * int_change > 0 else "<=")
    line.append(str(D * (-1) * int_change))
    return ' '.join(line)


def print_correct_planes(faces):
    planes = []

    def get_list_all_planes():
        for data_plane in faces:
            planes.append(get_print_plane(data_plane))

    def change_if_first_minus():
        for idx, plane in enumerate(planes):
            if plane[0] == '-':
                plane = plane.replace("+", "-").replace("-", "+").replace(">=", "<=")
                planes[idx] = plane

    get_list_all_planes()
    change_if_first_minus()
    planes = list(set(planes))
    planes.sort()
    print(f"Number of faces: {len(planes)}")
    print("\n".join(planes))


def main():
    filepath = r'tets/new_test.txt'
    is_file(filepath)

    file_format, points = read_coordinates(filepath)

    if file_format == 'V':
        faces = find_convex_hull(points)
        print_correct_planes(faces)
    elif file_format == 'H':
        print(points)


if __name__ == '__main__':
    main()
