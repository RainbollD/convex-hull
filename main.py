import os
from itertools import combinations
import matplotlib.pyplot as plt


class Points:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __sub__(self, other):
        if isinstance(other, Points):
            return Points(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Points):
            return Points(
                self.y * other.z - self.z * other.y,
                - self.x * other.z + self.z * other.x,
                self.x * other.y - self.y * other.x
            )
        return NotImplemented


def is_file(filepath):
    if not os.path.exists(filepath):
        print("Файл не найден")
        quit(1)


def read_coordinates(filepath):
    try:
        with open(filepath, 'r') as file:
            file = file.read()
        file_format = file[0]
        coordinates = file[4:].split('\n')
        points = [Points(*list(map(int, line.strip().split()))) for line in coordinates]

        return file_format, points
    except:
        pass


def is_one_line(coord):
    a, b, c = coord
    if c - a == b - a:
        return True
    return False


def count_plane_three_points(coord):
    a, b, c = coord
    vect_ab = b - a
    vect_ac = c - a
    vect_normally = vect_ab * vect_ac
    return lambda x, y, z: (vect_normally.x * (x - c.x) +
                            vect_normally.y * (y - c.y) +
                            vect_normally.z * (z - c.z)
                            ), vect_normally
    # D = -(N.x * a.x + N.y * a.y + N.z * a.z)

def is_one_side(plane, points):
    if all(plane(point.x, point.y, point.z) for point in points)>=0:
        return True
    elif all(plane(point.x, point.y, point.z) for point in points)<=0:
        return True
    return False

def find_convex_hull(points):
    faces = []
    for three_points in combinations(points, 3):
        if not is_one_line(three_points):
            plane = count_plane_three_points(three_points)
            if is_one_side(plane[0], points):
                print(three_points, plane[1])
                faces.append(plane[1])

def surface_plot(points):
    x = [point.x for point in points]
    y = [point.y for point in points]
    z = [point.z for point in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(list(x,y,z))

    plt.show()


def main():
    filepath = r'tets/test_1_tetrahedron.txt'
    is_file(filepath)

    file_format, points = read_coordinates(filepath)

    if file_format == 'V':
        find_convex_hull(points)


if __name__ == '__main__':
    main()
