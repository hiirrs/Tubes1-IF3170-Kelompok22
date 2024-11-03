import numpy as np # type: ignore
import random
import time
from concurrent.futures import ThreadPoolExecutor

class MagicCube:
    def __init__(self, n, cube=None):
        self.n = n
        self.cube = cube if cube is not None else self.generate_random_cube()
        self.magic_number = self.calculate_magic_number()
        self.row_sums = np.sum(self.cube, axis=2)
        self.col_sums = np.sum(self.cube, axis=1)
        self.layer_sums = np.sum(self.cube, axis=0)

    def generate_random_cube(self):
        numbers = list(range(1, self.n**3 + 1))
        random.shuffle(numbers)
        cube = np.array(numbers).reshape(self.n, self.n, self.n)
        return cube

    def calculate_magic_number(self):
        return (self.n * (self.n**3 + 1)) // 2

    def swap(self, i1, j1, k1, i2, j2, k2):
        self.cube[i1, j1, k1], self.cube[i2, j2, k2] = self.cube[i2, j2, k2], self.cube[i1, j1, k1]
        self.update_sums(i1, j1, k1, i2, j2, k2)

    def update_sums(self, i1, j1, k1, i2, j2, k2):
        self.row_sums[i1, j1] = np.sum(self.cube[i1, j1, :])
        self.row_sums[i2, j2] = np.sum(self.cube[i2, j2, :])
        self.col_sums[i1, k1] = np.sum(self.cube[:, i1, k1])
        self.col_sums[i2, k2] = np.sum(self.cube[:, i2, k2])
        self.layer_sums[j1, k1] = np.sum(self.cube[i1, :, k1])
        self.layer_sums[j2, k2] = np.sum(self.cube[i2, :, k2])

    def calculate_objective_value(self):
        total_error = (np.abs(self.row_sums - self.magic_number).sum() +
                    np.abs(self.col_sums - self.magic_number).sum() +
                    np.abs(self.layer_sums - self.magic_number).sum())
        
        row_error_count = np.sum(np.abs(self.row_sums - self.magic_number) > 0)
        col_error_count = np.sum(np.abs(self.col_sums - self.magic_number) > 0)
        layer_error_count = np.sum(np.abs(self.layer_sums - self.magic_number) > 0)

        diagonal_error_count = 0

        for i in range(self.n):
            main_diagonal_sum = self.cube[i, range(self.n), range(self.n)].sum()
            anti_diagonal_sum = self.cube[i, range(self.n), range(self.n-1, -1, -1)].sum()
            vertical_diagonal_sum = self.cube[range(self.n), i, range(self.n)].sum()
            vertical_anti_diagonal_sum = self.cube[range(self.n), i, range(self.n-1, -1, -1)].sum()
            depth_diagonal_sum = self.cube[range(self.n), range(self.n), i].sum()
            depth_anti_diagonal_sum = self.cube[range(self.n), range(self.n-1, -1, -1), i].sum()

            main_diagonal_error = abs(main_diagonal_sum - self.magic_number)
            anti_diagonal_error = abs(anti_diagonal_sum - self.magic_number)
            vertical_diagonal_error = abs(vertical_diagonal_sum - self.magic_number)
            vertical_anti_diagonal_error = abs(vertical_anti_diagonal_sum - self.magic_number)
            depth_diagonal_error = abs(depth_diagonal_sum - self.magic_number)
            depth_anti_diagonal_error = abs(depth_anti_diagonal_sum - self.magic_number)

            if main_diagonal_error > 0:
                total_error += main_diagonal_error
                diagonal_error_count += 1
                
            if anti_diagonal_error > 0:
                total_error += anti_diagonal_error
                diagonal_error_count += 1
                
            if vertical_diagonal_error > 0:
                total_error += vertical_diagonal_error
                diagonal_error_count += 1
                
            if vertical_anti_diagonal_error > 0:
                total_error += vertical_anti_diagonal_error
                diagonal_error_count += 1
                
            if depth_diagonal_error > 0:
                total_error += depth_diagonal_error
                diagonal_error_count += 1
                
            if depth_anti_diagonal_error > 0:
                total_error += depth_anti_diagonal_error
                diagonal_error_count += 1

        total_error_count = row_error_count + col_error_count + layer_error_count + diagonal_error_count

        return total_error, total_error_count

    def copy(self):
        new_cube = MagicCube(self.n, np.copy(self.cube))
        new_cube.row_sums = np.copy(self.row_sums)
        new_cube.col_sums = np.copy(self.col_sums)
        new_cube.layer_sums = np.copy(self.layer_sums)
        return new_cube

class SteepestAscent:
    def __init__(self, initial_cube):
        self.cube = initial_cube

    def get_neighbor(self):
        best_neighbor = None
        best_cost, best_error_count = self.cube.calculate_objective_value()

        for i1 in range(self.cube.n):
            for j1 in range(self.cube.n):
                for k1 in range(self.cube.n):
                    for i2 in range(self.cube.n):
                        for j2 in range(self.cube.n):
                            for k2 in range(self.cube.n):
                                if (i1, j1, k1) != (i2, j2, k2):
                                    neighbor = self.cube.copy()
                                    neighbor.swap(i1, j1, k1, i2, j2, k2)

                                    neighbor_cost, neighbor_error_count = neighbor.calculate_objective_value()

                                    if neighbor_error_count < best_error_count:
                                        best_neighbor = neighbor
                                        best_cost = neighbor_cost
                                        best_error_count = neighbor_error_count
                                    elif neighbor_error_count == best_error_count and neighbor_cost < best_cost:
                                        best_neighbor = neighbor
                                        best_cost = neighbor_cost                                    
        return best_neighbor

    def hill_climb(self):
        start_time = time.time()
        i = 0
        while True:
            print(f"Iteration {i}")
            best_cost, best_error_count = self.cube.calculate_objective_value()
            print(f"Best cost: {best_cost}, Best error count: {best_error_count}")

            neighbor = self.get_neighbor()
            if not neighbor or (neighbor.calculate_objective_value()[0] >= best_cost and neighbor.calculate_objective_value()[1] >= best_error_count):
                break
            
            self.cube = neighbor
            i += 1

        end_time = time.time()
        duration = end_time - start_time
        final_cost, final_error_count = self.cube.calculate_objective_value()
        return self.cube, final_cost, final_error_count, duration

def run_with_custom_cube(custom_cube):
    n = custom_cube.shape[0]
    initial_cube = MagicCube(n, cube=custom_cube)
    sa = SteepestAscent(initial_cube)
    best_solution, best_cost, best_error_count, duration = sa.hill_climb()
    return best_solution, best_cost, best_error_count, duration

custom_input = np.array([
    [[110,  51, 109,  81,  26],
     [123,  73,  91,  79,  37],
     [30,   72,  45,  99,  64],
     [35,    8, 100,  58,  40],
     [3,    67,  85, 103, 105]],

    [[31,   83,  41, 124, 116],
     [2,    68,  97,  98,  63],
     [5,   117,  96,  17,  21],
     [11,  125,  25,  77,   7],
     [33,  121,  74,  47, 113]],

    [[60,  108,  89, 101, 112],
     [62,  115,  27,  39, 102],
     [65,   53,  15, 119,  90],
     [66,  120,  59, 122,  20],
     [61,   24,  23,  95,  12]],

    [[69,   82,  34,   9, 107],
     [10,   94,  52, 111,  16],
     [55,   50,  76,  78,  56],
     [106,  44,  18,   1, 118],
     [71,   54,  93,  36,  75]],

    [[80,   29,   6,  57,  86],
     [48,   46,  70,  92,  87],
     [4,    42,  22,  13,  43],
     [38,   49,  14,  84, 114],
     [19,  104,  88,  28,  32]]
])

test_solution = np.array([
    [[25, 16, 80, 104, 90],
     [115, 98, 4, 1, 97],
     [42, 111, 85, 2, 75],
     [66, 72, 27, 102, 48],
     [67, 18, 119, 106, 5]],

    [[91, 77, 71, 6, 70],
     [52, 64, 117, 69, 13],
     [30, 118, 21, 123, 23],
     [26, 39, 92, 44, 114],
     [116, 17, 14, 73, 95]],

    [[47, 61, 45, 76, 86],
     [107, 43, 38, 33, 94],
     [89, 68, 63, 58, 37],
     [32, 93, 88, 83, 19],
     [40, 50, 81, 65, 79]],

    [[31, 53, 112, 109, 10],
     [12, 82, 34, 87, 100],
     [103, 3, 105, 8, 96],
     [113, 57, 9, 62, 74],
     [56, 120, 55, 49, 35]],

    [[121, 108, 7, 20, 59],
     [29, 28, 122, 125, 11],
     [51, 15, 41, 124, 84],
     [78, 54, 99, 24, 60],
     [36, 110, 46, 22, 101]]
])

solution_swap = np.array([
       [[  5,  16,  80, 104,  90],
        [115,  98,   4,   1,  97],
        [ 42, 111,  85,   2,  75],
        [ 66,  72,  27, 102,  48],
        [ 67,  18, 119, 106,  25]],

       [[ 91,  77,  71,   6,  70],
        [ 52,  64, 117,  69,  13],
        [ 30,  21, 118, 123,  23],
        [ 26,  39,  92,  44, 114],
        [116,  17,  14,  73,  95]],

       [[ 47,  61,  45,  76,  86],
        [107,  43,  38,  33,  94],
        [ 89,  68,  63,  58,  37],
        [ 32,  93,  88,  83,  19],
        [ 40,  50,  81,  65,  79]],

       [[ 31,  53, 112, 109,  10],
        [ 12,  82,  34,  87,  99],
        [103,   3, 105,   8,  96],
        [113,  57,   9,  62,  74],
        [ 56, 120,  55,  49,  35]],

       [[121, 108,   7,  20,  59],
        [ 29,  28, 122, 125,  11],
        [ 51,  15,  41, 124,  84],
        [ 78,  54, 100,  24,  60],
        [ 36, 110,  46,  22, 101]]
])

best_solution, best_cost, best_error_count, duration = run_with_custom_cube(custom_input)
print("Optimized Cube:")
print(best_solution.cube)
print(f"Best Cost: {best_cost}")
print(f"Best Error Count: {best_error_count}")
print(f"Duration: {duration:.2f} seconds")
