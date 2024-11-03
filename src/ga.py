import numpy as np # type: ignore
import random
import time
import matplotlib.pyplot as plt # type: ignore

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

    def fitness(self):
        total_error, _ = self.calculate_objective_value()
        return 1 / (1 + total_error)  

    def copy(self):
        new_cube = MagicCube(self.n, np.copy(self.cube))
        new_cube.row_sums = np.copy(self.row_sums)
        new_cube.col_sums = np.copy(self.col_sums)
        new_cube.layer_sums = np.copy(self.layer_sums)
        return new_cube

class GeneticAlgorithm:
    def __init__(self, cube_size, population_size, max_iterations):
        self.cube_size = cube_size
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.population = [MagicCube(cube_size) for _ in range(population_size)]
        self.objective_log = []

    def run(self):
        start_time = time.time()
        initial_cube = self.population[0].cube.copy()  
        for iteration in range(self.max_iterations):
            fitness_values = [cube.fitness() for cube in self.population]
            max_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            self.objective_log.append((iteration, max_fitness, avg_fitness))
            self.population = self.selection(fitness_values)
            self.population = self.crossover_and_mutate(self.population)

        end_time = time.time()
        final_cube = self.population[0].cube  
        final_objective_value = self.population[0].calculate_objective_value()[0]

        print("Initial Cube State:\n", initial_cube)
        print("Final Cube State:\n", final_cube)
        print("Final Objective Value:", final_objective_value)
        print("Population Size:", self.population_size)
        print("Iterations:", self.max_iterations)
        print("Duration:", end_time - start_time, "seconds")
        
        self.plot_objective_values()

    def plot_objective_values(self):
        iterations, max_values, avg_values = zip(*self.objective_log)
        
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, max_values, label='Max Fitness Value', marker='o')
        plt.plot(iterations, avg_values, label='Average Fitness Value', marker='x')
        plt.title('Fitness Values Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def selection(self, fitness_values):
        selected_population = sorted(self.population, key=lambda cube: cube.fitness(), reverse=True)
        return selected_population[:self.population_size // 2]

    def crossover_and_mutate(self, selected_population):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected_population, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        return new_population

    def crossover(self, parent1, parent2):
        child_cube = np.copy(parent1.cube)
        for i in range(self.cube_size):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    if random.random() < 0.5:
                        child_cube[i, j, k] = parent2.cube[i, j, k]
        return MagicCube(self.cube_size, child_cube)

    def mutate(self, cube):
        i1, j1, k1 = random.randint(0, self.cube_size - 1), random.randint(0, self.cube_size - 1), random.randint(0, self.cube_size - 1)
        i2, j2, k2 = random.randint(0, self.cube_size - 1), random.randint(0, self.cube_size - 1), random.randint(0, self.cube_size - 1)
        cube.swap(i1, j1, k1, i2, j2, k2)

# Main program
if __name__ == "__main__":
    cube_size = 5
    population_size = 100
    max_iterations = 100

    ga = GeneticAlgorithm(cube_size, population_size, max_iterations)
    ga.run()