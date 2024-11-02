import random
import numpy as np # type: ignore
import time 

# Objective Function
def calculate_fitness(cube, magic_number):
    n = len(cube)
    fitness = 0

    for i in range(n):
        fitness += sum(cube[i, :, :].sum(axis=0)) 
        fitness += sum(cube[:, i, :].sum(axis=0)) 
        fitness += sum(cube[:, :, i].sum(axis=0))  

    fitness += np.sum(cube.diagonal(axis1=0, axis2=1))  
    fitness += np.sum(np.fliplr(cube).diagonal(axis1=0, axis2=1))  

    return abs(magic_number - fitness)

# Fungsi untuk menghasilkan kubus acak
def generate_random_cube(n):
    cube = np.random.randint(1, 10, (n, n, n))  
    return cube

# Fungsi untuk mutasi kubus
def mutate(cube):
    n = len(cube)
    x, y, z = random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0, n - 1)
    cube[x, y, z] = random.randint(1, 9)  

# Fungsi untuk crossover antara dua kubus
def crossover(parent1, parent2):
    n = len(parent1)
    offspring = np.zeros((n, n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if random.random() < 0.5:
                    offspring[i, j, k] = parent1[i, j, k]
                else:
                    offspring[i, j, k] = parent2[i, j, k]
    return offspring

# Genetic Algorithm
def genetic_algorithm(population_size, n, magic_number, generations):
    population = [(generate_random_cube(n), float('inf')) for _ in range(population_size)]

    for gen in range(generations):
        for i in range(population_size):
            fitness = calculate_fitness(population[i][0], magic_number)
            population[i] = (population[i][0], fitness)

        population.sort(key=lambda x: x[1])

        new_population = []
        for i in range(population_size // 2):
            parent1 = population[i][0]
            parent2 = population[population_size - 1 - i][0]

            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)

            mutate(offspring1)
            mutate(offspring2)

            new_population.append((offspring1, float('inf')))
            new_population.append((offspring2, float('inf')))

        population = new_population

    best_cube = population[0][0]
    return best_cube

# Testing
n = 5 
magic_number = 315  
population_size = 100
generations = 1000

start_time = time.time()
solution = genetic_algorithm(population_size, n, magic_number, generations)
end_time = time.time()

print("Kubus Ajaib:")
print(solution)
print("Kebugaran:", calculate_fitness(solution, magic_number))
print("Waktu Eksekusi: {:.4f} detik".format(end_time - start_time))