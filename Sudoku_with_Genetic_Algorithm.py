import time
import random as rd
from tkinter import *

# Function to check if placing 'num' in a given row, column, and 3x3 grid is safe
def is_safe(grid, row, col, num):
    if num in grid[row]:
        return False
    if any(num == grid[r][col] for r in range(9)):
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    return not any(num == grid[r][c] for r in range(start_row, start_row + 3) for c in range(start_col, start_col + 3))

# Function to solve the Sudoku puzzle using backtracking
def solve_sudoku(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_safe(grid, row, col, num):
                        grid[row][col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

# Function to generate the Sudoku puzzle based on selected difficulty level
def generate_sudoku(level):
    # Dynamically adjust the number of cells to remove for the selected difficulty level
    global level_cells_to_remove
    if level == 1:
        level_cells_to_remove = rd.randint(35, 45)  # Easy
    elif level == 2:
        level_cells_to_remove = rd.randint(45, 50)  # Medium
    elif level == 3:
        level_cells_to_remove = rd.randint(50, 55)  # Hard
    elif level == 4:
        level_cells_to_remove = rd.randint(55, 60)  # Very Hard
    elif level == 5:
        level_cells_to_remove = 64  # Expert
    grid = [[0] * 9 for _ in range(9)]
    for i in range(0, 9, 3):
        nums = rd.sample(range(1, 10), 9)
        for j in range(9):
            row, col = i + j // 3, i + j % 3
            grid[row][col] = nums[j]
    solve_sudoku(grid)
    
    # Remove some cells to create the puzzle
    cells_to_remove = level_cells_to_remove
    while cells_to_remove > 0:
        row, col = rd.randint(0, 8), rd.randint(0, 8)
        if grid[row][col] != 0:
            grid[row][col] = 0
            cells_to_remove -= 1
    return grid, level_cells_to_remove

# This function generates a "gene" for the genetic algorithm.
# A "gene" here represents a shuffled list of numbers (1-9) that corresponds to a row in a Sudoku grid.
# The gene is adjusted based on an initial input (a partially filled row in the Sudoku puzzle).
# The function ensures that any non-zero value from the initial list stays in the same position, while the remaining values are shuffled.
def make_gene(initial=None): 
    if initial is None:
        initial = [0] * 9  # If no initial list is provided, use an empty row with all zeros
    position_map = {}  # Map to track the position of each number
    new_gene = list(range(1, 10))  # Create a list of numbers 1-9
    rd.shuffle(new_gene)  # Shuffle the numbers to randomize the gene
    for index in range(9):
        position_map[new_gene[index]] = index  # Map each number to its position in the shuffled list
    for current_pos in range(9):
        if (initial[current_pos] != 0 and new_gene[current_pos] != initial[current_pos]):
            # If the initial value is non-zero and the shuffled value doesn't match the initial value, swap
            current_number = new_gene[current_pos]
            target_pos = position_map[initial[current_pos]]
            new_gene[current_pos], new_gene[target_pos] = new_gene[target_pos], new_gene[current_pos]
            position_map[initial[current_pos]] = current_pos  # Update the map with new positions
            position_map[current_number] = target_pos
    return new_gene  # Return the modified gene

# This function generates a "chromosome" for the genetic algorithm, which represents a complete Sudoku grid.
# A chromosome consists of multiple "genes", each of which is a shuffled row in the Sudoku grid.
# The function works by calling `make_gene` for each row of the Sudoku grid (either provided or initialized to zero).
# A chromosome is essentially a collection of genes (rows), where each row is a valid permutation of numbers (1-9) based on the given initial state.
def make_chromosome(initial=None):
    if initial is None:
        initial = [[0] * 9] * 9  # If no initial grid is provided, create a 9x9 grid filled with zeros.
    return [make_gene(row) for row in initial]  # Generate a chromosome by creating a gene for each row in the grid.

# This function generates a population of chromosomes for the genetic algorithm, 
# where each chromosome is a possible solution to the Sudoku puzzle.
# The population is a list of multiple "chromosomes", where each chromosome represents a complete Sudoku grid.
# It takes in two parameters: `count` which specifies the number of chromosomes (individuals) in the population,
# and `initial`, an optional 9x9 Sudoku grid that can be provided as the starting point for generating the chromosomes.
def make_population(count, initial=None):
    if initial is None:
        initial = [[0] * 9] * 9  # If no initial grid is provided, create a 9x9 grid filled with zeros.
    return [make_chromosome(initial) for _ in range(count)]  # Generate the population by creating `count` chromosomes.


def fitness_func(chromosome):
    fitness_score = 0  # Initialize fitness score to zero
    # Check fitness based on rows: each row should contain unique numbers (1-9)
    for row in chromosome:
        fitness_score += len(set(row))  # A set removes duplicates, so the length will reflect unique numbers
    # Check fitness based on columns: each column should contain unique numbers (1-9)
    for col in range(9):
        column = [chromosome[row][col] for row in range(9)]  # Create a list for the column
        fitness_score += len(set(column))  # A set removes duplicates, so the length will reflect unique numbers
    # Check fitness based on 3x3 subgrids (blocks): each block should contain unique numbers (1-9)
    for block_row in range(0, 9, 3):  # Iterate over the starting rows of the 3x3 blocks
        for block_col in range(0, 9, 3):  # Iterate over the starting columns of the 3x3 blocks
            block = set()  # Use a set to ensure unique numbers in the block
            for i in range(3):  # Iterate through the 3 rows in the 3x3 block
                for j in range(3):  # Iterate through the 3 columns in the 3x3 block
                    block.add(chromosome[block_row + i][block_col + j])  # Add the number to the block set
            fitness_score += len(block)  # Add the size of the block set to the fitness score (number of unique numbers)
    return fitness_score  # Return the total fitness score

def evaluate_population(population, fitness_func):
    # Calculate fitness score for each chromosome in the population
    scored_population = [(chromosome, fitness_func(chromosome)) for chromosome in population]
    # Sort the population based on the fitness score in descending order
    # A higher fitness score is considered better
    sorted_population = sorted(scored_population, key=lambda x: x[1], reverse=True)
    # Return the sorted population (higher fitness chromosomes come first)
    return sorted_population

def tournament_selection(population, TOURSIZE, selection_rate):
    # Determine how many of the top individuals (elites) will be directly kept for the next generation
    top_count = int(len(population) * selection_rate)  
    elites = [candidate for candidate, _ in population[:top_count]]  # Select elites based on top_count
    non_elite_population = population[top_count:]  # The remaining population for tournament selection
    mating_pool = []  # Will store the selected chromosomes for the next generation
    # Select individuals from non-elite population via tournament selection
    while len(mating_pool) < len(non_elite_population):
        tournament_contestants = rd.sample(non_elite_population, TOURSIZE)  # Randomly pick 'TOURSIZE' individuals
        winner = max(tournament_contestants, key=lambda x: x[1])  # The winner is the one with the highest fitness
        mating_pool.append(winner[0])  # Add the winner (chromosome) to the mating pool
    return mating_pool, elites  # Return the mating pool and the elites

def crossover(parent1, parent2, initial):
    # Create copies of the parents to generate the children
    child1 = [row[:] for row in parent1]
    child2 = [row[:] for row in parent2]

    # Randomly choose two crossover points (between 0 and 8)
    crossover_point1 = rd.randint(0, 8)
    crossover_point2 = rd.randint(0, 8)

    # Ensure the crossover points are different
    while crossover_point1 == crossover_point2:
        crossover_point2 = rd.randint(0, 8)

    # Swap points if crossover_point1 > crossover_point2 to ensure proper range
    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1

    # Perform crossover between the parents by swapping rows of chromosomes
    for i in range(crossover_point1, crossover_point2 + 1):
        child1[i], child2[i] = crossover_rows(child1[i], child2[i], initial[i])

    return child1, child2

def crossover_rows(row1, row2, initial_row):
    Nd = len(row1)  # Length of the row, typically 9 for a Sudoku puzzle
    child_row1 = [0] * Nd  # Initialize empty child rows
    child_row2 = [0] * Nd

    # Determine the numbers that can be used (i.e., those not in the initial row)
    remaining = [i for i in range(1, Nd + 1) if i not in initial_row]

    cycle = 0  # Cycle counter, alternates between filling child_row1 and child_row2

    # First, fill in the child rows with the fixed values from the initial_row
    for idx in range(Nd):
        if initial_row[idx] != 0:
            child_row1[idx] = initial_row[idx]
            child_row2[idx] = initial_row[idx]

    # Start the cycle crossover process
    while 0 in child_row1 or 0 in child_row2:
        if cycle % 2 == 0:  # Even cycle: Fill child_row1 with values from row1, child_row2 with values from row2
            index = find_unused(row1, remaining, child_row1)
            if index is None:
                break
            start = row1[index]
            child_row1[index] = row1[index]
            child_row2[index] = row2[index]
            next_value = row2[index]
            
            # Continue the cycle
            while next_value != start:
                index = find_value(row1, next_value)
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next_value = row2[index]

            cycle += 1
        else:  # Odd cycle: Reverse the roles, child_row1 gets values from row2 and child_row2 gets from row1
            index = find_unused(row1, remaining, child_row1)
            if index is None:
                break
            start = row1[index]
            child_row1[index] = row2[index]
            child_row2[index] = row1[index]
            next_value = row2[index]

            # Continue the cycle
            while next_value != start:
                index = find_value(row1, next_value)
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next_value = row2[index]

            cycle += 1
    
    return child_row1, child_row2

# 2 other crossover functions can be used
def two_point_crossover(ch1, ch2):
    point1 = rd.randint(0, 8)
    point2 = rd.randint(0, 8)
    while point1 == point2:
        point1 = rd.randint(0, 8)
        point2 = rd.randint(0, 8)
    if point1 > point2:
        point1, point2 = point2, point1
    new_child_1 = ch1[:point1] + ch2[point1:point2] + ch1[point2:]
    new_child_2 = ch2[:point1] + ch1[point1:point2] + ch2[point2:]
    return new_child_1, new_child_2

def uniform_crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = rd.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        elif x == 0:
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2

# This function searches for an unused position in a given row where a value from the remaining
# list can be placed, while also ensuring that the corresponding position in the child_row is empty (i.e., 0).
# It iterates over the row, checking if a value from the remaining list is found in the row
# and if the current position in the child_row is still unfilled (i.e., value is 0).
# If such a position is found, the index of that position is returned. 
# If no such position exists, the function returns None.
def find_unused(row, remaining, child_row):
    for idx, val in enumerate(row):
        if val in remaining and child_row[idx] == 0:
            return idx
    return None

# This function searches for a specific value in a given row and returns its index.
# If the value is found in the row, the index of the first occurrence is returned.
# If the value is not found, the function returns None.
def find_value(row, value):
    if value in row:
        return row.index(value)
    return None

# This function performs mutation on a chromosome with a given probability.
# For each row in the chromosome, it decides whether to mutate based on the mutation probability (pm).
# If a random number between 0 and 100 is less than pm * 100, it calls the make_gene function to generate a new row for that index,
# otherwise, the current row is retained.
# The function returns the mutated chromosome.
def mutation(ch, pm, initial):
    return [make_gene(initial[i]) if rd.randint(0, 100) < pm * 100 else ch[i] for i in range(9)]

# This function evaluates a population of chromosomes and finds the one with the highest fitness score.
# It iterates through each chromosome in the population, calculates its fitness using the fitness_func,
# and compares it to the current maximum fitness found.
# If a chromosome has a higher fitness, it updates the max_fitness and best_chromosome variables.
# The function returns the highest fitness score and the corresponding chromosome.
def find_max_fitness(population):
    max_fitness = 0
    best_chromosome = None
    for chromosome in population:
        fitness = fitness_func(chromosome)
        if fitness > max_fitness:
            max_fitness = fitness
            best_chromosome = chromosome
    return max_fitness, best_chromosome


# This function updates the global variables final_fitness and final_solution if a better solution is found.
# It compares the provided best_fitness with the current final_fitness.
# If the provided best_fitness is greater, it updates the final_fitness and final_solution 
# to reflect the new best fitness and its corresponding solution.
def update_solution(best_fitness, best_solution):
    global final_fitness, final_solution
    if best_fitness > final_fitness:
        final_fitness = best_fitness
        final_solution = best_solution


# This function generates the next generation of the population through crossover and mutation.
# It iterates over pairs of chromosomes in the population.
# For each pair, a random number is generated to decide if they will undergo crossover based on the crossover probability (pc).
# After possible crossover, both chromosomes are mutated based on the mutation probability (pm) and add to new generation pool.
# This process continues until the entire population has been processed.
def get_next_gen(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rd.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2, initial)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool

def toggle_pause():
    global paused
    paused = not paused
    if paused:
        btn_stop.config(text="Resume")
    else:
        btn_stop.config(text="Pause")

def genetic_algorithm(initial):
    global paused
    paused = False # Controls whether the algorithm is paused
    global probability_of_mutation

     # Step 1: Initialize the population with chromosomes based on the initial puzzle state
    population = make_population(POPULATION, initial)
    stagnant_generations = 0  # To track generations without improvement
    previous_max_fitness = 0  # To track the fitness of the previous generation

    # Step 2: Start the main loop for each generation
    for generation in range(MAX_GENERATIONS):
        while paused:
            root.update() 
            continue

        # Step 3: Evaluate the fitness of the population
        population = evaluate_population(population, fitness_func)

        # Step 4: Select the mating pool and elite chromosomes using tournament selection
        mating_pool, elite_chromosome = tournament_selection(population, TOURSIZE, PS)
        # Shuffle the mating pool to avoid any bias in crossover
        rd.shuffle(mating_pool)

        # Step 5: Generate the next generation through crossover and mutation
        population = get_next_gen(mating_pool, initial, probability_of_mutation, PC)
        # Add the elite chromosomes back into the population (elitism)
        population = population + elite_chromosome

        # Step 6: Evaluate the best solution in the population
        max_fitness, best_solution = find_max_fitness(population)

        # Step 7: Update the fitness label and the grid with the new best solution
        update_fitness(generation + 1, max_fitness)
        update_grid(best_solution)
        root.update()

        # Step 8: Check if the solution is found (maximum fitness reached)
        if max_fitness == 243:
            return max_fitness, best_solution
        
        # Step 9: Check if there is no improvement in fitness
        if max_fitness <= previous_max_fitness:
            stagnant_generations += 1
        elif max_fitness > previous_max_fitness:
            stagnant_generations = 0
            update_solution(max_fitness, best_solution)
        
        # Step 10: Update previous fitness for comparison
        previous_max_fitness = max_fitness

        # Step 11: If fitness is stagnant for 300 generations, reinitialize the population
        if stagnant_generations == 300:
            reinit_message_1 = "Fitness has not improved in 300 generations."
            reinit_message_2 = "Reinitializing population..."
            # Update the fitness label with the message
            current_text = fitness_label.cget("text")
            current_text = f"{current_text}\n{reinit_message_1}\n{reinit_message_2}" if current_text else f"{reinit_message_1}\n{reinit_message_2}"
            fitness_label.config(text=current_text)
            population = make_population(POPULATION, initial)
            probability_of_mutation = 0.05
            update_solution(max_fitness, best_solution)
            stagnant_generations = 0
            previous_max_fitness = 0
            update_parameters()

        # Step 12: Increase mutation rate every 50 generations if fitness stagnates
        if stagnant_generations % 50 == 0 and stagnant_generations != 0:
            probability_of_mutation += 0.01
            mutation_message_1 = f"Fitness has not improved in {stagnant_generations} generations."
            mutation_message_2 = f"Increased mutation rate to {probability_of_mutation:.3f}"         
            current_text = fitness_label.cget("text")
            current_text = f"{current_text}\n{mutation_message_1}\n{mutation_message_2}" if current_text else f"{mutation_message_1}\n{mutation_message_2}"
            fitness_label.config(text=current_text)
            update_parameters()
    return max_fitness, best_solution

# Function to update the Tkinter grid with the current Sudoku puzzle
def update_grid(grid):
    for row in range(9):
        for col in range(9):
            entry = cells[(row, col)]
            if grid[row][col] != 0:
                entry.delete(0, "end")
                entry.insert(0, str(grid[row][col]))
            else:
                entry.delete(0, "end")

# Function to clear all input values in the grid
def clearValues():
    solvedLabel.config(text="")  # Reset solved message
    fitness_label.config(text="") # Reset fitness message
    for cell in cells.values():
        cell.delete(0, "end")  # Clear the value in the entry widget

# Function to solve the Sudoku puzzle using the current grid values
def solveWithGeneticAlgorithm():
    # Step 1: Collect the initial Sudoku grid from the UI
    initial = [[int(cells[(row, col)].get() or 0) for col in range(9)] for row in range(9)]
    # Step 2: Start the timer to track how long the algorithm takes to solve the puzzle
    tic = time.time()
    # Step 3: Run the genetic algorithm to solve the puzzle
    m, c = genetic_algorithm(initial)
    # Step 4: Stop the timer and calculate the time taken to solve
    toc = time.time()
    time_taken = toc - tic
    # Step 5: If the genetic algorithm didn't find a solution (best fitness < final_fitness),
    # use the final solution found during the algorithm's run.
    if (m < final_fitness):
        m = final_fitness
        c = final_solution  
    # Step 6: Update the UI based on the result
    if m == 243: # If the fitness is 243, the puzzle is solved
        solvedLabel.config(text=f"Sudoku Solved with: {time_taken:.2f} seconds")
        update_grid(c)
    else: # If no solution is found, display the best attempt with its fitness score
        solvedLabel.config(text=f"No solution found with: {time_taken:.2f} second, best fitness: {m}")
        update_grid(c)

# Function to generate and display a new Sudoku puzzle based on the difficulty level
def generate_and_display_puzzle():
    level = level_var.get()  # Get the level from the radio button variable
    if level < 1 or level > 5:
        solvedLabel.config(text="Please select a level between 1 and 5")
        return
    grid, cells_to_remove = generate_sudoku(level)
    update_grid(grid)
    solvedLabel.config(text=f"Puzzle generated with level {level} and {cells_to_remove} cells removed")

def update_fitness(generation, max_fitness):
    # Get the current text in the label
    current_text = fitness_label.cget("text")
    
    # Add the new line for the current generation and max fitness
    new_line = f"Max Fitness in generation {generation}: {max_fitness}"

    # If current text exists, append the new line; otherwise, start fresh
    if current_text:
        current_text = f"{current_text}\n{new_line}"
    else:
        current_text = new_line
    
    # Split the current text into lines and keep only the last 10 lines
    lines = current_text.split("\n")
    lines = lines[-10:]  # Keep the last 10 lines

    # Join the lines back and update the label text
    fitness_label.config(text="\n".join(lines))

# Dictionary to store entry widgets for easy access


# Function to draw a 3x3 grid with alternating background color
def draw3x3Grid(start_row, start_col, bgcolor):
    for i in range(3):
        for j in range(3):
            entry = Entry(root, width=5, bg=bgcolor, justify='center')
            entry.grid(row=start_row + i, column=start_col + j, sticky="nsew", padx=3, pady=3, ipady=5)
            cells[(start_row + i, start_col + j)] = entry

# Function to draw the 9x9 Sudoku grid
def draw9x9Grid():
    color = "#CBC3E3"  # initial color
    for row in range(0, 9, 3):  # start every 3rd row (0, 3, 6)
        for col in range(0, 9, 3):  # start every 3rd column (0, 3, 6)
            draw3x3Grid(row, col, color)  # draw a 3x3 subgrid
            color = "#ffffff" if color == "#CBC3E3" else "#CBC3E3"  # alternate color

def update_parameters():
    population_label.config(text=f"Population: {POPULATION}")
    generations_label.config(text=f"Generations: {MAX_GENERATIONS}")
    ps_label.config(text=f"Selection Probability: {PS}")
    mutation_label.config(text=f"Mutation Probability: {probability_of_mutation:.3f}")
    crossover_label.config(text=f"Crossover Probability: {PC}")
    toursize_label.config(text=f"Tournament Size: {TOURSIZE}")

if __name__ == "__main__":
    # Population size
    POPULATION = 2000

    # Number of max generations
    MAX_GENERATIONS = 3000

    # Probability of selection
    PS = 0.05

    # Probability of mutation
    probability_of_mutation = 0.05

    # Probability of crossover
    PC = 0.95
    # Tournament size
    TOURSIZE = 3

    final_fitness = 0
    final_solution = []

    # Tkinter setup
    root = Tk()
    root.title("Sudoku Solver with Genetic Algorithm")
    root.geometry("800x650")

    # Solved label (to display result or messages)
    solvedLabel = Label(root, text="", fg='green')
    solvedLabel.grid(row=11, column=0, columnspan=9)

    # Label to display max fitness for generations
    fitness_label = Label(root, text="", fg="blue", font=("Arial", 10))
    fitness_label.grid(row=12, column=0, columnspan=9)
    # Difficulty level buttons using Radiobutton
    level_var = IntVar(value=1)  # Default level is 1 (Easy)

    # Easy Button
    easy_button = Radiobutton(root, text="Easy", variable=level_var, value=1, command=generate_and_display_puzzle)
    easy_button.grid(row=0, column=9, padx=5, pady=5)

    # Medium Button
    medium_button = Radiobutton(root, text="Medium", variable=level_var, value=2, command=generate_and_display_puzzle)
    medium_button.grid(row=1, column=9, padx=5, pady=5)

    # Hard Button
    hard_button = Radiobutton(root, text="Hard", variable=level_var, value=3, command=generate_and_display_puzzle)
    hard_button.grid(row=2, column=9, padx=5, pady=5)

    # Very Hard Button
    very_hard_button = Radiobutton(root, text="Very Hard", variable=level_var, value=4, command=generate_and_display_puzzle)
    very_hard_button.grid(row=3, column=9, padx=5, pady=5)

    # Expert Button
    expert_button = Radiobutton(root, text="Expert", variable=level_var, value=5, command=generate_and_display_puzzle)
    expert_button.grid(row=4, column=9, padx=5, pady=5)

    # Buttons Frame to keep them separate from the grid
    button_frame = Frame(root)
    button_frame.grid(row=5, column=9, rowspan=12, pady=20)

    # Solve Button
    btn_solve = Button(button_frame, command=solveWithGeneticAlgorithm, text="Solve", width=10)
    btn_solve.grid(row=0, column=0, padx=5)

    # Clear Button
    btn_clear = Button(button_frame, command=clearValues, text="Clear", width=10)
    btn_clear.grid(row=1, column=0, padx=5)

    # Stop Button
    btn_stop = Button(button_frame, command=toggle_pause, text="Pause", width=10)
    btn_stop.grid(row=2, column=0, padx=5, pady=5)

    # Parameters Frame for displaying genetic algorithm settings
    params_frame = Frame(root)
    params_frame.grid(row=13, column=10, padx=20, pady=20)

    # Labels for displaying parameters
    population_label = Label(params_frame, text=f"Population: {POPULATION}", font=("Arial", 10))
    population_label.grid(row=0, column=0, padx=5, pady=5)

    generations_label = Label(params_frame, text=f"Max Generations: {MAX_GENERATIONS}", font=("Arial", 10))
    generations_label.grid(row=1, column=0, padx=5, pady=5)

    ps_label = Label(params_frame, text=f"Selection Probability: {PS}", font=("Arial", 10))
    ps_label.grid(row=2, column=0, padx=5, pady=5)

    mutation_label = Label(params_frame, text=f"Mutation Probability: {probability_of_mutation}", font=("Arial", 10))
    mutation_label.grid(row=3, column=0, padx=5, pady=5)

    crossover_label = Label(params_frame, text=f"Crossover Probability: {PC}", font=("Arial", 10))
    crossover_label.grid(row=4, column=0, padx=5, pady=5)

    toursize_label = Label(params_frame, text=f"Tournament Size: {TOURSIZE}", font=("Arial", 10))
    toursize_label.grid(row=5, column=0, padx=5, pady=5)

    cells = {}
    # Draw the initial 9x9 grid
    draw9x9Grid()

    # Initial puzzle display
    generate_and_display_puzzle()

    root.mainloop()
