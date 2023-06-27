from generation import *
from chromosome import *
import math
MAX_GENERATION = 10000


def is_end(number_generation):
    if number_generation > MAX_GENERATION:
        return True
    return False


def calculate_average_score(array_of_chromosomes):
    sum_scores = 0
    for chromosome in array_of_chromosomes:
        sum_scores += chromosome.score
    return sum_scores / len(array_of_chromosomes)


def find_max_score(array_of_chromosomes):
    array_of_chromosomes.sort(key=lambda x: x.score, reverse=True)
    return array_of_chromosomes[0].score


def find_min_score(array_of_chromosomes):
    array_of_chromosomes.sort(key=lambda x: x.score, reverse=False)
    return array_of_chromosomes[0].score


def choose_two_parents(array_of_chromosomes):
    parent1, parent2 = random.sample(array_of_chromosomes, 2)
    return parent1, parent2


def selection(array_of_prev_chromosomes, selection_mode):
    selected_parents = []
    len_population = len(array_of_prev_chromosomes)

    # randomly (with their scores as their weights) select from previous generation
    if selection_mode == 'random':
        score_weights = return_scores(array_of_prev_chromosomes)
        selected_parents += random.choices(list(array_of_prev_chromosomes), weights=score_weights,
                                           k=math.ceil(len_population * 0.5))

    # sort array_of_prev_chromosomes and take top chromosomes
    elif selection_mode == 'best':
        sort_by_score(array_of_prev_chromosomes)
        selected_parents += array_of_prev_chromosomes[:math.ceil(len_population * 0.5)]

    return selected_parents


def crossover(chromosome1, chromosome2, crossover_point, crossover_mode):
    # caution: chromosomes are objects here not their strings!
    offspring1, offspring2 = "", ""
    chromosome_length = len(chromosome1.string)

    if crossover_mode == 'random 1':
        crossing_point = random.randint(2, chromosome_length - 2)
        offspring1 = chromosome1.string[:crossing_point] + chromosome2.string[crossing_point:]
        offspring2 = chromosome2.string[:crossing_point] + chromosome1.string[crossing_point:]

    elif crossover_mode == 'specified 1':
        offspring1 = chromosome1.string[:crossover_point] + chromosome2.string[crossover_point:]
        offspring2 = chromosome2.string[:crossover_point] + chromosome1.string[crossover_point:]

    elif crossover_mode == 'random 2':
        crossing_points = random.sample(range(2,chromosome_length-2), 2)
        crossing_points.sort()
        offspring1 = chromosome1.string[0:crossing_points[0]] + chromosome2.string[crossing_points[0]:crossing_points[1]] + chromosome1.string[crossing_points[1]:]
        offspring2 = chromosome2.string[0:crossing_points[0]] + chromosome1.string[crossing_points[0]:crossing_points[1]] + chromosome2.string[crossing_points[1]:]


    elif crossover_mode == 'specified 2':
        offspring1 = chromosome1.string[:crossover_point[0]] + chromosome2.string[crossover_point[0]:crossover_point[1]] + chromosome1.string[crossover_point[1]:]
        offspring2 = chromosome2.string[:crossover_point[0]] + chromosome1.string[crossover_point[0]:crossover_point[1]] + chromosome2.string[crossover_point[1]:]

    return offspring1, offspring2


def mutation(chromosome, mutation_probability, game, score_mode):
    chromosome_failure_points = chromosome.failure_points
    if len(chromosome_failure_points) > 0:
        mutation_index = random.sample(chromosome_failure_points, 1)[0]
        new_value_mutation_index = "0"
        left_part = chromosome.string[:mutation_index]
        right_part = chromosome.string[mutation_index + 1:]

        chromosome.string = left_part + new_value_mutation_index + right_part
        new_score, new_failure_points = game.get_score(chromosome.string, score_mode)
        chromosome.score = new_score
        chromosome.failure_points = new_failure_points
        # chromosome.update_score_failurepoints(new_score, new_failure_points)


def check_goal_chromosome(new_generation):
    new_generation_copy = copy.deepcopy(new_generation)
    new_generation_copy.sort(key=lambda x: x.score, reverse=True)
    for chromosome in new_generation_copy:
        if len(chromosome.failure_points) == 0:
            return True, chromosome

    return False, False


class Genetic:
    # population is a dictionary of generation and array of chromosome objects
    # => {1: [chromosome1, chromosome2, ...], 2:[chromosomeK, ...]}
    def __init__(self, generations, game_plate, selection_mode, crossover_mode, crossover_point, mutation_prob,
                 score_mode):
        self.generations = generations
        self.game_plate = game_plate
        self.selection_mode = selection_mode
        self.crossover_mode = crossover_mode
        self.crossover_point = crossover_point
        self.mutation_prob = mutation_prob
        self.score_mode = score_mode
        self.best_answer = ''
        self.generation_average_scores = {}
        self.generation_max_score = {}
        self.generation_min_score = {}

        len_population = len(self.generations[1])
        self.generation_average_scores[1] = calculate_average_score(self.generations[1])
        self.generation_max_score[1] = find_max_score(self.generations[1])
        self.generation_min_score[1] = find_min_score(self.generations[1])

        current_generation = 2
        game = Game(self.game_plate)

        while len(self.generations) < MAX_GENERATION:

            # 70% of new population are from new chromosome (children of selected parents)
            # 30% of new population are from selected parents (after selection step)

            # selection step
            selected_parents = selection(self.generations[current_generation - 1], self.selection_mode)

            # crossover step
            new_generation, new_generation_strings = [], []
            while True:

                parent1, parent2 = choose_two_parents(selected_parents)
                child1, child2 = crossover(copy.deepcopy(parent1), copy.deepcopy(parent2), self.crossover_point, self.crossover_mode)

                if child1 not in new_generation_strings:
                    score1, failure_points1 = game.get_score(child1, self.score_mode)
                    new_generation.append(Chromosome(child1, score1, current_generation, failure_points1))
                    new_generation_strings.append(child1)
                if len(new_generation_strings) == math.ceil(len_population * 0.7):
                    break

                if child2 not in new_generation_strings:
                    score2, failure_points2 = game.get_score(child2, self.score_mode)
                    new_generation.append(Chromosome(child2, score2, current_generation, failure_points2))
                    new_generation_strings.append(child2)
                if len(new_generation_strings) == math.ceil(len_population * 0.7):
                    break

            new_generation += selected_parents

            # mutation step
            self.generations[current_generation] = new_generation
            for chromosome in self.generations[current_generation]:
                if random.random() < self.mutation_prob:
                    mutation(chromosome, self.mutation_prob, game, self.score_mode)

            self.generation_average_scores[current_generation] = calculate_average_score(self.generations[current_generation])
            self.generation_max_score[current_generation] = find_max_score(self.generations[current_generation])
            self.generation_min_score[current_generation] = find_min_score(self.generations[current_generation])

            find_chromosome, goal_chromosome = check_goal_chromosome(self.generations[current_generation])
            if find_chromosome:
                self.best_answer = goal_chromosome
                print(len(self.generations))
                break

            current_generation += 1