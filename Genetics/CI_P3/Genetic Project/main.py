from genetic import *
from generation import *
import matplotlib.pyplot as plt

generation_dict = {}


def get_input():
    game_level = input("Level No. ? ")
    game_file = 'level' + game_level + ".txt"
    population = input("Population? ")
    s = int(input("Score mode? \n1) Without winning points\n2) With winning points\n"))
    if s == 1:
        score_mode = 0
    elif s == 2:
        score_mode = 1
    else:
        print('Wrong score mode input!')
        exit()
    selection_mode_input = input("Selection mode?\n1) Weighted Random Selection\n2) Best Selection\n")
    if selection_mode_input == '1':
        selection_mode = 'random'
    elif selection_mode_input == '2':
        selection_mode = 'best'
    else:
        print('Wrong selection mode input!')
        exit()

    crossover_mode_input = input(
        "Enter crossover mode:\n1)One point - Random\n2)One point - Specified\n3)Two points - Random\n4)Two points - Specified\n")
    if crossover_mode_input == '1':
        crossover_mode = 'random 1'
        crossover_point = 0
    elif crossover_mode_input == '2':
        crossover_mode = 'specified 1'
        crossover_point = int(input('Enter crossover point: \n'))
    elif crossover_mode_input == '3':
        crossover_mode = 'random 2'
        crossover_point = 0
    elif crossover_mode_input == '4':
        crossover_mode = 'specified 2'
        crossover_point = input('Enter crossover points: \n').split(" ")
        crossover_point = [int(point) for point in crossover_point]
    else:
        print('Wrong crossover mode input!')
        exit()

    mutation_prob = input("Mutation probability? ")
    return game_file, population, score_mode, selection_mode, crossover_mode, crossover_point, float(
        mutation_prob)


def genetic_algorithm(game_board, population, score_mode, selection_mode, crossover_mode, crossover_point,
                      mutation_prob):
    # population is creating....
    initial_population = generate_population(int(population), game_board, score_mode)
    # first generation
    generations = {1: initial_population}
    print('********************************')
    genetic = Genetic(generations, game_board, selection_mode, crossover_mode, crossover_point, mutation_prob,
                      score_mode)
    return genetic


def plot_results(genetic):
    generation_average_scores = genetic.generation_average_scores
    generation_max_scores = genetic.generation_max_score
    generation_min_scores = genetic.generation_min_score

    plt.plot(*zip(*sorted(generation_average_scores.items())), color='r', label='avg scores')
    plt.plot(*zip(*sorted(generation_max_scores.items())), color='g', label='max scores')
    plt.plot(*zip(*sorted(generation_min_scores.items())), color='b', label='min scores')

    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    game_file, population, score_mode, selection_mode, crossover_mode, crossover_point, mutation_prob = get_input()
    game_board = get_level(game_file)  # reading game file from source
    result = genetic_algorithm(game_board, population, score_mode, selection_mode,
                               crossover_mode, crossover_point, mutation_prob)

    if len(result.generation_average_scores) != 10000:
        result_path_str = result.best_answer.string
        print("Goal chromosome: {}".format(result_path_str))
        print("Generation of Goal chromosome: {}".format(result.best_answer.generation))
        print("Score of Goal chromosome: {}".format(result.best_answer.score))

        # plot results
        plot_results(result)


    else:
        print("Can't win the game! or maximum generation limit reached!")
