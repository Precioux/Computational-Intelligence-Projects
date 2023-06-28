from algorithm import *
from tools import *
import matplotlib.pyplot as plt

game_file = ''
population = 0
scoring = 0
selecting = 0
crossover = 0
crossover_point = 0
mutation = 0
generation_dict = {}


def get_input():
    global game_file, population, scoring, selecting, crossover, crossover_point, mutation
    print('***************** SUPER MARIO *****************')
    print('Lets design our game! ')
    game_level = input("Level No. ? ")
    game_file+= 'level' + game_level + ".txt"
    # GET POPULATION
    population = int(input("Population? "))
    # GET SCORING
    s = int(input("Scoring? \n1) Simple \n2) Compute Winning Points\n"))
    if s == 1:
        scoring = 0
    elif s == 2:
        scoring = 1
    else:
        print('Failed to set scoring!')
        exit()
    # GET SELECTING
    selecting_input = input("Selecting? \n1) Go with Weighting Randomly! \n2) Go for Best!\n")
    if selecting_input == '1':
        selecting = 'random'
    elif selecting_input == '2':
        selecting = 'best'
    else:
        print('Failed to set selecting!')
        exit()
    # GET CROSSOVER
    crossover_input = input("Crossover?\n1) Random - 1 point\n2) Random - 2 points\n")
    if crossover_input == '1':
        crossover = 'random 1'
        crossover_point = 0
    elif crossover_input == '2':
        crossover = 'random 2'
        crossover_point = 0
    else:
        print('Failed to set crossover!')
        exit()
    # GET MUTATION
    mutation = float(input("Mutation? "))


def genetic_algorithm(game_board, population, score_mode, selection_mode, crossover_mode, crossover_point,
                      mutation_prob):
    # population is creating....
    initial_population = generate_population(population, game_board, score_mode)
    # first generation
    first_generations = {1: initial_population}
    print('********************************')
    results = Genetic(first_generations, game_board, selection_mode, crossover_mode, crossover_point, mutation_prob,
                      score_mode)
    return results


def show_results(genetic):
    generation_average_scores = genetic.generation_average_scores
    generation_max_scores = genetic.generation_max_score
    generation_min_scores = genetic.generation_min_score

    plt.plot(*zip(*sorted(generation_average_scores.items())), color='r', label='Average')
    plt.plot(*zip(*sorted(generation_max_scores.items())), color='g', label='Maximum')
    plt.plot(*zip(*sorted(generation_min_scores.items())), color='b', label='Minimum')

    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    # Get Game's information from user
    get_input()
    # Reading game file from source
    game_board = get_level(game_file)
    # Apply genetic algorithm things
    result = genetic_algorithm(game_board, population, scoring, selecting, crossover, crossover_point, mutation)
    # Analysis result
    if len(result.generation_average_scores) != 10000:
        result_path_str = result.best_answer.string
        print("Goal chromosome: {}".format(result_path_str))
        print("Generation of Goal chromosome: {}".format(result.best_answer.generation))
        print("Score of Goal chromosome: {}".format(result.best_answer.score))
        # plot results
        show_results(result)
    else:
        print("Cannot find result! Limit is reached!")
