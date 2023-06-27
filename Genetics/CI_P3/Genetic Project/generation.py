from chromosome import *
from game import *
import random


# 1- JAKHALI! 22 IS OK!
# 2- PARIDAN! 1 IS OK ONLY!
def rules_check(current_string, new_char):
    res = True
    if len(current_string) >= 1 and new_char == "1" and current_string[-1] == "1":  # 2
        res = False
    elif len(current_string) >= 2 and new_char == "2" and current_string[-1] == "2" and current_string[-2] == "2":  # 1
        res = False

    return res


# GENERATING CHROMOSOME
def generate_chromosome(length):
    chromosome, i = "", 0
    while i < length:
        random_number = random.randint(1, 10)
        char = ""
        if random_number <= 5:
            char = "0"
            i += 1
        elif 5 < random_number <= 8:
            if rules_check(chromosome, "1"):
                char = "1"
                i += 1
        elif 8 < random_number <= 10:
            if rules_check(chromosome, "2"):
                char = "2"
                i += 1
        chromosome += char
    print(f'Generated Chromosome is {chromosome}')
    return chromosome


# READING GAME LEVEL
def read_level_game(test_case_name):
    path = "./attachments/levels/" + test_case_name
    with open(path, 'r') as file:
        game_plate = file.readline()

    return game_plate


def generate_initial_population(number_of_population, game_plate_string, score_mode):
    print(f'Game is now  : {game_plate_string}')
    game = Game(game_plate_string)
    chromosome_length = len(game_plate_string)
    array_of_chromosome = []
    for i in range(number_of_population):
        print(f'Creating Chromosome No.{i}')
        chromosome_string = generate_chromosome(chromosome_length)
        print('Getting score of it...')
        chromosome_score, chromosome_failure_points = game.get_score(chromosome_string, score_mode)
        chromosome = Chromosome(chromosome_string, chromosome_score, 1, chromosome_failure_points)
        array_of_chromosome.append(chromosome)
    print('Population Created!')
    return array_of_chromosome


def sort_by_score(array_of_chromosomes):
    # sort by their scores
    array_of_chromosomes.sort(key=lambda x: x.score, reverse=True)


def return_scores(array_of_chromosomes):
    scores = [chromosome.score for chromosome in array_of_chromosomes]
    return scores
