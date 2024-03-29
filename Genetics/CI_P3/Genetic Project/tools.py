from chromosome import *
from game import *
import random


# 1- JAKHALI! 22 IS OK!
# 2- PARIDAN! 1 IS OK ONLY ONE TIME!
def rules_checker(current_string, new_char):
    res = True
    if len(current_string) >= 1 and new_char == "1" and current_string[-1] == "1":  # 2
        res = False
    elif len(current_string) >= 2 and new_char == "2" and current_string[-1] == "2" and current_string[-2] == "2":  # 1
        res = False

    return res


# GENERATING CHROMOSOME
def chromosome_generator(length):
    chromosome, i = "", 0
    while i < length:
        random_number = random.randint(1, 10)
        char = ""
        if random_number <= 5:
            char = "0"
            i += 1
        elif 5 < random_number <= 8:
            if rules_checker(chromosome, "1"):
                char = "1"
                i += 1
        elif 8 < random_number <= 10:
            if rules_checker(chromosome, "2"):
                char = "2"
                i += 1
        chromosome += char
    print(f'Generated Chromosome is {chromosome}')
    return chromosome


# READING GAME LEVEL
def get_level(test_case_name):
    path = "./attachments/levels/" + test_case_name
    with open(path, 'r') as file:
        game_plate = file.readline()

    return game_plate


# GENERATING POPULATION
def generate_population(population, game_board, score_mode):
    print(f'Game : {game_board}')
    game = Game(game_board)
    chromosome_length = len(game_board)
    chromosomes_theFirst = []
    print('Population is getting created...')
    for i in range(population):
        print(f'Creating Chromosome No.{i}')
        chromosome_string = chromosome_generator(chromosome_length)
        chromosome_score, chromosome_failure_points = game.get_score(chromosome_string, score_mode)
        chromosome = Chromosome(chromosome_string, chromosome_score, 1, chromosome_failure_points)
        chromosomes_theFirst.append(chromosome)
    print('Population Created!')
    return chromosomes_theFirst


def sort_score(array_of_chromosomes):
    # sort by their scores
    array_of_chromosomes.sort(key=lambda x: x.score, reverse=True)


def get_scores(array_of_chromosomes):
    scores = [chromosome.score for chromosome in array_of_chromosomes]
    return scores
