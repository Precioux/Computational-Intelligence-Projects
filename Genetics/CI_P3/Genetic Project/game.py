import copy


class Game:
    def __init__(self, game_plate):
        # Get strings as game_plate
        # Store level length to determine if a sequence of action passes all the steps
        self.game_plate = game_plate

    def calculate_maximum_substring_length(self, failure_points):
        game_plate = self.game_plate
        length_game_plate = len(game_plate)
        substring_length = []

        failure_points.insert(0, 0)
        failure_points.append(length_game_plate)

        for i in range(1, len(failure_points)):
            length = failure_points[i] - failure_points[i - 1]
            substring_length.append(length)

        return max(substring_length)

    def get_score(self, actions, get_score_mode):
        # Get an action sequence and determine the steps taken/score
        # Return a tuple, the first one indicates if these actions result in victory
        # and the second one shows the steps taken
        print('Getting score function:')
        game_plate = self.game_plate
        length_game_plate = len(game_plate)
        failure_points = []
        steps = 0
        scores = 0
        for i in range(length_game_plate):
            print(f'Current step: {game_plate[i]}')
            current_step = game_plate[i]
            if current_step == '_':  # ok to walk right
                print('Walking..')
                steps += 1
            elif current_step == 'G':  # Gumpa is here!
                print('Gumpa!')
                if actions[i - 1] == '1':  # if you have jumped thats fine!
                    print('jumped!')
                    steps += 1
                elif i - 2 >= 0 and actions[i - 2] == "1":  # killing Gumpa!
                    print('killing gumpa!')
                    steps += 1
                    scores += 2
                else:
                    print('failed!')
                    failure_points.append(i)  # any other choice is failure!

            elif current_step == "M":  #
                print('Mushroom!')
                if actions[i - 1] != "1":  # if you havent jump from mushroom
                    print('Mushroom is delicious!')
                    scores += 2
                print('Walking...')
                steps += 1
            elif current_step == 'L':
                print('Lakipo!')
                if i - 2 >= 0:
                    if actions[i - 1] == "2" and actions[i - 2] != "1":  # if JAKHALI and not jump!
                        print('Survived from lakipo!')
                        steps += 1
                    else:
                        print('failed!')
                        failure_points.append(i)
                elif actions[i - 1] == '2':
                    print('Survived from lakipo!')
                    steps += 1
                else:
                    print('failed!')
                    failure_points.append(i)
            else:
                print('failed!')
                failure_points.append(i)

        if actions[-1] == "1":
            scores += 1  # game is done!

        print(f'scores : {scores}')
        print(f'failure_points : {failure_points}')
        failure_points_copy = copy.deepcopy(failure_points)
        maximum_substring_length = self.calculate_maximum_substring_length(failure_points_copy)

        if get_score_mode == "1": # with calculating winning points
            if len(failure_points) == 0:
                scores += 5  # if there is no failure points then chromosome is a winner
        # we score by summing max substring + scores!
        return maximum_substring_length + scores, failure_points
