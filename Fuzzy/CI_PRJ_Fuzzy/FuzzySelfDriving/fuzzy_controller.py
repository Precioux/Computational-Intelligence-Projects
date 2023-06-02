class FuzzyController:
    def __init__(self):
        pass

    def fuzzify_rotation(self, relative_rotation):
        fuzzy_values = []
        value = relative_rotation

        if value < -50 or value > 50:
            return fuzzy_values
        # high_right
        if -50 <= value <= -20:
            membership = (value + 50) / 30
            fuzzy_values.append(('high_right', membership))
        elif -20 <= value <= -5:
            membership = 1 - abs((value + 5) / 15)
            fuzzy_values.append(('high_right', membership))

        # low_right
        if -20 <= value <= -10:
            membership = (value + 20) / 10
            fuzzy_values.append(('low_right', membership))
        elif -10 <= value <= 0:
            membership = 1 - abs((value + 10) / 10)
            fuzzy_values.append(('low_right', membership))

        # nothing
        if -10 <= value <= 0:
            membership = (value + 10) / 10
            fuzzy_values.append(('nothing', membership))
        elif 0 <= value <= 10:
            membership = 1 - abs(value / 10)
            fuzzy_values.append(('nothing', membership))

        # low_left
        if 0 <= value <= 10:
            membership = value / 10
            fuzzy_values.append(('low_left', membership))
        elif 10 <= value <= 20:
            membership = 1 - abs((value - 10) / 10)
            fuzzy_values.append(('low_left', membership))

        # high_left
        if 5 <= value <= 20:
            membership = (value - 5) / 15
            fuzzy_values.append(('high_left', membership))
        elif 20 <= value <= 50:
            membership = 1 - abs((value - 20) / 30)
            fuzzy_values.append(('high_left', membership))

        return fuzzy_values

    def fuzzify_left_dist(self, relative_left_dist):
        fuzzy_values = {}
        value = relative_left_dist

        if value <= 0 or value >= 100:
            return fuzzy_values

        if value <= 50:
            membership = 1 - (value / 50)
            fuzzy_values['close'] = membership
        else:
            membership = (value - 50) / 50
            fuzzy_values['far'] = membership

        if 35 <= value <= 65:
            membership = 1 - abs((value - 50) / 15)
            fuzzy_values['moderate'] = membership

        return fuzzy_values

    def fuzzify_right_dist(self, relative_right_dist):
        fuzzy_values = {}
        value = relative_right_dist

        if value <= 0 or value >= 100:
            return fuzzy_values

        if value <= 50:
            membership = 1 - (value / 50)
            fuzzy_values['close'] = membership
        else:
            membership = (value - 50) / 50
            fuzzy_values['far'] = membership

        if 35 <= value <= 65:
            membership = 1 - abs((value - 50) / 15)
            fuzzy_values['moderate'] = membership

        return fuzzy_values

    # Interface
    def decide(self, relative_left_dist, relative_right_dist):
        # Call the fuzzification methods to obtain the fuzzy values for each input variable
        fuzzy_left_dist = self.fuzzify_left_dist(relative_left_dist)
        fuzzy_right_dist = self.fuzzify_right_dist(relative_right_dist)
        print(fuzzy_left_dist)
        print(fuzzy_right_dist)
        print('result:')
        # Perform inference using fuzzy rules
        fuzzy_output = {}


        # Rule 1: IF (d_L IS close_L) AND (d_R IS moderate_R) THEN Rotate IS low_right
        if 'close' in fuzzy_left_dist and 'moderate' in fuzzy_right_dist:
            fuzzy_output['low_right'] = min(fuzzy_left_dist['close'], fuzzy_right_dist['moderate'])
        else:
            fuzzy_output['low_right'] = 0

        # Rule 2: IF (d_L IS close_L) AND (d_R IS far_R) THEN Rotate IS high_right
        if 'close' in fuzzy_left_dist and 'far' in fuzzy_right_dist:
            fuzzy_output['high_right'] = min(fuzzy_left_dist['close'], fuzzy_right_dist['far'])
        else:
            fuzzy_output['high_right'] = 0

        # Rule 3: IF (d_L IS moderate_L) AND (d_R IS close_R) THEN Rotate IS low_left
        if 'moderate' in fuzzy_left_dist and 'close' in fuzzy_right_dist:
            fuzzy_output['low_left'] = min(fuzzy_left_dist['moderate'], fuzzy_right_dist['close'])
        else:
            fuzzy_output['low_left'] = 0

        # Rule 4: IF (d_L IS far_L) AND (d_R IS close_R) THEN Rotate IS high_left
        if 'far' in fuzzy_left_dist and 'close' in fuzzy_right_dist:
            fuzzy_output['high_left'] = min(fuzzy_left_dist['far'], fuzzy_right_dist['close'])
        else:
            fuzzy_output['high_left'] = 0

        # Rule 5: IF (d_L IS moderate_L) AND (d_R IS moderate_R) THEN Rotate IS nothing
        if 'moderate' in fuzzy_left_dist and 'moderate' in fuzzy_right_dist:
            fuzzy_output['nothing'] = min(fuzzy_left_dist['moderate'], fuzzy_right_dist['moderate'])
        else:
            fuzzy_output['nothing'] = 0

        print(fuzzy_output)
        return 0



    # def decide(self, relative_left_dist, relative_right_dist):
    #     # Call the fuzzification methods to obtain the fuzzy values for each input variable
    #     fuzzy_left_dist = self.fuzzify_left_dist(relative_left_dist)
    #     fuzzy_right_dist = self.fuzzify_right_dist(relative_right_dist)
    #
    #     print(f'left')
    #     print(fuzzy_left_dist)
    #     print('right')
    #     print(fuzzy_right_dist)
    #
    #     return 0
