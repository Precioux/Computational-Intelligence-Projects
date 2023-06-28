class FuzzyController:
    def __init__(self):
        pass

    def fuzzify_left_dist(self, left_dist):
        close_l = 0
        moderate_l = 0
        far_l = 0
        if 50 >= left_dist >= 0:
            close_l = (-(1 / 50) * left_dist) + 1
        if 50 >= left_dist >= 35:
            moderate_l = (1 / 18) * left_dist + ((-1) * 32 / 18)
        if 65 >= left_dist >= 50:
            moderate_l = ((-1) * (1 / 15) * left_dist) + (65 / 15)
        if 100 >= left_dist >= 50:
            far_l = (1 / 50) * left_dist + 1
        vector_membership = [close_l, moderate_l, far_l]
        return vector_membership

    def fuzzify_right_dist(self, right_dist):
        close_r = 0
        moderate_r = 0
        far_r = 0
        if 50 >= right_dist >= 0:
            close_r = (-(1 / 50) * right_dist) + 1
        if 50 >= right_dist >= 35:
            moderate_r = (1 / 18) * right_dist + ((-1) * 32 / 18)
        if 65 >= right_dist >= 50:
            moderate_r = ((-1) * (1 / 15) * right_dist) + (65 / 15)
        if 100 >= right_dist >= 50:
            far_r = (1 / 50) * right_dist + 1
        vector_membership = [close_r, moderate_r, far_r]
        return vector_membership

    def get_y(self, fuzzy_output, input_name):
        for name, y in fuzzy_output:
            if name == input_name:
                return y

    def integrate(self, min_num, max_num, line, fuzzy_output):
        name, (a, b) = line
        i = min_num
        sigma = 0.0
        sigma_m = 0.0
        delta = 0.01
        while i < max_num:
            y = self.get_y(fuzzy_output, name)
            y_prime = a * i + b
            if y_prime > y:
                sigma += y * delta * i
                sigma_m += y * delta
            else:
                sigma += y_prime * delta * i
                sigma_m += y_prime * delta
            i += delta

        return sigma, sigma_m

    def aggregator(self, fuzzy_output, line_equations):
        # min number of range ,  max number of range , (a,b) that has max y , name
        name = list()
        a = list()
        b = list()
        validation = list()
        for i in range(len(line_equations)):
            name_i, (a_i, b_i), validation_i = line_equations[i]
            name.append(name_i)
            a.append(a_i)
            b.append(b_i)
            validation.append(validation_i)
        rang_number = []
        # -50,-20
        rang_number.append((-50, -20, (a[0], b[0]), name[0]))
        # -20 , -10
        if self.get_y(fuzzy_output, name[1]) < self.get_y(fuzzy_output, name[2]):
            rang_number.append((-20, -10, (a[2], b[2]), name[2]))
        else:
            rang_number.append((-20, -10, (a[1], b[1]), name[1]))
        # -10, -5
        max_y = max(self.get_y(fuzzy_output, name[1]), self.get_y(fuzzy_output, name[3]),
                    self.get_y(fuzzy_output, name[4]))
        for i in range(1, 5):
            if i == 2:
                continue
            if max_y == self.get_y(fuzzy_output, name[i]):
                rang_number.append((-10, -5, (a[i], b[i]), name[i]))
        # -5,0
        if self.get_y(fuzzy_output, name[3]) < self.get_y(fuzzy_output, name[4]):  # -5,0
            rang_number.append((-5, 0, (a[4], b[4]), name[4]))
        else:
            rang_number.append((-5, 0, (a[3], b[3]), name[3]))
        # 0,5
        if self.get_y(fuzzy_output, name[5]) < self.get_y(fuzzy_output, name[6]):
            rang_number.append((0, 5, (a[6], b[6]), name[6]))
        else:
            rang_number.append((0, 5, (a[5], b[5]), name[5]))
        # 5 , 10
        max_y = max(self.get_y(fuzzy_output, name[5]), self.get_y(fuzzy_output, name[6]),
                    self.get_y(fuzzy_output, name[8]))
        for i in range(5, 8):
            if i == 7:
                continue
            if max_y == self.get_y(fuzzy_output, name[i]):
                rang_number.append((5, 10, (a[i], b[i]), name[i]))
        # 10,20
        if self.get_y(fuzzy_output, name[7]) < self.get_y(fuzzy_output, name[8]):  # 10,20
            rang_number.append((10, 20, (a[8], b[8]), name[8]))
        else:
            rang_number.append((10, 20, (a[7], b[7]), name[7]))
        # 20, 50
        rang_number.append((20, 50, (a[9], b[9]), name[9]))
        return rang_number

    def CoG_Finder(self, line_equations, fuzzy_output):
        range_number = self.aggregator(fuzzy_output, line_equations)
        sigma_tot = 0
        sigma_m_tot = 0
        for min_num, max_num, (a, b), name in range_number:
            s, s_ = self.integrate(min_num, max_num, (name, (a, b)), fuzzy_output)
            sigma_tot += s
            sigma_m_tot += s_
        CoG = float(sigma_tot) / float(sigma_m_tot)
        print(f'fuzzy controller - CoG :  {CoG}')
        if CoG < 0:
            print("fuzzy controller - CoG is negative")
        return CoG

    def get_equation(self, p1, p2):
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - (a * p1[0])
        return a, b

    def get_lines(self):
        line_equations = []
        line_equations.append(("high_right", self.get_equation((-50, 0), (-20, 1)), 1))
        line_equations.append(("high_right", self.get_equation((-20, 1), (-5, 0)), -1))
        line_equations.append(("low_right", self.get_equation((-20, 0), (-10, 1)), 1))
        line_equations.append(("low_right", self.get_equation((-10, 1), (0, 0)), -1))
        line_equations.append(("nothing", self.get_equation((-10, 0), (0, 1)), 1))
        line_equations.append(("nothing", self.get_equation((0, 1), (10, 0)), -1))
        line_equations.append(("low_left", self.get_equation((0, 0), (10, 1)), 1))
        line_equations.append(("low_left", self.get_equation((10, 1), (20, 0)), -1))
        line_equations.append(("high_left", self.get_equation((5, 0), (20, 1)), 1))
        line_equations.append(("high_left", self.get_equation((20, 1), (50, 0)), -1))
        return line_equations

    # Interface
    def decide(self, relative_left_dist, relative_right_dist):
        print('Deciding...')
        # Call the fuzzification methods to obtain the fuzzy values for each input variable
        fuzzy_left = self.fuzzify_left_dist(relative_left_dist)
        fuzzy_right = self.fuzzify_right_dist(relative_right_dist)

        fuzzy_left_dist = {}
        fuzzy_left_dist['close'] = fuzzy_left[0]
        fuzzy_left_dist['moderate'] = fuzzy_left[1]
        fuzzy_left_dist['far'] = fuzzy_left[2]

        fuzzy_right_dist = {}
        fuzzy_right_dist['close'] = fuzzy_right[0]
        fuzzy_right_dist['moderate'] = fuzzy_right[1]
        fuzzy_right_dist['far'] = fuzzy_right[2]

        # Perform interface using fuzzy rules
        fuzzy_output = []

        # Rule 1: IF (d_L IS close_L) AND (d_R IS moderate_R) THEN Rotate IS low_right
        fuzzy_output.append(("low_right", min(fuzzy_left_dist['close'], fuzzy_right_dist['moderate'])))

        # Rule 2: IF (d_L IS close_L) AND (d_R IS far_R) THEN Rotate IS high_right
        fuzzy_output.append(('high_right', min(fuzzy_left_dist['close'], fuzzy_right_dist['far'])))

        # Rule 3: IF (d_L IS moderate_L) AND (d_R IS close_R) THEN Rotate IS low_left
        fuzzy_output.append(('low_left', min(fuzzy_left_dist['moderate'], fuzzy_right_dist['close'])))

        # Rule 4: IF (d_L IS far_L) AND (d_R IS close_R) THEN Rotate IS high_left
        fuzzy_output.append(('high_left', min(fuzzy_left_dist['far'], fuzzy_right_dist['close'])))

        # Rule 5: IF (d_L IS moderate_L) AND (d_R IS moderate_R) THEN Rotate IS nothing
        fuzzy_output.append(('nothing', min(fuzzy_left_dist['moderate'], fuzzy_right_dist['moderate'])))

        # #line equations:
        line_equations = self.get_lines()
        CoG = self.CoG_Finder(line_equations, fuzzy_output)

        return CoG
