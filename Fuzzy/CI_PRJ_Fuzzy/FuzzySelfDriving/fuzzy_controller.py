class FuzzyController:
    def __init__(self):
        pass



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

    def find_y(self,rules:List[Tuple[str, float]], name_):
        for name, y in rules:
            if name == name_:
                return y

    def integeral_final(self, min_num, max_num,line,rules:List[Tuple[str,float]]):
        name, (a, b) = line
        i = min_num
        sigma = 0.0
        sigma_m = 0.0
        delta = 0.01
        while i < max_num:
            y = self.find_y(rules, name)
            y_prime = a * i + b
            if y_prime > y:
                sigma += y * delta * i
                sigma_m += y * delta
            else:
                sigma += y_prime * delta * i
                sigma_m += y_prime * delta
            i += delta

        return  sigma, sigma_m


    def aggregator(self, rules: List[Tuple[str, float]],line_equations:List[Tuple[str, Tuple[float, float], int]]):
        rang_number : List[Tuple[float, float, Tuple[float, float],str]] = list()
        # min number of range ,  max number of range , (a,b) that has max y , name
        name = list()
        a = list()
        b = list()
        validation = list()
        for i in range(len(line_equations)):
            name_, (a_, b_), validation_ = line_equations[i]
            name.append(name_)
            a.append(a_)
            b.append(b_)
            validation.append(validation_)

        print('###############################################')
        print(name)
        print(a)
        print(b)
        print(validation)
        print('##############################################')
        ####################
        rang_number.append((-50, -20, (a[0], b[0]),name[0]))  # -50,-20
        ######################
        if self.find_y(rules, name[1]) < self.find_y(rules,name[2]): # -20 , -10
            rang_number.append((-20, -10, (a[2], b[2]),name[2]))
        else:
            rang_number.append((-20, -10, (a[1], b[1]),name[1]))
        #########################
        max_y = max(self.find_y(rules, name[1]), self.find_y(rules,name[3]),self.find_y(rules, name[4])) # -10, -5
        for i in range(1, 5):
            if i == 2:
                continue
            if max_y == self.find_y(rules, name[i]):
                rang_number.append((-10, -5, (a[i], b[i]),name[i]))
        ##############
        if self.find_y(rules, name[3]) < self.find_y(rules, name[4]):# -5,0
            rang_number.append((-5, 0, (a[4], b[4]),name[4]))
        else:
            rang_number.append((-5, 0, (a[3], b[3]),name[3]))
        ########################
        if self.find_y(rules, name[5]) < self.find_y(rules, name[6]): # 0,5
            rang_number.append((0, 5, (a[6], b[6]),name[6]))
        else:
            rang_number.append((0, 5, (a[5], b[5]),name[5]))
        #######################
        max_y = max(self.find_y(rules, name[5]), self.find_y(rules, name[6]), self.find_y(rules, name[8])) #5,10
        for i in range(5, 8):
            if i == 7:
                continue
            if max_y == self.find_y(rules, name[i]):
                rang_number.append((5, 10, (a[i], b[i]), name[i]))
        ##############################
        if self.find_y(rules, name[7]) < self.find_y(rules, name[8]): #10,20
            rang_number.append((10, 20, (a[8], b[8]),name[8]))
        else:
            rang_number.append((10, 20, (a[7], b[7]),name[7]))
        ############################
        rang_number.append((20, 50, (a[9], b[9]),name[9])) # 20, 50
        return rang_number


    def CoG_Finder(self,line_equations:List[Tuple[str, Tuple[float, float], int]], fuzzy_output: List[Tuple[str, float]]):
        range_number = self.aggregator(fuzzy_output, line_equations)
        sigma_tot = 0
        sigma_m_tot = 0
        for min_num, max_num, (a, b), name in range_number:
            s, s_ = self.integeral_final(min_num , max_num , (name, (a,b)), fuzzy_output)
            sigma_tot += s
            sigma_m_tot += s_
        result = float(sigma_tot) / float(sigma_m_tot)
        print(f'result {result}')
        if result < 0:
            print("negative")
        return result

    # Interface
    def decide(self, relative_left_dist, relative_right_dist):
        # Call the fuzzification methods to obtain the fuzzy values for each input variable
        fuzzy_left_dist = self.fuzzify_left_dist(relative_left_dist)
        fuzzy_right_dist = self.fuzzify_right_dist(relative_right_dist)

        # Perform inference using fuzzy rules
        fuzzy_output = []

        # Rule 1: IF (d_L IS close_L) AND (d_R IS moderate_R) THEN Rotate IS low_right
        if 'close' in fuzzy_left_dist and 'moderate' in fuzzy_right_dist:
            fuzzy_output.append(("low_right", min(fuzzy_left_dist['close'], fuzzy_right_dist['moderate'])))
        else:
            fuzzy_output.append(("low_right", 0))

        # Rule 2: IF (d_L IS close_L) AND (d_R IS far_R) THEN Rotate IS high_right
        if 'close' in fuzzy_left_dist and 'far' in fuzzy_right_dist:
            fuzzy_output.append(('high_right', min(fuzzy_left_dist['close'], fuzzy_right_dist['far'])))
        else:
            fuzzy_output.append(('high_right',0))

        # Rule 3: IF (d_L IS moderate_L) AND (d_R IS close_R) THEN Rotate IS low_left
        if 'moderate' in fuzzy_left_dist and 'close' in fuzzy_right_dist:
            fuzzy_output.append(('low_left',min(fuzzy_left_dist['moderate'], fuzzy_right_dist['close'])))
        else:
            fuzzy_output.append(('low_left',0))

        # Rule 4: IF (d_L IS far_L) AND (d_R IS close_R) THEN Rotate IS high_left
        if 'far' in fuzzy_left_dist and 'close' in fuzzy_right_dist:
            fuzzy_output.append(('high_left', min(fuzzy_left_dist['far'], fuzzy_right_dist['close'])))
        else:
            fuzzy_output.append(('high_left', 0))

        # Rule 5: IF (d_L IS moderate_L) AND (d_R IS moderate_R) THEN Rotate IS nothing
        if 'moderate' in fuzzy_left_dist and 'moderate' in fuzzy_right_dist:
            fuzzy_output.append(('nothing', min(fuzzy_left_dist['moderate'], fuzzy_right_dist['moderate'])))
        else:
            fuzzy_output.append(('nothing', 0))

        #line equations:
        line_equations= []
        #y = (1 / 30) * x + 5 / 3
        line_equations.append(('high_right',1/40,5/3,1))
        #y = (-1 / 15) * x - 1 / 3
        line_equations.append(('high_right', -1 / 15, 1 / 3, -1))
        #y = (1 / 10) * x + 2
        line_equations.append(('low_right', 1 / 10, 2, 1))
        #y = (-1 / 10) * x - 1
        line_equations.append(('low_right', -1 / 10, -1, -1))
        #1/10 * x + 1= y
        line_equations.append(('nothing', 1 / 10, 1, 1))
        #
        line_equations.append(('nothing', -1 / 10, 1, -1))
        #
        line_equations.append(('low_left', 1 / 10, 0, 1))
        #
        line_equations.append(('low_left', -1 / 10, 2, -1))
        #1/15 * x1 - 1/3 = y
        line_equations.append(('high_left', 1 / 15, -1/3, 1))
        #-1/30 * x2 + 2/3 = y
        line_equations.append(('high_left', -1 / 30, 2 / 3, -1))

        print(line_equations)
        print(fuzzy_output)

        CoG = self.CoG_Finder(line_equations,fuzzy_output)

        return CoG