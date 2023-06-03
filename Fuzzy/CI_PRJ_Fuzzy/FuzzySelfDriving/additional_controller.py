class FuzzyGasController:
    def __init__(self):
        pass

    def fuzzify_center_dist(self, center_dist):
        close = 0
        moderate = 0
        far = 0
        if 50 >= center_dist >= 0:
            close = ((-1) * (1 / 50) * center_dist) + 1
        if 50 >= center_dist >= 40:
            moderate = ((1 / 10) * (center_dist)) - 4
        if 100 >= center_dist >= 50:
            moderate = (-1) * (1 / 50) * center_dist + 2
        if 200 >= center_dist >= 90:
            far = ((1 / 110) * center_dist) - (90 / 110)
        if center_dist >= 200:
            far = ((1 / 110) * center_dist) - (90 / 110)
        vector_membership = [close, moderate, far]
        return vector_membership

    def get_equation(self, p1, p2):
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - (a * p1[0])
        return a, b

    def get_lines(self):
        line_equations = []
        line_equations.append(("low", self.get_equation((0, 0), (5, 1)), 1))
        line_equations.append(("low", self.get_equation((5, 1), (10, 0)), -1))
        line_equations.append(("medium", self.get_equation((0, 0), (15, 1)), 1))
        line_equations.append(("medium", self.get_equation((15, 1), (30, 0)), -1))
        line_equations.append(("high", self.get_equation((25, 0), (30, 1)), 1))
        line_equations.append(("high", self.get_equation((30, 1), (90, 0)), -1))
        return line_equations

    def get_y(self,fuzzy_output, input_name):
        for name, y in fuzzy_output:
            if name == input_name:
                return y

    def aggregator(self, fuzzy_output,line_equations):
        rang_number = []
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
        #0,5
        if self.get_y(fuzzy_output, name[0]) < self.get_y(fuzzy_output, name[2]):
            rang_number.append((0, 5, (a[2], b[2]), name[2]))
        else:
            rang_number.append((0, 5, (a[0], b[0]), name[0]))
        #5,10
        if self.get_y(fuzzy_output, name[1]) < self.get_y(fuzzy_output, name[2]):
            rang_number.append((5, 10, (a[2], b[2]), name[2]))
        else:
            rang_number.append((5, 10, (a[1], b[1]), name[1]))
        #10,15
        rang_number.append((10, 15, (a[2], b[2]), name[2]))
        #15,25
        rang_number.append((15, 25, (a[3], b[3]), name[3]))
        #25,30
        if self.get_y(fuzzy_output, name[3]) < self.get_y(fuzzy_output, name[4]):
            rang_number.append((25, 30, (a[4], b[4]), name[4]))
        else:
            rang_number.append((25, 30, (a[3], b[3]), name[3]))
        #30,90
        rang_number.append((30, 90, (a[5], b[5]), name[5]))
        return rang_number

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

    def CoG_Finder(self, line_equations,fuzzy_output):
        range_number = self.aggregator(fuzzy_output, line_equations)
        sigma_tot = 0
        sigma_m_tot = 0
        for min_num, max_num, (a, b), name in range_number:
            s, s_ = self.integrate(min_num, max_num, (name, (a, b)), fuzzy_output)
            sigma_tot += s
            sigma_m_tot += s_
        if sigma_m_tot != 0:
            CoG = float(sigma_tot) / float(sigma_m_tot)
        else:
            return 0
        print(f'additional controller - CoG : {CoG}')
        if CoG < 0:
            print("additional controller - CoG is negative")
        return CoG


    def decide(self, center_dist):
        print('Deciding..')
        fuzzify_center = self.fuzzify_center_dist(center_dist)

        fuzzy_center_dist={}
        fuzzy_center_dist['close']=fuzzify_center[0]
        fuzzy_center_dist['moderate'] = fuzzify_center[1]
        fuzzy_center_dist['far'] = fuzzify_center[2]

        # Perform inference using fuzzy rules
        fuzzy_output = []

        # Rule 1: IF (center_dist IS close )  THEN  gas IS low
        fuzzy_output.append(("low", fuzzy_center_dist['close']))

        # Rule 2: IF (center_dist IS moderate )  THEN  gas IS medium
        fuzzy_output.append(('medium',fuzzy_center_dist['moderate']))

        # Rule 3: IF (center_dist IS far )  THEN  gas IS high
        fuzzy_output.append(('high',fuzzy_center_dist['far']))

        line_equations = self.get_lines()
        CoG = self.CoG_Finder(line_equations, fuzzy_output)

        return CoG

