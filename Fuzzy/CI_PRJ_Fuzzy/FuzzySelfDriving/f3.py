from typing import List, Tuple, Set, Dict


class FuzzyController:
    """
    #todo
    write all the fuzzify,inference,defuzzify method in this class
    """

    def __init__(self):
        pass

    def compute_membership_dist(self, left_dist: int):
        close_l = 0
        moderate_l = 0
        far_l = 0
        if 50 >= left_dist >= 0:
            # in this part we in the clos_l part
            close_l = (-(1 / 50) * left_dist) + 1
        if 50 >= left_dist >= 35:
            moderate_l = (1 / 18) * left_dist + ((-1) * 32 / 18)
        if 65 >= left_dist >= 50:
            moderate_l = ((-1) * (1 / 15) * left_dist) + (65 / 15)
        if 100 >= left_dist >= 50:
            far_l = (1 / 50) * left_dist + 1
        vector_membership = [close_l, moderate_l, far_l]  # the elements relate to close_L , moderate_L , far_L
        return vector_membership


    def line_equation(self, point_1: Tuple[float, float], point_2: Tuple[float, float]):
        a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
        b = point_1[1] - (a * point_1[0])
        return a, b

    def compute_rule(self, vector_l, vector_r):
        close_l = vector_l[0]
        moderate_l = vector_l[1]
        far_l = vector_l[2]
        close_r = vector_r[0]
        moderate_r = vector_r[1]
        far_r = vector_r[2]

        rules: List[Tuple[str, float]] = list()
        rules.append(("low_right", min(close_l, moderate_r)))
        rules.append(("high_right", min(close_l, far_r)))
        rules.append(("low_left", min(moderate_l, close_r)))
        rules.append(("high_left", min(far_l, close_r)))
        rules.append(("nothing", min(moderate_l, moderate_r)))

        print(rules)

        return self.find_rotate_2(rules)


#FIXED
    def find_line_equations(self):
        line_equations: List[Tuple[str, Tuple[float, float], int]] = list()
        line_equations.append(("high_right", self.line_equation((-50, 0), (-20, 1)), 1))
        line_equations.append(("high_right",  self.line_equation((-20, 1), (-5, 0)), -1))
        line_equations.append(("low_right",self.line_equation((-20, 0), (-10, 1)), 1))
        line_equations.append(("low_right",self.line_equation((-10, 1), (0, 0)), -1))
        line_equations.append(("nothing",self.line_equation((-10, 0), (0, 1)), 1))
        line_equations.append(("nothing",self.line_equation((0, 1), (10, 0)), -1))
        line_equations.append(("low_left",self.line_equation((0, 0), (10, 1)), 1))
        line_equations.append(("low_left",self.line_equation((10, 1), (20, 0)), -1))
        line_equations.append(("high_left",self.line_equation((5, 0), (20, 1)), 1))
        line_equations.append(("high_left", self.line_equation((20, 1), (50, 0)), -1))
        return line_equations
    def find_y(self,rules: List[Tuple[str, float]], name_):
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

    def calculate_range_number_and_maximum_effective(self, rules: List[Tuple[str, float]],line_equations:List[Tuple[str, Tuple[float, float], int]]):
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



    def find_center_of_gravity(self,line_equations:List[Tuple[str, Tuple[float, float], int]], rules: List[Tuple[str, float]]):
        range_number = self.calculate_range_number_and_maximum_effective(rules, line_equations)
        sigma_tot = 0
        sigma_m_tot = 0
        for min_num, max_num, (a, b), name in range_number:
            s, s_ = self.integeral_final(min_num , max_num , (name, (a,b)), rules)
            sigma_tot += s
            sigma_m_tot += s_
        result = float(sigma_tot) / float(sigma_m_tot)
        print(f'result {result}')
        if result < 0:
            print("negative")
        return result
    def find_rotate_2(self, rules: List[Tuple[str, float]]):
        line_equations = self.find_line_equations()
        return self.find_center_of_gravity(line_equations, rules)

    def decide(self, left_dist, right_dist):
        """
        main method for doin all the phases and returning the final answer for rotation
        """
        vector_l = self.compute_membership_dist(left_dist)
        vector_r = self.compute_membership_dist(right_dist)
        return self.compute_rule(vector_l, vector_r)

