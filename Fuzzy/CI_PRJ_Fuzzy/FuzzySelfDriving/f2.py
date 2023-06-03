class FuzzyController:
    def __init__(self):
        pass

    def line_down(self, x1, x2, value):
        y = value * (x2 - x1)
        return y

    def line_up(self, x1, x2, value):
        y = (value / 2) * (x2 ^ 2 - x1 ^ 2)
        return y

    def high_left_integral_down(self, x1, x2, side):
        y = 0
        # high_right
        if side == 0:
            y = (1 / 30) * (x2 ^ 2 - x1 ^ 2) + (1 / 3) * (x2 - x1)
        if side == 1:
            y = (-1 / 60) * (x2 ^ 2 - x1 ^ 2) - (2 / 3) * (x2 - x1)

        return y

    def high_left_integral_up(self, x1, x2, side):
        y = 0
        # high_right
        if side == 0:
            y = (1 / 45) * (x2 ^ 3 - x1 ^ 3) - (1 / 6) * (x2 ^ 2 - x1 ^ 2)
        if side == 1:
            y = (-1 / 90) * (x2 ^ 3 - x1 ^ 3) + (1 / 3) * (x2 ^ 2 - x1 ^ 2)

        return y

    def high_left_value_1(self, y):
        # high_left
        x2 = (y - 2 / 3) * -30
        return x2

    def high_left_value_0(self, y):
        # high_left
        x1 = (y + 1 / 3) * 15
        return x1

    def low_left_integral_down(self, x1, x2, side):
        y = 0
        # low_right
        if side == 0:
            y = (1 / 20) * (x2 ^ 2 - x1 ^ 2)
        if side == 1:
            y = (-1 / 20) * (x2 ^ 2 - x1 ^ 2) - 2 * (x2 - x1)

        return y

    def low_left_integral_up(self, x1, x2, side):
        y = 0
        # low_right
        if side == 0:
            y = (1 / 30) * (x2 ^ 3 - x1 ^ 3)
        if side == 1:
            y = (-1 / 30) * (x2 ^ 3 - x1 ^ 3) - (x2 ^ 2 - x1 ^ 2)

        return y

    def low_left_value_1(self, y):
        x2 = (y - 2) * -10
        return x2

    def low_left_value_0(self, y):
        x1 = y * 10
        return x1

    def nothing_integral_down(self, x1, x2, side):
        y = 0
        # nothing
        if side == 0:
            y = (1 / 20) * (x2 ^ 2 - x1 ^ 2) + (x2 - x1)
        if side == 1:
            y = (-1 / 20) * (x2 ^ 2 - x1 ^ 2) - (x2 - x1)

        return y

    def nothing_integral_up(self, x1, x2, side):
        y = 0
        # nothing
        if side == 0:
            y = (1 / 30) * (x2 ^ 3 - x1 ^ 3) + 0.5 * (x2 ^ 2 - x1 ^ 2)
        if side == 1:
            y = (-1 / 30) * (x2 ^ 3 - x1 ^ 3) - (1 / 2) * (x2 ^ 2 - x1 ^ 2)

        return y

    def nothing_value_1(self, y):
        # nothing
        x2 = (y - 1) * -10
        return x2

    def nothing_value_0(self, y):
        # nothing
        x1 = (y - 1) * 10
        return x1

    def low_right_integral_down(self, x1, x2, side):
        y = 0
        # low_right
        if side == 0:
            y = (1 / 20) * (x2 ^ 2 - x1 ^ 2) + 2 * (x2 - x1)
        if side == 1:
            y = (-1 / 20) * (x2 ^ 2 - x1 ^ 2) - (x2 - x1)

        return y

    def low_right_integral_up(self, x1, x2, side):
        y = 0
        # low_right
        if side == 0:
            y = (1 / 30) * (x2 ^ 3 - x1 ^ 3) + (x2 ^ 2 - x1 ^ 2)
        if side == 1:
            y = (-1 / 30) * (x2 ^ 3 - x1 ^ 3) - (1 / 2) * (x2 ^ 2 - x1 ^ 2)

        return y

    def low_right_value_1(self, y):
        # low_right
        x2 = (y + 1) * -10

        return x2

    def low_right_value_x_1(self, x):
        y = (-1 / 10) * x - 1
        return y

    def low_right_value_0(self, y):
        # low_right
        x1 = (y - 2) * 10

        return x1

    def low_right_value_x_0(self, x):
        y = (1 / 10) * x + 2
        return y

    def high_right_integral_down(self, x1, x2, side):
        y = 0
        # high_right
        if side == 0:
            y = (1 / 60) * (x2 ^ 2 - x1 ^ 2) + (5 / 3) * (x2 - x1)
        if side == 1:
            y = (-1 / 30) * (x2 ^ 2 - x1 ^ 2) - (1 / 3) * (x2 - x1)

        return y

    def high_right_integral_up(self, x1, x2, side):
        y = 0
        # high_right
        if side == 0:
            y = (1 / 90) * (x2 ^ 3 - x1 ^ 3) + (5 / 6) * (x2 ^ 2 - x1 ^ 2)
        if side == 1:
            y = (-1 / 45) * (x2 ^ 3 - x1 ^ 3) - (1 / 6) * (x2 ^ 2 - x1 ^ 2)

        return y

    def high_right_value_x_1(self, x):
        # high_right
        y = (-1 / 15) * x - 1 / 3
        return y

    def high_right_value_1(self, y):
        # high_right
        x2 = (y + 1 / 3) * -15
        return x2

    def high_right_value_x_0(self, x):
        # high_right
        y = (1 / 30) * x + 5 / 3
        return y

    def high_right_value_0(self, y):
        # high_right
        x1 = (y - 5 / 3) * 30
        return x1

    def defuzzify(self, fuzzy_output):
        print('input')
        print(fuzzy_output)
        up = 0
        down = 0

        # high_right
        x_hr = []
        if fuzzy_output['high_right']:
            x_hr[0] = -50
            x_hr[1] = self.high_right_value_0(fuzzy_output['high_right'])
            x_hr[2] = self.high_right_value_1(fuzzy_output['high_right'])
            x_hr[3] = -5
        else:
            for i in range(0, 4):
                x_hr[i] = 0

        # low_right
        x_lr = []
        if fuzzy_output['low_right']:
            x_lr[0] = -20
            x_lr[1] = self.high_right_value_0(fuzzy_output['low_right'])
            x_lr[2] = self.high_right_value_1(fuzzy_output['low_right'])
            x_lr[3] = 0
        else:
            for i in range(0, 4):
                x_lr[i] = 0

        # nothing
        x_n = []
        if fuzzy_output['nothing']:
            x_n[0] = -10
            x_n[1] = self.nothing_value_0(fuzzy_output['nothing'])
            x_n[2] = self.nothing_value_1(fuzzy_output['nothing'])
            x_n[3] = 10
        else:
            for i in range(0, 4):
                x_n[i] = 0

        # low_left
        x_ll = []
        if fuzzy_output['low_left']:
            x_ll[0] = 0
            x_ll[1] = self.low_left_value_0(fuzzy_output['low_left'])
            x_ll[2] = self.low_left_value_1(fuzzy_output['low_left'])
            x_ll[3] = 20
        else:
            for i in range(0, 4):
                x_ll[i] = 0

        # high_left
        x_hl = []
        if fuzzy_output['high_left']:
            x_hl[0] = 5
            x_hl[1] = self.high_left_value_0(fuzzy_output['high_left'])
            x_hl[2] = self.high_left_value_1(fuzzy_output['high_left'])
            x_hl[3] = 50
        else:
            for i in range(0, 4):
                x_hl[i] = 0
        up = 0
        down = 0

        if fuzzy_output['high_right'] != 0 and fuzzy_output['low_right'] != 0 and fuzzy_output['nothing'] != 0 and \
                fuzzy_output['high_left'] != 0 and fuzzy_output['low_left'] != 0:
            # -50 to x_hr[1]
            up = up + self.high_right_integral_up(x_hr[0], x_hr[1], 0)
            down = down + self.high_right_integral_down(x_hr[0], x_hr[1], 0)

            # x_hr[1] to -20
            up = up + self.line_up(x_hr[1], -20, fuzzy_output['high_right'])
            down = down + self.line_down(x_hr[1], -20, fuzzy_output['high_right'])

            # -20 to x_hr[2]
            if fuzzy_output['high_right'] > fuzzy_output['low_right']:
                up = up + self.line_up(-20, x_hr[2], fuzzy_output['high_right'])
                down = down + self.line_down(-20, x_hr[2], fuzzy_output['high_right'])

                if x_hr[2] <= -14:
                    up = up + self.high_right_integral_up(x_hr[2], -14, 1)
                    down = down + self.high_right_integral_down(x_hr[2], -14, 1)
                    if -14 <= x_lr[1] <= -10:
                        up = up + self.low_right_integral_up(-14, x_lr[1], 0)
                        down = down + self.low_right_integral_down(-14, x_lr[1], 0)

                        up = up + self.line_up(x_lr[1], x_lr[2], fuzzy_output['low_right'])
                        down = down + self.line_down(x_lr[1],  x_lr[2], fuzzy_output['low_right'])

                        up = up + self.low_right_integral_up(x_lr[2], -5, 1)
                        down = down + self.low_right_integral_down(x_lr[2], -5, 1)
                    else:
                        x_in_14_10 = self.high_right_value_1(fuzzy_output['low_right'])
                        if -14 <= x_in_14_10 <= -10:
                            up = up + self.high_right_integral_up(-14,x_in_14_10, 1)
                            down = down + self.high_right_integral_down(-14,x_in_14_10, 1)

                            up = up + self.line_up(x_in_14_10,-10, fuzzy_output['low_right'])
                            down = down + self.line_down(x_in_14_10,-10, fuzzy_output['low_right'])
                            if fuzzy_output['low_right'] >= 0.5:
                                up = up + self.line_up(x_in_14_10, x_lr[2], fuzzy_output['low_right'])
                                down = down + self.line_down(x_in_14_10,x_lr[2], fuzzy_output['low_right'])

                                up = up + self.low_right_integral_up(x_lr[2], -5, 1)
                                down = down + self.low_right_integral_down(x_lr[2], -5, 1)
                            else:
                                x_in_10_5 = self.high_right_value_1(fuzzy_output['low_right'])


                        else:
                            up = up + self.high_right_integral_up(-14,-10, 1)
                            down = down + self.high_right_integral_down(-14,-10, 1)





                else:
                    up = up + self.line_up(x_hr[2], -14, fuzzy_output['high_right'])
                    down = down + self.line_down(x_hr[2], -14, fuzzy_output['high_right'])

            # 20 to x_hl[2]
            up = up + self.line_up(20, x_hl[2], fuzzy_output['high_left'])
            down = down + self.line_down(20, x_hl[2], fuzzy_output['high_left'])

            # x_hl[2] to 50
            up = up + self.high_left_integral_up(x_hl[2], x_hl[3], 1)
            down = down + self.high_left_integral_down(x_hr[2], x_hr[3], 1)

        # first collision
        # coll_hl_r = self.low_left_value_0(fuzzy_output['high_right'])
        # if coll_hl_r < x_hr[2] and fuzzy_output['high_right'] < fuzzy_output['low_right']:
        #     # before collision
        #     up = up + self.line_up(-20, coll_hl_r, fuzzy_output['high_right'])
        #     down = down + self.line_down(-20, coll_hl_r, fuzzy_output['high_right'])
        #     # after collision
        #     up = up + self.low_right_integral_up(coll_hl_r, x_lr[1], 0)
        #     down = down + self.low_right_integral_down(coll_hl_r, x_lr[1], 0)
        #     # till -10
        #     up = up + self.line_up(x_lr[1], -10, fuzzy_output['low_right'])
        #     down = down + self.line_down(x_lr[1], -10, fuzzy_output['low_right'])
        #
        # if coll_hl_r < x_hr[2] and fuzzy_output['high_right'] > fuzzy_output['low_right']:
        #     # before collision
        #     up = up + self.line_up(-20, x_hr[2], fuzzy_output['high_right'])
        #     down = down + self.line_down(-20, x_hr[2], fuzzy_output['high_right'])
        #
        #     up = up + self.high_right_integral_up(x_hr[2], -10, 1)
        #     down = down + self.high_right_integral_down(x_hr[2], -10, 1)
        #
        # if coll_hl_r > x_hr[2] and fuzzy_output['high_right'] > fuzzy_output['low_right']:
        #     # before collision
        #     up = up + self.line_up(-20,  x_hr[3], fuzzy_output['high_right'])
        #     down = down + self.line_down(-20,  x_hr[3], fuzzy_output['high_right'])
        #     # after collision
        #     up = up + self.high_right_integral_up(x_hr[3],x_lr[1], 1)
        #     down = down + self.high_right_integral_down(x_hr[3],x_lr[1], 1)
        #     # till -10
        #     up = up + self.line_up(x_lr[1], -10, fuzzy_output['low_right'])
        #     down = down + self.line_down(x_lr[1], -10, fuzzy_output['low_right'])
        #
        # if coll_hl_r > x_hr[2] and fuzzy_output['high_right'] < fuzzy_output['low_right']:
        #     # before collision
        #     up = up + self.line_up(-20, x_hr[3], fuzzy_output['high_right'])
        #     down = down + self.line_down(-20, x_hr[3], fuzzy_output['high_right'])
        #     # after collision
        #     up = up + self.high_right_integral_up(x_hr[3], -14, 1)
        #     down = down + self.high_right_integral_down(x_hr[3], -14, 1)
        #     # till -10
        #     up = up + self.low_right_integral_up(-14, -10, 0)
        #     down = down + self.low_right_integral_down(-14, -10, 0)
        #
        # # if same
        # if coll_hl_r == x_hr[2]:
        #     up = up + self.line_up(-20,  x_lr[3], fuzzy_output['high_right'])
        #     down = down + self.line_down(-20,  x_lr[3], fuzzy_output['high_right'])
        #
        # #second collision
        # coll_lr_n = self.nothing_value_0(fuzzy_output['low_right'])

    # def defuzzify(self, fuzzy_output):
    #     print('fuzzy input: ')
    #     print(fuzzy_output)
    #     up = 0
    #     down = 0
    #
    #     for x in range(-50, 51):
    #         if -50 <= x <= -20:
    #             if fuzzy_output['high_right']:
    #                 x1 = self.high_right_value_0(float(fuzzy_output['high_right']))
    #                 if x < x1:
    #                     up = up + self.high_right_integral_up(float(x), float(x + 1), 0)
    #                     down = down + self.high_right_integral_down(float(x), float(x + 1), 0)
    #                 if x >= x1:
    #                     up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                     down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #         if -20 < x <= -10:
    #             if fuzzy_output['high_right'] and fuzzy_output['low_right']:
    #                 if fuzzy_output['high_right'] > fuzzy_output['low_right']:
    #                     y1 = self.high_right_value_x_1(float(x))
    #                     if fuzzy_output['high_right'] <= y1:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                     if fuzzy_output['high_right'] > y1 >= fuzzy_output['low_right']:
    #                         up = up + self.high_right_integral_up(float(x), float(x + 1), 1)
    #                         down = down + self.high_right_integral_down(float(x), float(x + 1), 1)
    #                     if y1 < fuzzy_output['low_right']:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                 if fuzzy_output['high_right'] == fuzzy_output['low_right']:
    #                     up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                     down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                 if fuzzy_output['high_right'] < fuzzy_output['low_right']:
    #                     y2 = self.low_right_value_x_0(float(x))
    #                     if fuzzy_output['high_right'] >= y2:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                     if fuzzy_output['high_right'] < y2 < fuzzy_output['low_right']:
    #                         up = up + self.low_right_integral_up(float(x), float(x + 1), 0)
    #                         down = down + self.low_right_integral_down(float(x), float(x + 1), 0)
    #                     if y2 <= fuzzy_output['low_right']:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #             if fuzzy_output['low_right'] and not fuzzy_output['high_right']:
    #                 y2 = self.low_right_value_x_0(float(x))
    #                 if y2 < fuzzy_output['low_right']:
    #                     up = up + self.low_right_integral_up(float(x), float(x + 1), 0)
    #                     down = down + self.low_right_integral_down(float(x), float(x + 1), 0)
    #                 else:
    #                     up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                     down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #             if fuzzy_output['high_right'] and not fuzzy_output['low_right']:
    #                 y2 = self.high_right_value_x_1(float(x))
    #                 if y2 < fuzzy_output['high_right']:
    #                     up = up + self.high_right_integral_up(float(x), float(x + 1), 1)
    #                     down = down + self.high_right_integral_down(float(x), float(x + 1), 1)
    #                 else:
    #                     up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                     down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #
    #         if -10 < x <= -5:
    #             if not fuzzy_output['nothing']:
    #                 if fuzzy_output['high_right'] and fuzzy_output['low_right']:
    #                     if fuzzy_output['high_right'] > fuzzy_output['low_right']:
    #                         y1 = self.high_right_value_x_1(float(x))
    #                         if fuzzy_output['high_right'] <= y1:
    #                             up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                             down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                         if fuzzy_output['high_right'] > y1 >= fuzzy_output['low_right']:
    #                             up = up + self.high_right_integral_up(float(x), float(x + 1), 1)
    #                             down = down + self.high_right_integral_down(float(x), float(x + 1), 1)
    #                         if y1 < fuzzy_output['low_right']:
    #                             up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                             down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                     if fuzzy_output['high_right'] == fuzzy_output['low_right']:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                     if fuzzy_output['high_right'] < fuzzy_output['low_right']:
    #                         y2 = self.low_right_value_x_0(float(x))
    #                         if fuzzy_output['high_right'] >= y2:
    #                             up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                             down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                         if fuzzy_output['high_right'] < y2 < fuzzy_output['low_right']:
    #                             up = up + self.low_right_integral_up(float(x), float(x + 1), 0)
    #                             down = down + self.low_right_integral_down(float(x), float(x + 1), 0)
    #                         if y2 <= fuzzy_output['low_right']:
    #                             up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                             down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                 if fuzzy_output['low_right'] and not fuzzy_output['high_right']:
    #                     y2 = self.low_right_value_x_0(float(x))
    #                     if y2 < fuzzy_output['low_right']:
    #                         up = up + self.low_right_integral_up(float(x), float(x + 1), 0)
    #                         down = down + self.low_right_integral_down(float(x), float(x + 1), 0)
    #                     else:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['low_right']))
    #
    #                 if fuzzy_output['high_right'] and not fuzzy_output['low_right']:
    #                     y2 = self.high_right_value_x_1(float(x))
    #                     if y2 < fuzzy_output['high_right']:
    #                         up = up + self.high_right_integral_up(float(x), float(x + 1), 1)
    #                         down = down + self.high_right_integral_down(float(x), float(x + 1), 1)
    #                     else:
    #                         up = up + self.line_up(float(x), float(x + 1), float(fuzzy_output['high_right']))
    #                         down = down + self.line_down(float(x), float(x + 1), float(fuzzy_output['high_right']))

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

        # Defuzzify the fuzzy output
        defuzzified_value = self.defuzzify(fuzzy_output)
        print(f'defuzzified: {defuzzified_value}')

        return defuzzified_value


def main():
    # Create an instance of the fuzzy controller
    rotate_fuzzy_system = FuzzyController()

    # Define the fuzzy output values
    fuzzy_output = {
        'high_right': 0.4,
        'low_right': 0.6,
        'nothing': 0,
        'low_left': 0,
        'high_left': 0
    }

    # Calculate the centroid using the defuzzify method
    centroid = rotate_fuzzy_system.defuzzify(fuzzy_output)


if __name__ == "__main__":
    main()
