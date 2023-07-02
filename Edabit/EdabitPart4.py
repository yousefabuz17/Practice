from typing import NamedTuple

class ones_threes_nines:
    def __init__(self, num):
        self.num = num

    class Nums(NamedTuple):
        one: int
        three: int
        nine: int
    choices = Nums(1,3,9)
    ones = property(lambda self: self.num // self.choices.one)
    threes = property(lambda self: self.num // self.choices.three)
    nines = property(lambda self: self.num // self.choices.nine)



