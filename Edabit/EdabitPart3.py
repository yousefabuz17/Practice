from math import *
import fractions
import re
import datetime
import numpy as np
import operator
from itertools import *
import random as r
import string as s
import textwrap as tw
import calendar as c


def count_digits(n, d):
    def squared(x):
        return "".join([str(i**2) for i in range(0,x+1)])
    return squared(n).count(str(d))


def even_or_odd(x):
    def checker(y):
        return "even" if y%2==0 else "odd"
    return checker(sum(x))

def invert_list(x):
    return [-i for i in x]

def diving_minigame(x):
    breath_meter = 10
    for i in x:
        if i < 0:
            breath_meter -= 2
            if breath_meter>10:
                breath_meter = 10
            if breath_meter <= 0:
                return False
        if i >= 0:
            breath_meter += 4
            if breath_meter>10:
                breath_meter = 10
            if breath_meter <= 0:
                return False
    return bool(breath_meter)

# multiply = lambda x: lambda n: [n*i for i in x]

def list_index(lst,idx):
    lst = enumerate([j for i in lst for j in i],1)
    return "".join([j for i,j in lst if i in idx])


def exponents(x,y):
    return [i**j for i in x for j in y]

#Write your function here
def over_nine_thousand(x):
    def calculate_sum(y):
        total = 0
        while total<9000:
            for i in y:
                total += i
                if total>=9000:
                    return total
    return sum(x) if sum(x)<9000 else calculate_sum(x)

def clean_up_list(x):
    even,odd = [int(i) for i in x if int(i)%2==0],[[int(i) for i in x if int(i)%2==1]]
    return [even]+odd

def funny_numbers(n,p):
    s = 0
    for i,c in enumerate(str(n)):
        s += pow(int(c),p+i)
    return s/n if s%n==0 else None


print(funny_numbers(1385,3))
print(funny_numbers(92,1))
# print(funny_numbers(46288,5))