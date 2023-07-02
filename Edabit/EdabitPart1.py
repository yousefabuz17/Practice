from math import *
import fractions
import re
import datetime
import numpy as np
import operator
from functools import reduce
import itertools
import random as r
import string as s
import textwrap as tw
import calendar as c

def order(x):
    for i in range(len(x)):
        min_ = x[i]
        idx = i
        counter = i
        for j in range(i, len(x)):
            if x[j] < min_:
                min_ = x[j]
                idx = counter
            counter += 1
        temp = x[i]
        x[i] = min_
        x[idx] = temp
    return x


def free_shipping(order):
    return sum(order[i] for i in order) < 50


def detect_word(x):
    return "".join([i for i in x if i.islower()])


def reverse_list(num):
    return list(map(int, str(num)))[::-1]


def zip_it(x, y):
    import random as r
    return r.choices(list(zip(x, y))) if len(set(map(len, [x, y]))) <= 1 else "sizes don't match"


def all_truth(*args):
    return True if all(bool(i) == True for i in args) else False


def calc_determinant(matrix):
    (a, b), (c, d) = matrix
    return a * d - b * c


def mirror(x):
    return x[0:-1] + x[::-1]


def retrieve_major(x):
    return x.split(".")[0]


def retrieve_minor(x):
    return x.split(".")[1]


def retrieve_patch(x):
    return x.split(".")[2]


z = [[i for i in range(95, 100)] for j in range(30, 34)]


def matrix(x):
    return [j for i in x for j in i]


def pairs(z):
    return [(x, y) for x in z for y in z if x != y]


def factorial(n): return n*factorial(n-1) if n > 1 else 1


def nested(n):
    return [(x, y, z) for x in range(n) for y in range(x+1, n) for z in range(y+1, n) if x*y*z % 2 == 1]


def capital(x):
    return [i.capitalize() for i in x]


def mapy(x):
    return list(map(lambda x: x.capitalize(), x))


def nums(x):
    return list(map(lambda x: x**2 if x % 2 == 0 else x**3, range(1, 11)))


x = [[1, 2], [3], [4, 5, 6]]
y = [[7, 8], [9, 0]]


def newlst(x, y):
    return [a for x in x+y for a in x]


operator = operator.concat([j for i in x for j in i], [
                           j for i in y for j in i])

x = [1, 4, 3, 2, 5]
results = [i**2 if i in [1, 2] else i**3 if i == 3 else 0 for i in x]


def can_capture(rooks):
    return rooks[0][0].startswith(rooks[1][0]) or rooks[0][1].endswith(rooks[1][1])


k = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
m = [[12, 13], [14, 15], [16, 17]]


def combine(x, y):
    return [j for i in x for j in i] + [j for i in y for j in i]


def digital_root(n):
    return True if sum(eval(i)**len(str(n)) for i in str(n)) == n else False


def find_it(seq):
    return [i for i in seq if seq.count(i) % 2 == 1][0]


def exists_higher(x, y):
    return True if [i for i in x if i >= y] else False

def check_factors(x, y):
    return all(y % i == 0 for i in x)

def join_path(x, y):
    return x.replace("/", "") + "/"+y.replace("/", "")

#z = lambda x=[j for i in x for j in i],y=y: x+y


def operator(x, y):
    def multiply():
        print(f"{x}*{y} = {x*y}")

        def basepow():
            print(f"{x}**{y}= {x**y}")

            def factorial(x, y):
                def factorial(num): return 1 if num <= 1 else num * \
                    factorial(num-1)
                print(f"{x}! = {factorial(x)}, y! = {factorial(y)}")
            factorial(x, y)
        basepow()
    multiply()

    def divide():
        print(f"{x}/{y}= {x/y:.2}")

        def intdivide():
            print(f"{x}//{y}= {x//y}")

            def modulo():
                print(f"{x}%{y}= {x%y}")

                def sqrt():
                    print(f"(sqrt({x}),sqrt({y})) = ({x**0.5:.4},{y**0.5:.4})")
                sqrt()
            modulo()
        intdivide()
    divide()


matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])

matrixa = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

squared = list(map(lambda x: x**2, range(-5, 5)))
less_than_zero = list(filter(lambda x: x < 0, range(-5, 5)))
product = reduce((lambda x, y: x*y), range(1, 5))


def prime_numbers(x):
    return sum(1 for i in range(2, x) if all(i % j for j in range(2, i)))


def get_decimal_places(x):
    return len(x[x.find('.')+1:]) if '.' in x else 0


def identical_filter(x):
    return [i for i in x if len(set(i)) == len(i)]


def smaller_num(x, y):
    return min(x, y, key=int)


def change_enough(x, y):
    return True if sum([a*b for a, b in zip(x, [0.25, 0.10, 0.05, 0.01])]) >= y else False


def is_valid(x):
    return True if len(x) == 5 and x.isnumeric() else False


def after_n_years(x, n):
    return {key: value+n for key, value in x.items()}


def not_not_not(x, y):
    return (y == True and x % 2 == 0) or (y == False and x % 2 == 1)


def add_name(obj, name, value):
    obj.setdefault(name, value)
    # or
    obj[name] = value
    # or
    obj.update({name: value})
    return obj


def has_hidden_fee(x, y):
    y = y.strip("$")
    x = [i.strip("$") for i in x]
    return False if sum([int(i) for i in x]) == int(y) else True
# print(has_hidden_fee(["$25", "$6", "$19", "$9", "$32", "$15", "$10", "$9", "$7", "$8", "$37", "$23", "$18"], "$232"))


class Car:
    def __init__(self):
        self.pub = "i am public"
        self._pub = "i am protected"
        self.__pub = "i am private"


now = datetime.datetime.now()
old = datetime.datetime(1998, 3, 17, 3, 0, 0, 0)
print(f"I am {now-old} old")

# print([[j*i for i in range(5)] for j in range(3)])
# for i in range(5):
#     for j in range(3):
#         print(f"i={i}")
#         print(f"j={j}")
#         print(f"i*j={i*j}")
#         print("--------------")


def even_odd_partition(x):
    return [list(filter(lambda i: i % 2 == 0, x)), list(filter(lambda i: i % 2 == 1, x))]


def subset(x, y):
    return set(x).issubset(set(y))
    return all([i in set(y) for i in x])


def largest_swap(x):
    return True if x >= int(str(x)[::-1]) else False


def month_name(x):
    import calendar
    return calendar.month_name[x]


def odd_products(x):
    import numpy
    return numpy.prod([i for i in x if i % 2 != 0])


def my_pi(x):
    import math
    return round(math.pi, x)


def website(x):
    import re
    return re.split("/", x)[-2]


def strip_sentence(txt, chars):
    return txt.translate({ord(i): None for i in chars})


def divisible(x):
    from functools import reduce
    return reduce((lambda x, y: x*y), x) % reduce((lambda x, y: x+y), x) == 0


def index_shuffle(x):
    return "".join([x[i] for i in range(len(x)) if i % 2 == 0]+[x[i] for i in range(len(x)) if i % 2 == 1])


def count_ones(x):
    return sum(1 for i in x for j in i if j == 1)


def is_prefix(word, prefix):
    return word.startswith(prefix[0:len(prefix)-1])


def is_suffix(word, suffix):
    return word.endswith(suffix[1:])


def sum_found_indexes(x, y):
    return sum([i for i, j in enumerate(x) if j == y])


def findLargestNums(x):
    return [max(i) for i in x]

def spelling(x):
    return [x[:i+1] for i in range(len(x))]


def first_vowel(x):
    return [x.index(i) for i in x if i.lower() in 'aeiou'][0]


def filter_state_names(x, y):
    return [i for i in x if i.isupper()] if y == 'abb' else [i for i in x if not i.isupper()]


def inclusive_list(start_num, end_num):
    return list(range(start_num, end_num+1)) or [start_num]


def test_fairness(x, y):
    return sum([i*j for i, j in x]) == sum([j*i for j, i in y])


def clear_fog(x):
    return "".join(i for i in x if i not in "fog") if 'frog' in x else "It's a clear day!"


def society_name(friends):
    return "".join(sorted([i[0] for i in friends]))


def count_vowels(x):
    return sum(1 for i in x if i in 'aeiou')


def relation_to_luke(x):
    dic = {'Darth Vader': 'father', "Leia": "sister",
           "Han": "brother in law", "R2D2": "droid"}
    return f"Luke, I am your {dic[x]}"


def count_ones(x):
    return bin(x).count(str(1))


def filter_list(x):
    return list(filter(lambda i: type(i) == int, x))


def missing_num(x):
    return sum([i for i in range(1, 11) if i not in x])


def name_shuffle(x):
    return " ".join([i for i in x.split(' ')][::-1])


def next_in_line(x, y):
    return x[1:] + [y] if len(x) >= 1 else "No list has been selected"


def is_valid_PIN(x):
    return all([i.isdigit() for i in x]) and len(x) == 4 or len(x) == 6


def index_of_caps(x):
    return [index for index, value in enumerate(x) if value == value.upper() and value.isalpha()]


def letter_counter(x, y):
    return sum(1 for i in x for j in i if j == y)


def index_multiplier(x):
    return sum([index*value for index, value in enumerate(x)])


def alphanumeric_restriction(x):
    return all(i.isdigit() for i in x) or all(i.isalpha() for i in x) if len(x) >= 1 else False


def filter_list(x):
    return list(filter(lambda i: isinstance(i, int), x))


def find_bob(x):
    return [index for index, value in enumerate(x) if value == 'Bob'][0] if "Bob" in x else -1


def letters_only(x):
    return "".join([i for i in x if i.isalpha()])


def replace_vowels(x, y):
    return re.sub('[aeiou]', y, x.lower())


def card_hide(x):
    last = x[len(x)-4:]
    return re.sub(r'\d', '*', x[:len(x)-4]) + last


def mood_today(x='neutral'):
    return "Today, I am feeling {}".format(x)


def num_layers(x):
    return f"Paper folded once is {x}mm (equal to {(0.05*(2**x))/100}m)"


def integer_boolean(x):
    return [bool(eval(i)) for i in x]


def mapping(x):
    return {key: key.title() for key in x}


def cap_to_front(x):
    return "".join(sorted(x, key=str.islower))


def is_vowel_sandwhich(x):
    return x[0] not in 'aeiou' and x[-1] not in 'aeiou' and x[len(x)//2] in 'aeiou'


def get_budgets(x):
    return sum([i['budget'] for i in x])


def is_in_order(x):
    return "".join(sorted(x)) == x


def oddeven(x):
    return sum(1 for i in x if i % 2 == 0) < sum(1 for i in x if i % 2 == 1)


def left_digit(x):
    return eval([i for i in x if i.isnumeric()][0])


def find_odd(x):
    return [x[i] for i in range(len(x)) if x.count(x[i]) % 2 == 1][0]


def correct_stream(x, y):
    return [1 if i == j else -1 for i, j in zip(x, y)]


def amplify(x):
    return [i*10 if i % 4 == 0 else i for i in range(1, x+1)]


def mean0(x):
    return round(sum(eval(i) for i in str(x))/sum(1 for i in str(x)))


def halloween(x):
    return 'Bonfire toffee' if 1 <= int(x.split('/')[0]) <= 9999 and int(x.split('/')[1]) == 10 and int(x.split('/')[2]) == 31 else 'toffee'


def sort_by_length(x):
    return sorted(x, key=len)


def evenly_divisible(x, y, z):
    return sum([i for i in range(x, y+1) if i % z == 0])


def steps_to_convert(x):
    return min(sum([1 for i in x if i.islower()]), sum([1 for i in x if i.isupper()]))


def chatroom_status(x):
    return "no one online" if len(x) < 1 else "{} online".format(" and ".join([i for i in x])) if 1 <= len(x) <= 2 else "{} and {} more online".format(", ".join([i for i in x[:2]]), sum([1 for i in x[2:]]))


def unique(x):
    return [i for i in x if x.count(i) == 1][0]


def binary(x):
    return "{:b}".format(x)


def convert_to_decimal(x):
    return [float(i.strip("%"))/100 for i in x]


def list_operation(x, y, z):
    return list(filter(lambda a: a % z == 0, range(x, y+1)))


def sum_of_evens(x):
    return sum([j for i in x for j in i if j % 2 == 0])


def ascii_capitalize(x):
    return "".join([i.swapcase() if ord(i) % 2 == 0 else i.lower() for i in x])


def add_indexes(x):
    return [i+j for i, j in enumerate(x)]
    return [i+x[i] for i in range(len(x))]


def return_only_integer(x):
    return [i for i in x if isinstance(i, int)]


def profit_margin(x, y):
    return "{:.1%}".format((y-x)/y)


def convert_cartesian(x, y):
    return [list(i) for i in zip(x, y)]
    return list(map(list, zip(x, y)))


def nth_smallest(x, y):
    return sorted(x)[y-1] if len(x) >= y else None


def has_digit(txt):
    return bool(re.search('\d{1}', txt))


def first_and_last(s):
    return ["".join(sorted(s)), "".join(sorted(s, reverse=True))]


def probability(x, y):
    return round(100*len([i for i in x if i >= y])/len(x), 1)


def hamming_distance(x, y):
    return sum(i != j for i, j in zip(x, y))


def showdown(x, y):
    return "tie" if x.index('B') == y.index("B") else "p2" if x.index("B") > y.index("B") else "p1"


def in_box(lst):
    for i in range(1, len(lst)-1):
        for j in range(1, len(lst[i])-1):
            if lst[i][j] == '*':
                return True
            return False


def remove_vowels(x):
    return re.sub('[aeiou]', '', x, flags=re.IGNORECASE)


def remove_enemies(x, y):
    return [i for i in x if i not in y]


def is_automorphic(x):
    return str(x**2).endswith(str(x))


def dict_to_list(x):
    return sorted([(i, j) for i, j in x.items()])


def maximum_score(x):
    return sum(i.get('score') for i in x)


def total_amount_adjectives(x):
    return sum(1 for i in x.values())


def jazzify(lst):
    return [f"{i}7" if not i.endswith("7") else i for i in lst]


def number_split(x):
    return [floor(x/2), ceil(x/2)]

def to_list0(x):
    return sum(value * 10**index for index, value in enumerate(x[::-1]))
    return eval("".join(str(i) for i in x))

def reverse0(x):
    return x[::-1].swapcase()

def marathon_distance(x):
    return
    return sum(abs(i) for i in x) >= 25

def equal(x, y, z):
    return {3: 0, 2: 2, 1: 3}[len({x, y, z})]

def collatz(num):
    return 1 + collatz([num // 2, num * 3 + 1][num % 2]) if num > 1 else 0

def vow_replace(x, y):
    return re.sub('[aeiou]', y, x)

def is_triplet(*x):
    min_, mid, max_ = sorted(x)
    return sum([min_**2, mid**2]) == max_**2

def multiply_nums(x):
    return eval(x.replace(", ", "*"))
    return np.prod([1*eval(i) for i in x.split(', ')])


def space_weights(planet_a, weight, planet_b):
    combined = dict(zip(["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"], [
                    3.7, 8.87, 9.81, 3.711, 24.79, 10.44, 8.69, 11.15]))
    return round((weight/combined[planet_a])*combined[planet_b], 2)

def apocalyptic(x):
    return "Repent, {} days until the Apocalypse!".format(str(2**x).index('666')) if '666' in str(2**x) else "Crisis averted. Resume sinning."


def abcmath(x, y, z):
    return x*2**y % z == 0


def square_digits(x):
    return int("".join(str(int(i)**2) for i in str(x)))


def factor_chain(x):
    return all(x[i] % x[i-1] == 0 for i in range(1, len(x)))


def war_of_numbers(x):
    return max([sum(i for i in x if i % 2 == 0), sum(i for i in x if i % 2 == 1)])-min([sum(i for i in x if i % 2 == 0), sum(i for i in x if i % 2 == 1)])


def total_volume(*x):
    return sum(np.prod(i) for i in x)


def move_to_end(x, y):
    return [i for i in x if i != y] + [i for i in x if i == y]


def print_all_groups():
    return ", ".join([", ".join(i) for i in [[str(i)+j for j in ['a', 'b', 'c', 'd', 'e']] for i in range(1, 7)]])


def weight(r, h):
    return round((pi*(r**2)*h)/1000, 2)


def minutes_to_seconds(x):
    m, s = x.split(":")
    if int(s) >= 60:
        return False
    return int(m)*60 + int(s)


def prevent_distractions(x):
    return "NO!" if any(i in x for i in ("anime", "meme", "vines", "roasts", "Danny DeVito")) else "Safe watching!"


def is_harshad(x):
    return x != 0 and x % sum(int(i) for i in str(x)) == 0


def get_discounts(x, y):
    return [int(y.strip("%"))*i/100 for i in x]

def tuck_in(x,y):
    return [x[0]]+[i for i in y]+[x[-1]]

def count0(x):
    return sum({key:value for key,value in zip([10,'J',"Q","K","A",7,8,9,2,3,4,5,6],[-1,-1,-1,-1,-1,0,0,0,1,1,1,1,1])}.get(i,0) for i in x)

def vowel_split(x):
    def is_vowel(i): 
        return i.lower() in "aeiou"
    return "".join([i for i in x if is_vowel(i)]+[i for i in x if not is_vowel(i)])

def remove_vowels(x):
    return "".join(i for i in x if i not in 'aeiou')

def convert_binary(x):
    return "".join("0" if ord(i) in range(ord('a'),ord('m')+1) else "1" if ord(i) in range(ord("n"),ord("z")) else None for i in x.lower())

def factor_group(x):
    return "even" if len([j for i in range(x,x+1) for j in range(1,x+1) if i%j==0])%2==0 else "odd"
    return 'odd' if (x**0.5)%1==0 else 'even'

def inator_inator(x):
    return "{}{}inator {}".format(x, '-' if x[-1].lower() in 'aeiou' else '',len(x)*1000)

def special_reverse(x,y):
    return " ".join(i if not i.startswith(y) else i[::-1] for i in x.split())

def worm_length(x):
    return "{} mm.".format(len(list(filter(lambda i: i=='-',x)))*10) if all(i=='-' for i in x) and x else "invalid"
    # worm_length=lambda w:["invalid","%d0 mm."%len(w)][set(w)=={"-"}]

def valid_str_number(x):
    try:
        float(x)
    except:
        return False
    return True

def shhh(x):
    return f"\"{x.capitalize()}\", whispered Edabit."

def matrix(x,y,z):
    return [[z for i in range(y)] for j in range(x)]

def odds_vs_evens(x):
    equal = sum(int(i) for i in str(x) if int(i)%2==0)==sum(int(i)for i in str(x) if int(i)%2==1)
    even = sum(int(i) for i in str(x) if int(i)%2==0) > sum(int(i)for i in str(x) if int(i)%2==1)
    odd = sum(int(i)for i in str(x) if int(i)%2==1)>sum(int(i) for i in str(x) if int(i)%2==0)
    return "equal" if equal else "even" if even else "odd"

def emotify(x):
    # return re.sub('smile',':D',x) if 'smile' in x else re.sub('grin',':)',x) if 'grin' in x else re.sub('mad',':P',x) if 'mad' in x else re.sub('sad',':(',x)
    emojis = {"smile":":D","grin":":)","sad":":(","mad":":P"}
    for i,j in emojis.items():
        x = x.replace(i,j)
    return x

def parity_analysis(x):
    return sum(int(i) for i in str(x))%2==x%2

def remove_dups(x):
    return sorted(set(x),key=x.index)

def double_factorial(x):
    return 1 if x<=0 else x*double_factorial(x-2)

def color_invert(x):
    return tuple([255-i for i in x])

def circle_or_square(x,y):
    return (2*3.14*x),4*sqrt(y)

def get_type(x):
    return type(x).__name__

def count_palindromes(x,y):
    return sum(str(i)==str(i)[::-1] for i in range(x,y+1))

def longest_zero(x):
    return sorted(x.split('1'),key=len,reverse=True)[0]

def flatten_the_curve(x):
    return np.array([np.mean(x) for i in x]).astype(float)

def find_nemo(x):
    return "I found Nemo at {}".format(str(x.split().index("Nemo")+1)) if "Nemo" in x.split() else "I can't find Nemo :("

def retrieve(x):
    return [i for i in x[:-1].lower().split() if i[0] in 'aeiou']

def count_all(x):
    return dict(zip(["LETTERS","DIGITS"],[sum(1 for i in x if i.isalpha()),sum(1 for i in x if i.isdigit())]))

def is_apocalyptic(x):
    return ['Safe','Single','Double','Triple'][str(2**x).count('666')]

def get_xp(x):
    levels = {
        "Very Easy":5,
        "Easy":10,
        "Medium":20,
        "Hard":40,
        "Very Hard":80
    }
    for i,j in x.items():
        x[i] *= levels[i]
    return "{}XP".format(sum(x.values()))

def error(x):
    errors = {
        1:"Check the fan: e1",
        2:"Emergency stop: e2",
        3:"Pump Error: e3",
        4:"c: e4",
        5:"Temperature Sensor Error: e5"
    }
    return errors[x] if 1<=x<=5 else 101

def get_only_evens(x):
    return [j for i,j in enumerate(x) if i%2==0 and j%2==0]

def double_letters(x):
    return any(i*2 for i in x)
    return any(i==j for i,j in zip(x, x[1:]))

def adjacent_product(x):
    return max([[i*j] for i,j in zip(x,x[1:])])[0]

def sort_descending(num):
    return int("".join(sorted(str(num),reverse=True)))

def alph_num(x):
    return " ".join(str(ord(i)-ord("A")) for i in x)

def count_towers(x):
    return str(x[-1]).count("##")
    return "".join(x[-1]).count("##")

def parse_list(x):
    return list(map(str,x))

def sum_two_smallest_nums(x):
    return sum(sorted(filter(lambda i: i>0,x))[:2])

def get_frequencies(x):
    return {value: x.count(value) for value in x}

def asc_des_none(x,y):
    return sorted(x,reverse=y=="Des") if y else x
    return sorted(x,reverse=False) if y=="Asc" else sorted(x,reverse=True) if y=="Des" else x

def hacker_speak(x):
    return x.translate("".maketrans('aeios','43105'))
    return x.replace("a","4").replace("o","0").replace("e","3").replace("i","1").replace("s","5")

def temp_conversion(x):
    return "invalid" if round(x+273.15,2)<=0 else [round(x*(9/5)+32,2),round(x+273.15,2)]

def maskify(x):
    return x if len(x)<=4 else "".join(i.replace(i,"#") for i in x[:-4])+x[-4:]

def sum_of_two(x,y,z):
    return any(i+j==z for i in x for j in y)

def is_palindrome(x):
    return "".join(filter(lambda i: i.isalpha(),x)).lower()=="".join(filter(lambda i: i.isalpha(),x)).lower()[::-1]

def get_distance(x,y):
    return round(sqrt(((y['y']-x['y'])**2)+((y['x']-x['x'])**2)),3)

def sum_neg(x):
    return x and [sum(1 for i in x if i>=0),sum(i for i in x if i<0)]

class Solution:
    def twoSum(self,x,y):
        for i in range(len(x)-1):
            if x[i]+x[i+1]!=y:
                continue
        return [x.index(x[i]),x.index(x[i])+1]
def two_sum(x,y):
    for i in range(len(x)):
        for j in range(len(x)):
            if x[:i+1]+x[:j+1]!=y:
                continue
    return x.index(x[i]),x.index(x[j])

def addTwoNumbers(x,y):
    return [sum([i,j]) for i in x for j in y]

def high_low(x):
    return '{} {}'.format(max(sorted(x.split(),key=int)),min(sorted(x.split(),key=int)))

def remove_empty_arrays(x):
    return list(filter(lambda i: i!= [],x))

def ascii_sort(x):
    return sorted(x,key=lambda i: sum(map(ord,i)))[0]

def upward_trend(x):
    try:
        return sorted(x)==x
    except:
        return "Strings not permitted!"

def letter_check(x):
    return all(i in x[0].lower() for i in x[1].lower())

def median(x):
    x.sort()
    return (x[:len(x)//2][-1]+x[len(x)//2:][0])/2 if len(x)%2==0 else x[len(x)//2]

def reverse0(x):
    return " ".join([i[::-1] if len(i)>=5 else i for i in x.split()])

def one_list(x):
    return [j for i in x for j in i]

def one_odd_one_even(x):
    return x//10%2 != x%10%2

def grab_city(x):
    return x.replace("]","").split('[')[-1]

def scale_tip(x):
    left = x[:x.index('I')]
    right = x[x.index("I")+1:]
    return "balanced" if sum(left)==sum(right) else "left" if sum(left)>sum(right) else "right"

def format_math(x):
    return "{} = {}".format(x,int(eval(x.replace("x","*"))))

def eda_bit(x,y):
    return ["EdaBit" if (i%3==0 and i%5==0) else "Eda" if i%3==0 else "Bit" if i%5==0 else i for i in range(x,y+1)]

def catch_zero_division(x):
    try:
        eval(x)
    except ZeroDivisionError:
        return True
    return False

def get_middle(x):
    return x[(len(x)-1)//2:(len(x)+2)//2]

def is_prime(x):
    return x>1 and all(x%i for i in range(2,x))

def is_orthogonal(x,y):
    #Dot Product
    return sum(i*j for i,j in zip(x,y))

def spin_around(x):
    return abs(int(sum(-90 if i=="left" else 90 for i in x)/360))

def format_phone_number(x):
    x = "".join(map(str, x))
    return "({}) {}-{}".format(x[0:3], x[3:6], x[6:])

def how_many_missing(x):
    return len([i for i in range(min(x),max(x)+1) if i not in x]) if len(x)>=1 else 0

def cumaltive_sum(x):
    return [sum(x[:i+1]) for i in range(len(x))]

def flip_end_chars(x):
    return "Incompatible." if len(x)<2 or not isinstance(x,str) else "Two's a pair." if x[0]==x[-1] else "".join(x[-1]+x[1:-1]+x[0])

def superheroes(x):
    return sorted([i for i in x if i.lower().endswith('man') and not i.lower().endswith('woman')])

def data_type(x):
    types = {
        "list":"list",
        "dict":"dictionary",
        "float":"float",
        "int":"integer",
        "str":"string",
        "bool":"boolean",
        "datetime.date":"date"}
    return types[str(type(x)).split("'")[1]]

def numbers_sum(x):
    return sum(i for i in x if isinstance(i,int) and not isinstance(i,bool))

def complete_binary(x):
    return x.zfill(len(x)+8-len(x)%8)

def char_index(x,y):
    return [x.index(y),x.rfind(y)]

def remove_abc(x):
    return None if all(i not in 'abc' for i in x) else re.sub('[abc]','',x)

def verbify(x):
    x_split = [i for i in x.split()]
    return x_split[0]+'ed '+x_split[-1] if x_split[0][-2:]!='ed' else x

def partially_hide(x):
    return ' '.join([(i[0]+'-'*(len(i)-2)+i[-1]) for i in x.split()])

def cms_selector(x,y):
    return sorted(x) if len(y)<1 else sorted([i for i in x if y in i.lower()])

def amazing_edabit(x):
    return x if 'edabit' in x else x.replace('amazing','not amazing')

def partition(x,y):
    return [x[i:i+y] for i in range(0,len(x),y)]
    return tw.wrap(x,y)

def DECIMATOR(x):
    return x[:-ceil(len(x)/10)]

def number_len_sort(x):
    return sorted(x,key=lambda i: len(str(i)))

def days(x,y):
    return c.monthrange(y,x)

def construct_fence(x):
    return "H"*(1000000//int(re.sub(r'[$,]','',x)))
    return "H"*(1000000//int(x.replace('$','').replace(',','')))

def unique_lst(x):
    return sorted(set(i for i in x if i>0),key=x.index)

def is_narcissistic(x):
    return x==sum(int(i)**len(str(x)) for i in str(x))

def stmid(x):
    return "".join(i[0] if len(i)%2==0 else i[len(i)//2] for i in x.split())

def is_special_array(x):
    return all(x[i]%2==i%2 for i in range(len(x)))

def puzzle_pieces(x,y):
    return len(set(map(lambda i,j: i+j,x,y)))==1 if len(x)==len(y) else False
    return len(set(sum(i) for i in zip(x,y)))==1 if len(x)==len(y) else False

def is_good_match(x):
    return [sum(i) for i in zip(x[::2],x[1::2])] if not len(x)%2 else "bad match"

def magic(x):
    m,d,y = x.split()
    return y.endswith(str(int(m)*int(d)))
    return str(np.prod([int(i) for i in x.split()][:2])).endswith(x[-1])

def remove_special_characters(x):
    return re.sub(r'[^\w\s-]','',x)

def double_pay(x):
    return 2**x-1

def fib(x):
    return 0 if x==0 else 1 if 1<=x<=2 else fib(x-1)+fib(x-2)

def index_filter(x,y):
    return "".join(y[i] for i in x)
    return "".join(y[x[j]].lower() for i in range(len(y.lower())) for j in range(len(x)))[:len(x)]

def flip(x, y):
    return ' '.join([i[::-1] if y == "word" else i for i in x.split()]) if y == 'word' else ' '.join([i for i in x.split()][::-1])

def change_types(x):
    return [i.capitalize()+'!' if isinstance(i,str) else not i if isinstance(i,bool) else i//2*2 + 1 for i in x]
    lst = []
    for i in x:
        if isinstance(i,bool):
            lst.append(not i)
        elif isinstance(i,int):
            if i%2==0:
                lst.append(i+1)
        elif isinstance(i,str):
            lst.append(i.capitalize()+'!')
    return lst

def ranged_reversal(x,y,z):
    return x[:y] + x[y:z+1][::-1] + x[z+1:]

def is_prime(x):
    return False if 0<=x<2 else len([i for i in range(1,x+1) if x%i==0])==2

def accum(x):
    return "-".join((j*i).capitalize() for i,j in enumerate(x,1))

def transform_upvotes(x):
    return list(map(lambda i: int(eval(i.replace('k',''))*1000) if 'k' in i else eval(i),x.split()))

def letter_at_position(x):
    return "invalid" if x<1 or x!=int(x) else {key:value for key, value in zip(range(1,27),s.ascii_lowercase)}[x]
    return chr(int(x) + 96) if x in range(1,27) else "invalid"

def format_num(x):
    return "{:,}".format(x)

def tetra(x):
    return int((x*(x+1)*(x+2))/6)

def make_title(x):
    return s.capwords(x)
    return " ".join(i[0].capitalize()+i[1:] for i in x.split())

def replace_vowel(word):
    return word.translate("".maketrans('aeiou','12345'))

def alliteration_correct(x):
    return len(set(i[0] for i in x.lower().split() if len(i)>=4))==1

def de_nest(x):
    return eval(str(x).strip('[]'))
    de_nest=lambda x:eval(re.sub('[\[\]]+','',str(x)))

def is_smooth(x):
    return all(i[-1]==j[0] for i,j in zip(x.lower().split(),x.lower().split()[1:]))
    
def find_zip(x):
    return x.rindex('zip') if x.count('zip')>=2 else -1

def count_adverbs(x):
    return len(re.findall(r'ly[^\w]',x))
    return len([i for i in re.sub('[!@,#$%^&*.]','',x).split() if i.endswith('ly')])

def oldest(x):
    return sorted(x.items(),key=lambda i: i[1])[-1][0]

def discount(x,y):
    return round(x - (x*(y*0.01)),2)

def progress_bar(x,y):
    #return "{:10}" adds spaces
    return "|{:10}| {}".format(x*(y//10),"Completed!" if y==100 else "Progress: {}%".format(y))

def century(x):
    return "{}{} century".format(int(str(x)[:2])+1 if not str(x).endswith('000') else int(str(x)[:2]),'st' if int(str(x)) in range(2001,2099) else 'th')

def find_occurrences(x,y):
    return {key: key.count(y.lower()) for key in x.lower().split()}
    return {key:key.count(value) for key,value in zip(x.lower().split(),y)}

def jay_and_bob(x):
    size = {"half":14,"quater":7,"eighth":3.5,"sixteenth":1.75}
    return "{} grams".format(size[x])

def unique_in_order(x):
    return [i[0] for i in itertools.groupby(x)]

def quadratic_equation(x,y,z):
    return (-y+sqrt(y**2-(4*x*z)))/(2*x)

def is_disarium(x):
    return sum(j for i in [[int(j)**i] for i,j in enumerate(str(x),1)] for j in i)

def format_date(x):
    return "".join(x.split('/')[::-1])

def construct_deconstruct(x):
    return [x[:i+1] for i in range(len(x))]+[x[:i+1] for i in range(len(x)-2,-1,-1)]
    print("\n".join(x[:i+1] for i in range(len(x))))
    print("\n".join(x[:i+1] for i in range(len(x)-2,-1,-1)))

def divisible_by_b(x,y):
    return [i for i in range(x,x+50) if i%y==0][0]

def fix_import(x):
    return "from {} import {}".format(x.split()[-1],x.split()[1])

def list_of_multiples(x,y):
    return [j for i in range(1) for j in range(x,x*y+1)if j%x==0]

def solutions(a,b,c):
    try:
        return len(set([((-b+sqrt(b**2-(4*a*c)))/(2*a)),((-b-sqrt(b**2-(4*a*c)))/(2*a))]))
    except:
        return 0

def wash_hands(x,y):
    return "{} minutes and {} seconds".format(floor((x*21)*30*y/60),divmod((x*21)*30*y,60)[1])

def vol_shell(x,y):
    return round(abs((4/3)*pi*(x**3-y**3)),3)

def make_rug(x,y,z='#'):
    return ["".join([z for i in range(y)]) for j in range(x)]

def get_equivalent(x):
    notes = [("C#","Db"),("D#","Eb"),("F#","Gb"),("G#","Ab"),("A#","Bb")]
    return "".join([j for i in [i for i in notes if x in i] for j in i if j!=x])

def invert(x):
    return {value:key for key,value in x.items()}

def multiply0(x):
    return [[j for i in x] for j in x]

def to_dict(x):
    return x if len(x)==0 else [{i:ord(i)} for i in x]

def word_builder(x,y):
    return ''.join([x[i] for i in y])
    return "".join([j for i in sorted([[i,j] for i,j in zip(x,y)],key=lambda i: i[1]) for j in i[0]])

def get_indices(x,y):
    return [i for i in range(len(x)) if x[i]==y]
    return [i[0] for i in [[i,j] for i,j in enumerate(x) if j==y]]

def normalize(x):
    return x.capitalize()+'!' if all(i.isupper() for i in x.split()) else x

def peel_layer_off(x):
    return [i[1:len(i)-1] for i in x[1:len(x)-1]]

def greeting(x):
    return "Hi! I'm {}{}.".format("a guest" if x not in GUEST_LIST else x+','," and I'm from {}".format(GUEST_LIST[x]) if x in GUEST_LIST else '')

def is_equal(x):
    return sum(int(j) for i in str(x[0]) for j in str(i))==sum(int(j) for i in str(x[1]) for j in str(i))

def expensive_orders(x,y):
    return {key:value for key,value in x.items() if value>=y}

def num_of_sublists(x):
    return str(x).count('[')-1

def histogram(x,y):
    return "\n".join(map(lambda i: i*y,x))

def oddish_or_evenish(x):
    return "Oddish" if sum(int(i) for i in str(x))%2>0 else "Evenish"

def concatt(*x):
    return [j for i in x for j in i]

def sum_odd_and_even(x):
    return [sum(i for i in x if i%2==0),sum(i for i in x if i%2==1)]

def count_uppercase(x):
    return sum(1 for i in x for j in i if j.isupper())

def move(x):
    letters = {key:value for key, value in list(zip(s.ascii_lowercase,s.ascii_lowercase[1:]))}
    return "".join(i.replace(i,letters[i]) for i in x)

def square_patch(x):
    return [[x for i in range(x)] for j in range(x)]

def extend_vowels(x,y):
    return "invalid" if y<0 or not isinstance(y,int) else "".join(i*(y+1) if i.lower() in 'aeiou' else i for i in x)

def gimme_the_letters(x):
    x = x.split('-')
    return "".join([chr(i) for i in range(ord(x[0]),ord(x[1]))])

def repdigit(x):
    return len(set(str(x)))==1

def valid_division(x):
    try:
        x = x.split('/')
        return divmod(eval(x[0]),eval(x[-1]))[1]==0
    except ZeroDivisionError:
        return "invalid"

def sum_fractions(x):
    return int(sum([i/j for i,j in x]))

def percentage_changed(x,y):
    x = int(x.strip('$'))
    y = int(y.strip('$'))
    percent = ((y-x)/x)*100
    return "{}% {}".format(abs(int(percent)),"decrease" if x>y else "increase")

def fizz_buzz(x):
    return ["FizzBuzz" if i%3==0 and i%5==0 else "Fizz" if i%3==0 else "Buzz" if i%5==0 else i for i in range(1,x+1)]

def censor(x):
    return " ".join([i.replace(i,"*"*len(i)) if len(i)>4 else i for i in x.split()])

def even_odd_transform(x,y):
    return [i+(y*2) if i%2 else i-(y*2) for i in x]

def digit_occurrences(x,y,z):
    return [j for i in range(x,y+1) for j in str(i)].count(str(z))

def letters_only(x):
    return False if len(x)<1 else all(i.islower() and i.isalpha() for i in x.split())

def longest_word(x):
    return max(x.split(),key=len)

def is_pandigital(x):
    return len(set(str(x)))==10

def check_perfect(x):
    return sum([i for i in range(1,x) if x%i==0])==x

def sum_of_vowels(x):
    vows = {"A":4,"E":3,"I":1,"O":0,"U":0}
    return sum([vows[i.upper()] for i in x if i.upper() in vows])

def get_student_top_notes(x):
    return [max(i['notes']) if i['notes'] else 0 for i in x]

def mystery_func(num):
    return np.prod([int(i) for i in str(num)])

def get_total_price(x):
    return round(sum([i['quantity']*i['price'] for i in x]),2)

def lst_ele_sum(x):
    return [sum(x)-i for i in x]

def char_appears(x,y):
    return [i.lower().count(y.lower()) for i in x.split()]

def find_it(x,y):
    return "{} is gone...".format(y.title()) if y in x else "{} is here!".format(y.title())

def to_scottish_screaming(x):
    return x.upper().translate("".maketrans('AIOU','EEEEE'))

def duplicates(x):
    values = {key:value-1 for key,value in zip(x,[x.count(i) for i in x]) if value>=2}
    return sum(values[i] for i in values)
    return len(x)-len(set(x))

def all_prime(x):
    return all(list(map(lambda i:len([j for j in range(1,i+1) if i%j==0])==2,x)))

def get_days(x,y):
    return (y-x).days
    return int(str((y-x)).split()[0])

def half_a_fraction(x):
    return fractions.Fraction(int(x.split('/')[0]),int(x.split('/')[1]))/2

def first_place(x):
    return None if all(not i.isalpha() for i in x) else [i for i in x if i.isalpha()][-1]

def countdown(x,y):
    return "{} {}!".format(" ".join([str(i)+'.' for i in range(x,0,-1)]),y.upper())

def cap_space(x):
    return "".join(i if not i.isupper() else ' '+i.lower() for i in x)

def is_alpha(x):
    return sum([ord(x[i].lower())-96 for i in range(len(x)) if x[i].isalpha()])%2==0

def same_upsidedown(x):
    return x.translate("".maketrans("69","96"))[::-1]==x

def sum_every_nth(x,y):
    return sum(x[y-1::y])

def happiness_number(x):
    return -sum([len(re.findall(r':\(',x)),len(re.findall(r'\):',x))])+sum([len(re.findall(r':\)',x)),len(re.findall(r'\(:',x))])

def sum_digits(x,y):
    return sum([int(j) for i in range(x,y+1) for j in str(i)])

def is_slidey(x):
    return all(abs(int(i)-int(j))==1 for i,j in zip(str(x),str(x)[1:]))

def sum_missing_numbers(x):
    return sum([i for i in range(min(x),max(x)+1) if i not in x])

def accumulating_product(x):
    return list(itertools.accumulate(x,lambda i,j: i*j))
    return list(reduce(lambda i,j:i*j,x))

def convert_to_number(x):
    return {key:int(value) for key,value in x.items()}

def single_occurrence(x):
    return "".join([i.upper() for i in x.lower() if x.lower().count(i)==1])

def time_saved(sp_limit,avg_sp,d):
    return round((d/sp_limit)*60-(d/avg_sp)*60,1)

def is_prime(x):
    return 'true' if len([i for i in range(1,x+1) if x%i==0])==2 else 'false'

def last_dig(x,y,z):
    return str(x*y)[-1]==str(z)[-1]

def binary_to_decimal(x):
    return int("".join(str(i) for i in x),2)

def right_triangle(x,y,z):
    sides = sorted([x,y,z])
    return False if any(i<=0 for i in sides) else sqrt(sides[0]**2 + sides[1]**2)==sqrt(sides[2]**2) 

def logarithm(x,y):
    return int(log(y,x))

def find(x):
    return True if re.search(r'\.py$|\.pyw$',x) else False

def neutralize(x,y):
    return "".join([str(0) if i!=j else str(i) for i,j in zip(x,y)])

def value_at(x,y):
    return x[int(y)]

def total_overs(x):
    return divmod(x,6)[0]+divmod(x,6)[1]/10

def i_sqrt(x,i=0):
    return "invalid" if x<0 else i if x<2 else i_sqrt(x**0.5,i+1)

def is_sastry(x):
    return int(str(x)+str(x+1))**0.5%1==0
    return sqrt(int(str(x)+str(x+1)))-floor(sqrt(int(str(x)+str(x+1))))==0

def sum13(x):
    return x[x.index(13):x.index(13)+2]

def fifth(*x):
    return "Not enough arguments" if len(x)<4 else [type(i) for i in x][-1]

def check_square_and_cube(x):
    return sqrt(x[0])==x[1]/x[0]

def operation(x,y,z):
    try:
        operators = {"add":'+',
                    "subtract":'-',
                    "divide":'/',
                    "multiply":'*'}
        return str(eval(x+operators[z]+y))
    except:
        return "undefined"

def count_overlapping(x,y):
    return len([[i,j] for i,j in x if i<=y<=j])

def free_throws(x,y):
    return "{}%".format(round((int(x.strip("%"))/100)**y*100))

def to_list(x):
    return [] if len(x)<1 else [[i,j] for i,j in x.items()]

def is_valid_date(x,y,z):
    day,month,year = x,y,z
    try:
        return bool(datetime.date(year,month,day))
    except:
        return False

class Name:
    def __init__(self,first,last):
        self.first = first.title()
        self.last = last.title()
        self.fname = self.first
        self.lname = self.last
        self.fullname = "{} {}".format(self.first,self.last)
        self.initials = "{}.{}".format(self.first[0],self.last[0])

def progress_days(x):
    return len([True for i,j in zip(x,x[1:]) if i<j])

def has_syncopation(x):
    return '#' in x[1::2]
    return any([[i,j] for i,j in enumerate(x,1) if i%2==0 and '#' in j])

def harmonic(x):
    return round(sum(1/i for i in range(1,x+1)),3)

def longest_time(h, m, s):
    return [h, m, s][[h*3600, m*60, s].index(max([h*3600, m*60, s]))]

def solve(y):
    return int(y.split()[-1])-int(y.split()[2]) if '+' in y else int(y.split()[-1])+int(y.split()[2])
    return -eval(y[1:].replace("=", "-"))

def vowels(x):
    return sum(1 for i in x.lower() if i in 'aeiou')

def consonants(x):
    return sum(1 for i in x.lower() if i not in 'aeiou' and i.isalpha())

def shared_letters(x,y):
    return len(set(x)&set(y))
    return sum(1 for i in set(x) if i in set(y))

def dice_game(x):
    return 0 if any(i==j for i,j in x) else sum(i+j for i,j in x)

class Book:
    def __init__(self,title,author):
        self.title = title
        self.author = author
    def get_title(self):
        return "Title: {}".format(self.title)
    def get_author(self):
        return "Author: {}".format(self.author)

def dna_to_rna(x):
    return x.translate("".maketrans('ATGC','UACG'))

def calc_bundled_temp(x,y):
    return "{:.1f}*C".format(int(y.split('*')[0])*1.1**x,1)

def circular_shift(x,y,z):
    return x[abs(z):]+x[:z]==y

def is_central(x):
    letter =  int("".join([str(x.index(i)) for i in x if i!=' ']))
    return sum(1 for i in range(0,letter))==sum(1 for i in range(letter+1,len(x)))
    return len(x.lstrip())==len(x.rstrip())

def palindromic_date(x):
    ddmmyy = "".join(x.split('/'))
    mmddyy = ddmmyy[2:4]+ddmmyy[:2]+ddmmyy[4:]
    return ddmmyy==mmddyy

def seven_boom(x):
    return "Boom!" if '7' in [i for i in str(x)] else "there is no 7 in the list"

def duplicate_nums(x):
    return None if len(x)==len(set(x)) else sorted(i for i in set(x) if x.count(i)>1)

def match_last_item(x):
    return "".join(str(i) for i in x[:-1])==x[-1]
    return "".join(map(str,x[:-1]))==x[-1]

def digit_distance(x,y):
    return sum(abs(int(i)-int(j)) for i,j in zip(str(x),str(y)))
    return sum(map(int,str(abs(x-y))))

def cons(x):
    return sorted(x)==list(range(min(x),max(x)+1))

def cap_last(x):
    return " ".join([i[:-1]+i[-1].title() for i in x.split()])

def most_expensive_item(x):
    return sorted([[i,j] for i,j in x.items()],key= lambda i: i[1])[-1][0]


def filter_by_rating(x,y):
    return "No results found" if y not in x.values() else {key:value for key, value in x.items() if value==y}

def unrepeated(x):
    return "".join(sorted(set(x),key=x.index))

def unstretch(x):
    return x[0] + ''.join(x[i] for i in range(1,len(x)) if x[i] != x[i-1])

def alphabet_index(x):
    return " ".join(list(map(str,[ord(i)-96 for i in x.lower() if i in s.ascii_lowercase])))
    return " ".join([str(ord(i)-96) for i in x.lower() if i.isalpha()])

def power_of_two(x):
    return x&(x-1)==0

def element_from_set(x):
    return x.pop()

def sum2(x,y):
    carry = 0
    result = ""
    for i,j in itertools.zip_longest(x[::-1], y[::-1], fillvalue=0):
        z = int(i) + int(j) + carry
        if z >= 10:
            carry = 1
        else:
            carry = 0
        result += str(z%10)
    return result[::-1] if carry==0 else "1" + result[::-1]

def largest_gap(x):
    x.sort()
    return sorted([j-i for i,j in zip(x,x[1:])])[-1]

def reversible_inclusive_list(x,y):
    return list(range(x,y+1)) if y>x else list(range(x,y-1,-1))

def double_swap(x,y,z):
    return x.translate("".maketrans(y+z,z+y))
    return "".join(i.replace(i,z) if i==y else i.replace(i,y) if i==z else i for i in x)

def median(x):
    x.sort()
    return sum(x[len(x)//2-1:len(x)//2+1])/2 if len(x)%2==0 else x[len(x)//2]

def reverse_binary_integer(x):
    return int(bin(x).split('0b')[1][::-1],2)

def abbreviate(x,y=4):
    return "".join(i[0].title() for i in x.split() if len(i)>=y)

def split_code(x):
    return ["".join([i for i in x if i.isalpha()]),int("".join(i for i in x if i.isdigit()))]

def x_length_words(x,y):
    return all(len(i)>=y for i in x.split())

def valid(x):
    while len(x)==4 or len(x)==6:
        return all(i.isnumeric()for i in x)
    return False

def secret(x):
    y = x.split('*')[0]
    n = int(x.split('*')[-1])
    return '<{}></{}>'.format(y,y)*n

def lonely_integer(x):
    return int("".join(set([str(i) for i in x if not -i in x or not i in x])))

def radians_to_degrees(x):
    return round(x*(180/pi),1)

def fifty_thirty_twenty(x):
    return {'Needs':round(x*0.5),
            'Wants':round(x*0.3),
            'Savings':round(x*0.2)}

def ohms_law(v,r,i):
    return 'Invalid' if len([i for i in [v,r,i] if i==None])>1 or None not in [v,r,i] else round(r*i,2) if v==None else round(v/r,2) if i==None else round(v/i,2)

def age_difference(x):
    x.sort()
    return 'No age difference between spouses.' if x[-1]-x[-2]>18 or x[-1]-x[-2]==0 else '1 year' if x[-1]-x[-2]==1 else '{} years'.format(x[-1]-x[-2])

def get_number_of_apples(x,y):
    return 'The children didn\'t get any apples' if x==0 or floor(x-(x*int(y.strip('%'))/100))<1 else floor(x-(x*int(y.strip('%'))/100))

