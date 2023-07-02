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

def keyboard_mistakes(x):
    return x.translate("".maketrans('4501','ASOI'))

def weight_allowed(x,y,z):
    return (x+sum(y))<z/0.453592

def points(x,y):
    return (x*2)+(y*3)

def top_note(x):
    return {'name':x['name'],'top_note':max(x['notes'])}

class Composer:
    count = 0
    def __init__(self,name,dob,country):
        self.name = name
        self.dob = dob
        self.country = country
        Composer.count+=1

def array_diff(x,y):
    return [i for i in x if i!=y]

def scramble(x,y):
    return [i for i in set(x) if i in y],len(y)

def boom_intensity(x):
    div_2 = 'B'+'o'*x+'m!'
    div_5 = 'B'+'O'*x+'M'
    div_both = 'B'+'O'*x+'M!'
    return 'boom' if x<2 else div_both if x%2==0 and x%5==0 else div_2 if x%2==0 else div_5

def pattern(x):
    for i in range(len(x)+1):
        for j in range(i):
            print(x[j],end='')
        print()
    for i in range(len(x)-1,0,-1):
        for j in range(i):
            print(x[j],end='')
        print()

def eval_factorial(x):
    return sum(factorial(int(i.strip('!'))) for i in x)

def add(x):
    def add2(y):
        return x+y
    return add2
    #print(add(10)(20)) = 30

def math_expr(x):
    try:
        return bool(eval(x))
    except:
        return False


def reverse_odd(x):
    return " ".join([i[::-1] if len(i)%2 else i for i in x.split()])

def balanced(x):
    left = x[:len(x)//2]
    right = x[len(x)//2:]
    return left*2 if sum(left)>sum(right) else right*2 if sum(left)<sum(right) else x

def face_interval(x):
    try:
        x.sort()
        return ':)' if x[-1]-x[0] in x else ':('
    except:
        return ':/'

def ends_add_to_10(x):
    return sum(1 for i in x if int(str(abs(i))[0])+int(str(abs(i))[-1])==10)

def check(x, y, k):
    return True if x[k] and y[k] else 'One\s empty'

def first_n_vowels(x,y):
    return "invalid" if len([i for i in x if i in 'aeiou'])<y else "".join(list(filter(lambda i: i in 'aeiou',x)))[:y]

def true_equations(x):
    return [i for i in x if eval(i.replace('=','=='))]

def sock_pairs(x):
    return sum(x.count(i)//2 for i in set(x))

def product(x):
    try:
        x = sorted(set(x))
        return x[-2]*x[-1]
    except:
        return x[0]**2

def bound_sort(x,y):
    x = sorted(x[:y[1]+1])
    return all(j-i <=2 for i,j in zip(x,x[1:]))

def neighboring(x):
    return all([abs(ord(j)-ord(i))==1 for i,j in zip(x,x[1:])])

def operate(x,y,z):
    return eval('{}{}{}'.format(x,z,y))

def check_title(x):
    return all(i[0].isupper() for i in x.split())

def is_mini_sudoku(x):
    return sorted([j for i in x for j in i])==list(range(1,10))

def empty_values(x):
    dic = {
        "<class 'list'>":[],
        "<class 'tuple'>":(),
        "<class 'str'>":"",
        "<class 'set'>":set(),
        "<class 'int'>":0,
        "<class 'float'>":0.0,
        "<class 'bool'>":False,
        "<class 'NoneType'>":None
        }
    return [dic[str(type(i))] for i in x]
    return [type(i)() for i in x]

def remove_repeats(x):
    return [x[i] for i in range(len(x)) if x[i:]==x[i+1:]]

def major_sum(x):
    zeros = x.count(0)
    pos = sum([i for i in x if i>0])
    neg = sum([i for i in x if i<0])
    return neg if zeros<abs(neg)>pos else pos if zeros<pos>abs(neg) else zeros

def insert_whitespace(x):
    return "".join([' '+i if i.isupper() else i for i in x])[1:]

def average_word_length(x):
    x = x.translate("".maketrans(".,?!",'    '))
    return round(sum([len(i) for i in x.split()])/len(x.split()),2)

def swimming_pool(x):
    first = all(i==0 for i in x[0])
    last = all(i==0 for i in x[-1])
    first_col = all(i[0]==0 for i in x)
    last_col = all(i[-1]==0 for i in x)
    return all([first,last,first_col,last_col])

def ordered_matrix(x,y):
    return [[y*j+i for i in range(1,y+1)] for j in range(x)]

def smallest(x,y):
    return [int(i) for i in [str(y*i) for i in range(x*100)] if len(i)==x][0]

def join_digits(x):
    return '-'.join(j for i in range(1,x+1) for j in str(i))

def find_broken_keys(x,y):
    z = [i for i,j in zip(x,y) if i!=j]
    return sorted(set(z),key=z.index)

def merge_sort(lst1, lst2):
    return sorted(lst1+lst2,reverse=False) if sorted(lst1)!=lst1[::-1] else sorted(lst1+lst2,reverse=True)

def can_find(x,y):
    return all(i in " ".join(y) for i in x)

def return_negative(x):
    return ~x+1 if x>0 else ~~x

def outlier_number(x):
    evens = [i for i in x if i%2==0]
    odds = [i for i in x if i%2]
    return evens[0] if len(evens)<len(odds) else odds[0]

def height(x):
    return "{:.1f} mm".format((0.5*sqrt(3)*x)*10)

def binary_to_decimal(x):
    return int("".join(map(str,x)),2)

def string_cycling(x,y):
    while len(x)<len(y):
        for i in x:
            x+=i
    if len(x)>len(y):
        return x[:len(y)]

def upload_count(x,y):
    return len([i for i in x if y in i])

def palindromic_date(x):
    x = x.split('/')
    mmddyyyy = x[0]+x[1]+x[2]
    ddmmyyyy = x[1]+x[0]+x[2]
    return mmddyyyy==mmddyyyy[::-1] and ddmmyyyy==ddmmyyyy[::-1]

def completely_filled(x):
    return len([i for i in x if ' ' not in i])==len(x)

def is_anti_list(x,y):
    return all({i,j}==set(x) for i,j in zip(x,y))

def digital_decipher(x,y):
    return "".join([chr(x[i]+96-int((str(y)*(i+1))[i])) for i in range(len(x))])  

class Person:
    def __init__(self,name,foods,hatefoods):
        self.name = name
        self.foods = foods
        self.hatefoods = hatefoods
    
    def taste(self,item):
        return "{} eats the {}!".format(self.name,item+" and loves it" if item in self.foods else item+" and hates it" if item in self.hatefoods else item)

def determine_lever(x):
    return ('third', 'first', 'second')[x.index('f')] + ' class lever'

def is_valid_phone_number(x):
    return bool(re.match(r'^\(\d{3}\)\s\d{3}\-\d{4}$',x))

def get_birthday_cake(x,y):
    x = str(x)
    y = str(y)
    birthday = "%s %s Happy Birthday %s! %s %s" %('#' if int(y)%2==0 else '*',y,x,y,'#' if int(y)%2==0 else '*')
    first = ['#'*len(birthday) if int(y)%2==0 else '*'*len(birthday)]
    last = ['#'*len(birthday) if int(y)%2==0 else '*'*len(birthday)]
    return first+[birthday]+last

def get_drink_ID(x,y):
    return "".join([i[:3] for i in x.split(' ')]).upper()+y.strip('ml')

def digit_count(x):
    x = str(x)
    return int("".join(str(x.count(i)) for i in x))

def parse_code(x):
    first,last,i_d = [i for i in x.split('0') if i.isalnum()]
    return {'first_name':first,
            'last_name':last,
            'id':i_d}

def power_ranger(power,min,max):
    return len([i**power for i in range(1,max+1) if min<=i**power<=max+1])

def count_datatypes(*x):
    lst = [type(i) for i in x]
    return [lst.count(i) for i in (int,str,bool,list,tuple,dict)]

def filter_primes(x):
    return list(filter(lambda i: is_prime(i),x))

def is_prime(x):
    return len([i for i in range(1,abs(x)+1) if abs(x)%i==0])==2

def next_prime(x):
    return [i for i in range(x,x+11) if is_prime(i)][0]

def anna_likes(x):
    return sum(1 for i in x if i in 'aeiouAEIOU')==len(x)/2

def edit_words(x):
    return [i.upper()[::-1][:len(i)//2+1]+'-'+i.upper()[::-1][len(i)//2+1:] if len(i)%2 else i.upper()[::-1][:len(i)//2]+'-'+i.upper()[::-1][len(i)//2:] for i in x]

def sort_by_letter(x):
    return sorted(x,key=lambda i: [k for k in i if k.isalpha()])

def get_frame(x,y,z):
    return "invalid" if x<=2 or y<=2 else [[i] for i in [z*x] + [z+z.rjust(x-2)]*(y-2)+[z*x]]

def dial(x):
    return x.lower().translate("".maketrans(s.ascii_lowercase,'22233344455566677778889999'))

def pluralize(x):
    return {i+'s' if x.count(i)>=2 else i for i in x}

def no_duplicate_letters(x):
    return all([i.count(j)==1 for i in x.lower().split() for j in set(i)])

def interview(x,y):
    times = [5,5,10,10,15,15,20,20]
    return "qualified" if sum(x)<=120 and y<=120 and all(i<=j for i,j in zip(x,times)) else "disqualified"

def palindrome_type(x):
    decimal = str(x)==str(x)[::-1]
    binary = bin(x).strip('b0')==bin(x).strip('b0')[::-1]
    return "{}".format("Decimal only." if decimal and not binary else "Binary only." if binary and not decimal else "Decimal and binary." if all([decimal,binary]) else "Neither!")

def vowel_links(x):
    last = [i[-1] for i in x.split()]
    first = [i[0] for i in x.split()[1:]]
    return any(i in 'aeiou' and j in 'aeiou' for i,j in zip(last,first))

def get_prices(x):
    return [float(i[-6:-1].strip('$')) for i in x]

def add_bill(x):
    d = [i.strip('d') for i in x.split(',') if 'd' in i]
    return sum([int(i.strip('k'))*1000 if 'k' in i else int(i) for i in d])

def count_missing_nums(x):
    x = [int(i) for i in x if i.isdigit()]
    return sum(1 for i in range(min(x),max(x)+1) if i not in x)

def almost_palindrome(x):
    return sum(1 for i,j in zip(x,x[::-1]) if i!=j)==2

def remove_letters(x,y):
    [x.remove(i) for i in y if i in x]
    return x

def is_valid_hex_code(x):
    return bool(re.match(r"#[A-Fa-f0-9]{6}$",x))

def mark_maths(x):
    x = [i.replace('=','==') for i in x]
    return "{:.0f}%".format((sum(1 for i in x if bool(eval(i)))/len(x))*100)

def my_sub(x,y):
    return y if x==0 else my_sub((~y&x)<<1,x^y)

def char_at_pos(x,y):
    return x[1::2] if y=='even' else x[::2]

def secret(x):
    tag,*class_name = x.split('.')
    return "<{} class='{}'></{}>".format(tag," ".join(class_name),tag)

def reorder_digits(x,y):
    return [int("".join(sorted(str(i),reverse=y=='desc'))) for i in x]

def checker_board(x,y,z):
    return "invalid" if y==z else [([y,z]*x)[:x] if i%2 else ([z,y]*x)[1:x+1] for i in range(1,x+1)]

def is_alphabetically_sorted(x):
    return any(["".join(sorted(i))==i for i in x.replace('.','').split() if len(i)>2])

def possibly_perfect(x,y):
    return len([[i,j] for i,j in zip(x,y) if i!=j])==x.count('_') or len([[i,j] for i,j in zip(x,y) if i!=j])==len(x)

def nearest_vowel(x):
    letters = list(enumerate(s.ascii_lowercase,1))
    vowels = [i for i in letters if i[1] in 'aeiou']
    up = letters[s.ascii_lowercase.find(x):]
    down = letters[:s.ascii_lowercase.find(x)+1]
    letter = up[:s.ascii_lowercase.find(x)+1][0][0]
    upp = [abs(i[0]-letter) for i in vowels]
    return x if x in 'aeiou' else min(list(zip(upp,'aeiou')))[1]

def unmix(x):
    return "".join([i+j for i,j in list(zip_longest(x[1::2],x[::2],fillvalue=''))])

def chunkify(x,y):
    # return [x[i:i+y] for i in range(0,len(x),y)]
    x = iter(x)
    return list(iter(lambda: list(islice(x,y)),[]))

def plus_sign(x):
    return len(re.findall(r'\+\w',x))==len([i for i in x if i.isalpha()])

def lengthen(x,y):
    short,long = sorted([x,y],key=len)
    return (short*10)[:len(long)]

def prime_count(x,y):
    return len([i for i in range(x,y+1) if is_prime(i)])

def who_passed(x):
    return sorted([i for i in {key: [eval(i) for i in value] for key, value in x.items() if all(eval(i)==1.0 for i in value)}])

def pad(x):
    return '{}{}{}'.format(x, ' '*(1 - len(x)%2), 'lo'*71)[:140]

def wrap_around(x,y):
    return x[y%len(x):]+x[:y%len(x)]

def min_miss_pos(x):
    x.sort()
    return [i for i in range(1,max(x)+2) if i not in x][0]

def str_match_by2char(x,y):
    x = [x[i:i+2] for i in range(len(x)-1)]
    y = [y[i:i+2] for i in range(len(y)-1)]
    return len([[i,j] for i,j in zip_longest(x,y,fillvalue='') if i==j])

def make_grlex(x):
    return sorted(sorted(x),key=len)

def area_of_country(x,y):
    return "{} is {:.2f}% of the total world's landmass".format(x,(y/148940000)*100)

def primes_below_num(x):
    return [i for i in range(x+1) if is_prime(i)]

def initialize(x):
    x = " ".join(x)
    y = [i[0] for i in x.split()]
    return ["{}. {}.".format("".join(y[i:i+2][0]),"".join(y[i:i+2])[1]) for i in range(len(y)-1)][::2]

def replace(x,y):
    return "".join([i.replace(i,'#') if ord(i) in range(ord(y[0]),ord(y[-1])+1) else i for i in x])

def sort_by_character(x,y):
    return sorted(x,key=lambda i: i[y-1])

def simplify_frac(x):
    return str(fractions.Fraction(eval(x)).limit_denominator())

def missing_letterr(x):
    return [chr(ord(i)+1) for i in x if chr(ord(i)+1) not in x][0]

def generate_palindromes(x):
    return [i for i in range(x+1) if str(i)==str(i)[::-1]][-15:]

def club_entry(x):
    letters = enumerate(s.ascii_lowercase,1)
    double = [i for i,j in zip(x,x[1:]) if i==j][0]
    return [i for i,j in letters if j in double][0]*4

def num_in_str(x):
    return [i for i in x if any(j.isdigit() for j in i)]

def sort_by_string(x,y):
    return sorted(x,key=lambda i: y.index(i[0]))

def reverse_words(x):
    return " ".join([i for i in x.split()[::-1]])

def letter_distance(x,y):
    return sum([abs(ord(i)-ord(j)) for i,j in zip(x[:max([len(x),len(y)])],y[:max([len(x),len(y)])])]) if len(x)==len(y) else sum([abs(ord(i)-ord(j)) for i,j in zip(x[:max([len(x),len(y)])],y[:max([len(x),len(y)])])])+abs(len(x)-len(y))

def validate_spelling(x):
    x = x[:-1]
    firstword = "".join([i.replace('.','').lower().lstrip() for i in x[:x.rfind('.')]])
    lastword = x[x.rfind('.')+2:].lower()
    return firstword==lastword

def word_builder(x,y):
    x = sorted([[j,i] for i,j in zip(x,y)])
    y = [i[1] for i in x]
    return "".join(y)

def check(x):
    return "increasing" if sorted(x)==x and all(x.count(i)==1 for i in x) else "decreasing" if sorted(x)[::-1]==x and all(x.count(i)==1 for i in x) else "neither"

def dolla_dolla_bills(x):
    return "${:,.2f}".format(x).replace('$-','-$')

def prime_in_range(x,y):
    return any(is_prime(i) for i in range(x,y+1))

def no_yelling(x):
    quest = [i for i in x if i=='?']
    esclam = [i for i in x[-5:] if i=='!']
    y = x[:x.find('?'if '?' in x else x[-1])+1]
    z = x[:x.rfind('!' if '!' in x else x[-1])-(len(esclam)-2)]
    return y if '?' in y else z if '!' in z else x

def fat_prime(x,y):
    return [i for i in range(x,y,-1) if is_prime(i)][0] if x>y else [i for i in range(x,y) if is_prime(i)][-1]

def to_camel_case(x):
    left,*right = x.split('_')
    right = [i.capitalize() for i in right]
    right = "".join([j for i in right for j in i])
    return "".join(left+right)

def to_snake_case(x):
    return "".join('_'+i.lower() if i.isupper() else i for i in x)

def total_sales(x,y):
    try: return sum([i[x[0].index(y)] for i in x[1:]])
    except: return 'Product not found'

def replace_the(x):
    x = list(zip(x.split(),x.split()[1:]))
    lastword = ["".join(x[-1][-1])]
    return " ".join([i.replace("the",'an') if j[0] in "aeiou" else i.replace("the",'a') for i,j in x]+lastword)

def missing_letter(x):
    x = [chr(ord(i)+1) for i,j in zip(x,x[1:]) if ord(j)-ord(i)!=1]
    return x[0] if len(x)==1 else 'No Missing Letter'

def where_is_waldo(x):
    waldo = [j for i in x for j in i if i.count(j)==1][0]
    return [[i,j.index(waldo)+1] for (i,j) in enumerate(x,1) if waldo in j][0]

def grab_number_sum(x):
    return sum([int(i) for i in re.findall(r'\d+',x)])

def x_pronounce(x):
    return " ".join([i.replace('x','z') if i.startswith('x') and len(i)!=1 else i.replace('x','ecks') if i=='x' and len(i)==1 else i.replace('x','cks') if 'x' in i else i for i in x.split()])

def turn_calc(x):
    x = "".join([i for i in str(x) if i.isdigit()])
    return x.translate("".maketrans('1234567890',"IZEHSGLB-O"))[::-1]

def dashed(x):
    return "".join(['-'+i+'-' if i in 'aeiouAEIOU' else i for i in x])

def convert(x):
    try:
        num = x.split('*')
        num,degree = float(num[0]),num[1]
        return c_to_f(num) if 'C' in x else f_to_c(num)
    except:
        return "Error"

def c_to_f(x):
    return "{:.0f}*F".format((float(x)*(9/5))+32)
def f_to_c(x):
    return "{:.0f}*C".format((float(x)-32)*(5/9))

def basic_calculator(x,y,z):
    try:
        return eval(str(x)+y+str(z))
    except:
        return None

def print_list(x):
    return list(range(1,x+1))

def regex(x):
    return 'true' if bool(re.findall('\d+[02468]$',x)) else 'false'

def worded_math(x):
    x = x.lower().split()
    sign = '+' if x[1]=='plus' else '-'
    words = {'zero':'0',
            'one':'1'}
    answer = eval(words[x[0]]+sign+words[x[2]])
    return 'Two' if answer==2 else 'One' if answer==1 else 'Zero'

def hours_passed(x,y):
    am = all(['AM' in x and 'AM' in y])
    pm = all(['PM' in x and 'PM' in y])
    x,y = x.split(':'),y.split(':')
    hour = int(y[0])-int(x[0]) if am or pm else 12-int(x[0])+int(y[0])
    return "{}".format(str(hour) + ' hours' if hour>=1 else 'no time passed')

def count_repetitions(x):
    return {key: x.count(key) for key in set(x)}

def arrow(x):
    return ['>'*i for i in range(1,x+1)]+['>'*i for i in range(x-x%2,0,-1)]

def string_pairs(x):
    return list("".join(i) for i in zip_longest(x[::2],x[1::2],fillvalue='*'))

def series_resistance(x):
    values = all(i<=1 for i in x)
    y = round(sum(x),1)
    return "{} {}".format(y,'ohm' if values else 'ohms')

def sum_lst(x):
    total = 0
    for i in range(len(x)):
        total += x[i]
    return total

def little_big(x,k=100):
    y = [i for i in range(5,20)]
    result = []
    while k<=1700000:
        result.append(k)
        k*= 2
    answer = [j for i in zip_longest(y,result,fillvalue='') for j in i]
    return answer[x-1]


def remove_last_vowel(x):
    return re.sub(r'[aeiou](?=[^aeiou]*[\s\.])','',x)

def sum_missing_numbers(x):
    return sum(i for i in range(min(x),max(x)) if i not in x)

def map_letters(x):
    return {key: [i for i,j in enumerate(x) if j==key] for key in set(x)}

def has22(x):
    x = "".join([str(i) for i in x])
    y = [i for i in range(len(x)) if x[i:i+1]=='2']
    answer = [i for i in [True if j-i==1 else None for i,j in zip(y,y[1:])] if i==True]
    return answer[0] if len(answer)>=1 else False

def centered_average(x):
    x.sort()
    [x.remove(i) for i in x if x.count(i)>2]
    return sum(sorted(set(x))[1:-1])

def correct_sentences(x):
    y = " ".join(['. '+j if j[0].isupper() else j for i,j in enumerate(x.split())]).replace(' .','.')
    return y[0].upper()+y[1:]+'.'

def is_powerful(x):
    values = [i for i in range(1,x+1) if x%i==0]
    primes = [i for i in range(1,x+1) if x%i==0 and is_prime(i)]
    return all(i**2 in values for i in primes)

def forbidden_letter(x,y):
    y = [j for i in y for j in i]
    return not any(i==j for i in y for j in x)

def simple_encoder(x):
    x = [i for i in x.lower()]
    return "".join([i.replace(i,'[') if x.count(i)<2 else i.replace(i,']') for i in x])

def fruit_salad(x):
    return "".join(sorted([j for i in [split_letters(i) for i in x] for j in i]))

def split_letters(x):
    return [x[:len(x)//2]]+[x[len(x)//2:]]

def wiggle_string(x):
    upper = [(' '*i)+x for i in range(len(x)+1)]
    lower = [upper[i] for i in range(len(upper)-2,-1,-1)]
    return upper+lower

def words_to_sentence(x):
    y = ", ".join([i for i in x if i!='']) if x!=None else ''
    z = [i for i in y.split(', ')][-1]
    return "".join(y[:y.rfind(',')])+" and "+z if len(y.split(', '))>=2 else y if len([y])==1 else ''

def mumbling(x):
    y = [x[i]*(i+1) for i in range(len([i for i in x]))]
    return "-".join([i.capitalize() for i in y])

def invert(x):
    return " ".join([i.swapcase()[::-1] for i in x.split()][::-1])

def parallel_resistance(x):
    return round(sum([(1/i) for i in x])**-1,1)

def monkey_talk(x):
    y = " ".join(i.replace(i,'eek') if i[0] in 'aeiou' else i.replace(i,'ook') for i in x.lower().split())
    return y.capitalize()+'.'

def absolute(x):
    return " ".join(i.replace(i,'an absolute') if i=='a' else i for i in x.lower().split()).capitalize()

def odd_sum_list(x):
    return [sum(i)%2==0 for i in [x[i:i+2] for i in range(len(x))][:-1]]

def digital_clock(x):
    hours = (x//3600)%24
    minutes = (x//60)%60
    seconds = x%60
    return "{:02}:{:02}:{:02}".format(hours,minutes,seconds)

def sum_minimums(x):
    return sum([sorted(i)[0] for i in x])

def average_index(x):
    letters = [i for i in enumerate(s.ascii_lowercase,1)]
    y = [i for i,j in letters for k in x if k==j]
    return round(sum(y)/len(y),2)


def name_score(x):
    scores = {
        "A": 100, "B": 14, "C": 9, "D": 28, "E": 145, "F": 12, "G": 3,
        "H": 10, "I": 200, "J": 100, "K": 114, "L": 100, "M": 25,
        "N": 450, "O": 80, "P": 2, "Q": 12, "R": 400, "S": 113,
        "T": 405, "U": 11, "V": 10, "W": 10, "X": 3, "Y": 210, "Z": 23
        }
    total = sum([scores[i] for i in x if i!=' '])
    return "PRETTY GOOD" if 61<=total<=300 else "VERY GOOD" if 301<=total<=599 else "THE BEST" if total>=600  else "NOT TOO GOOD"
