# CS 352 - Machine Learning
# Final Project
# Due: December 23rd, 2018
# Malcolm Gilbert and Weronika Nguyen

# This is a slightly modified formula and image generator from More's github.
# Link: https://github.com/avinashmore/data-generation-latex-cnn/blob/master/latex-eq/equations.py

from io import BytesIO
import matplotlib.pyplot as plt
import string
import random
import numpy as np
import pickle
import sys

letters = tuple([i for i in string.ascii_lowercase[0:26]])
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
operation1 = ('frac', 'int', '^', 'frac', 'sum', 'int', '^', 'sum')
operation = ('frac', 'int', 'sqrt', 'sum')
plt.rcParams['font.sans-serif'] = "Comic Sans MS"
plt.rcParams['font.family'] = "sans-serif"
data_size = 10
test_data_size = 10
y = np.empty((data_size, 4), dtype=int)
eq = []

def render_latex(formula, fontsize=10, dpi=300, format_='svg'):
    fig = plt.figure(figsize=(0.3, 0.25))
    fig.text(0.05, 0.35, '${}$'.format(formula), fontsize=fontsize)
    buffer_ = BytesIO()
    fig.savefig(buffer_, dpi=dpi, transparent=True, format=format_, pad_inches=0.0)
    plt.close(fig)
    return buffer_.getvalue()

def generate_data_simple(image_count):
    for i in range(0, image_count):
        op = random.randint(0, len(operation) - 1)
        letter1 = letters[random.randint(0, len(letters) - 1)]
        letter2 = letters[random.randint(0, len(letters) - 1)]
        letter3 = letters[random.randint(0, len(letters) - 1)]

        if operation[op] == 'frac':
            y[i] = [1, 0, 0, 0]
            expression = r'\frac' + '{' + letter1 + '}' + '{' + letter2 + '}' 
        elif operation[op] == 'sqrt':
            y[i] = [0, 1, 0, 0]
            expression = r'\sqrt' + '{' + letter1 + letter2 + '}'
        elif operation[op] == 'sum':
            y[i] = [0, 0, 1, 0]
            number = numbers[random.randint(0, len(numbers) - 1)]
            expression = r'\sum_{' + letter1 + '=1}^{\infty}' + number + '^{' + letter1 + '}'
        elif operation[op] == 'int':
            y[i] = [0, 0, 0, 1]
            expression = r'\int_{' + letter1 + '} ^ {' + letter2 + '}' + letter3 + '^ 2d' + letter3 
        


        #eq.append(expression)
        image_bytes = render_latex(expression, fontsize=5, dpi=200, format_='png')
        if operation[op] == 'frac':
            expression += '\empty\empty\empty\empty\empty\empty\empty\empty\empty'
        elif operation[op] == 'sqrt':
            expression += '\empty\empty\empty\empty\empty\empty\empty\empty\empty\empty\empty'
        elif operation[op] == 'int':
            expression += '\empty\empty'
        eq.append(expression)

        image_name = './train-data/' + str(i) + '.png'
        with open(image_name, 'wb') as image_file:
            image_file.write(image_bytes)

    pickle.dump(y, open("save.p", "wb"))
    pickle.dump(eq, open("equations.p", "wb"))
    with open("train-data.txt",'w') as eqfile:
        for item in eq:
            eqfile.write("%s\n" % item)

def generate_data_complex(image_count):
    for i in range(image_count):
        pass

if __name__ == '__main__':
    numimages = int(sys.argv[1])
    data_size = numimages
    y = np.empty((data_size, 4), dtype=int)
    generate_data_simple(numimages)