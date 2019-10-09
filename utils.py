# downloaded from: https://github.com/Miffyli/im2latex-dataset

import re
import random 
import argparse

# regexp used to tokenize formula
#   Pattern /[$-/:-?{-~!"^_`\[\]]/ was taken from:
#       http://stackoverflow.com/questions/8359566/regex-to-match-symbols
TOKENIZE_PATTERN = re.compile("(\\\\[a-zA-Z]+)|"+ # \[command name]
                              #"(\{\w+?\})|"+ # {[text-here]} Check if this is needed
                              "((\\\\)*[$-/:-?{-~!\"^_`\[\]])|"+ # math symbols
                              "(\w)|"+ # single letters or other chars
                              "(\\\\)") # \ characters

# regexps for removing "invisible" parts
# First item is regexp for searching, second is string/func used to replace
INVISIBLE_PATTERNS = [[re.compile("(\\\\label{.*?})"), ""],
                      [re.compile("(\$)"), ""],
                      [re.compile("(\\\>)"), ""],
                      [re.compile("(\\\~)"), ""],
                     ]

# regexps for normalizing
# First item is regexp for searching, second is string/func used to replace
NORMALIZE_PATTERNS = [[re.compile("\{\\\\rm (.*?)\}"), 
                            lambda x: "\\mathrm{"+x.group(1)+"}"],
                      [re.compile("\\\\rm{(.*?)\}"), 
                            lambda x: "\\mathrm{"+x.group(1)+"}"],
                      [re.compile("SSSSSS"), "$"],
                      [re.compile(" S S S S S S"), "$"],
                     ]
                

def tokenize_formula(formula):
    """Returns list of tokens in given formula.
    formula - string containing the LaTeX formula to be tokenized
    Note: Somewhat work-in-progress"""
    # Tokenize
    tokens = re.finditer(TOKENIZE_PATTERN, formula)
    # To list
    tokens = list(map(lambda x: x.group(0), tokens))
    # Clean up
    tokens = [x for x in tokens if x is not None and x != ""]
    return tokens

def remove_invisible(formula):
    """Removes 'invisible' parts of the formula.
    Invisible part of formula is part that doesn't change rendered picture, 
    eg. \label{...} doesn't change the visual output of formula 
    formula -- formula string to be processed 
    Returns processed formula
    Note: Somewhat work-in-progress"""
    for regexp in INVISIBLE_PATTERNS:
        formula = re.sub(regexp[0], regexp[1], formula)
    return formula
    
def normalize_formula(formula):
    """Normalize given formula string.
    Normalisation attempts to eliminate multiple different ways of writing
    same thing. Eg. 'x^2_3' results to same output as 'x_3^2', and normalisation
    would turn all of these to same form
    formula -- formula string to be normalised
    Returns processed formula
    Note: Somewhat work-in-progress"""
    for regexp in NORMALIZE_PATTERNS:
        formula = re.sub(regexp[0], regexp[1], formula)
    return formula