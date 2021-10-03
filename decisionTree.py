from os import replace
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from collections import Counter

def get_random_dataset(X):
  n_samples = X.shape[0]
  idxs = np.random.choice(n_samples, size=n_samples, replace=True)
  return X.iloc[idxs,:]

def entropy(y):
  if isinstance(y, pd.Series):
    # Randam skirtingų reikšmių kiekį stulpelyje.
    counts = y.value_counts()
    # Randam kokio ilgio yra užduotas stulpelis.
    shape = y.shape[0]
    # Gaunam sąrašą kiek procentaliai kiek kiekviena reikšmė pasikartoja
    a = counts/shape
    # Suskaičiuojam entropiją kiekvienai reikšmei ir visas jas sudedam
    entropy = np.sum(-a*np.log2(a+1e-9))
    return(entropy)

  else:
    raise('Object must be a Pandas Series.')

def variance(y): 
  # Suskaičiuojamas nuokrypis
  if(len(y) == 1):
    return 0
  else:
    return y.var()

def information_gain(y, mask, func=entropy):
  a = sum(mask)
  b = mask.shape[0] - a
  
  if(a == 0 or b ==0): 
    ig = 0
  
  else:
    if y.dtypes != 'O':
      ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
    else:
      ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])
  
  return ig

df = pd.read_csv("train.csv",sep=',', index_col='Id')
df = df.replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5,4,3,2,1], )
cols_to_norm = ['YearRemodAdd', 'OverallQual', 'OverallCond', 'LotArea']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df, test = train_test_split(df, test_size=0.2)
print(type(test))
#print(information_gain(df["SalePrice"], df["HeatingQC"] == "Ex"))

import itertools

def categorical_options(a):
  '''
  Creates all possible combinations from a Pandas Series.
  a: Pandas Series from where to get all possible combinations. 
  '''
  a = a.unique()

  opciones = []
  for L in range(0, len(a)+1):
      for subset in itertools.combinations(a, L):
          subset = list(subset)
          opciones.append(subset)
  return opciones[1:-1]

def max_information_gain_split(x, y, func=entropy):
  '''
  Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
  x: predictor variable as Pandas Series.
  y: target variable as Pandas Series.
  func: function to be used to calculate the best split.
  '''

  split_value = []
  ig = [] 

  numeric_variable = True if x.dtypes != 'O' else False

  # Create options according to variable type
  if numeric_variable:
    options = x.sort_values().unique()[1:]
  else: 
    options = categorical_options(x)

  
  # Calculate ig for all values
  for val in options:
    mask =   x < val if numeric_variable else x.isin(val)
    print(mask)
    val_ig = information_gain(y, mask, func)
    # Append results
    ig.append(val_ig)
    split_value.append(val)

  # Check if there are more than 1 results if not, return False
  if len(ig) == 0:
    return(None,None,None, False)

  else:
  # Get results with highest IG
    best_ig = max(ig)
    best_ig_index = ig.index(best_ig)
    best_split = split_value[best_ig_index]
    return(best_ig,best_split,numeric_variable, True)

def get_best_split(y, data):
  masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
  if sum(masks.loc[3,:]) == 0:
    return(None, None, None, None)

  else:
    # Get only masks that can be splitted
    masks = masks.loc[:,masks.loc[3,:]]

    # Get the results for split with highest IG
    split_variable = max(masks)
    #split_valid = masks[split_variable][]
    split_value = masks[split_variable][1] 
    split_ig = masks[split_variable][0]
    split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)


def make_split(variable, value, data, numeric):
    
  if numeric:
    data_1 = data[data[variable] < value]
    data_2 = data[(data[variable] < value) == False]

  else:
    data_1 = data[data[variable].isin(value)]
    data_2 = data[(data[variable].isin(value)) == False]
  
  #splitsMade.append(variable)
  return(data_1,data_2)

def make_prediction(data, target_factor):
 
  # Make predictions
  if target_factor:
    pred = data.value_counts().idxmax()
  else:
    pred = data.mean()

  return pred

def train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
 
  # Check that max_categories is fulfilled
  if counter==0:
    types = data.dtypes
    check_columns = types[types == "object"].index
    for column in check_columns:
      var_length = len(data[column].value_counts()) 
      if var_length > max_categories:
        raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

  # Check for depth conditions
  if max_depth == None:
    depth_cond = True

  else:
    if counter < max_depth:
      depth_cond = True

    else:
      depth_cond = False

  # Check for sample conditions
  if min_samples_split == None:
    sample_cond = True

  else:
    if data.shape[0] > min_samples_split:
      sample_cond = True

    else:
      sample_cond = False

  # Check for ig condition
  if depth_cond & sample_cond:

    var,val,ig,var_type = get_best_split(y, data)

    # If ig condition is fulfilled, make split 
    if ig is not None and ig >= min_information_gain:

      counter += 1

      left,right = make_split(var, val, data,var_type)

      # Instantiate sub-tree
      split_type = "<=" if var_type else "in"
      question =   "{} {}  {}".format(var,split_type,val)
      # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
      subtree = {question: []}


      # Find answers (recursion)
      yes_answer = train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

      no_answer = train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

      if yes_answer == no_answer:
        subtree = yes_answer

      else:
        subtree[question].append(yes_answer)
        subtree[question].append(no_answer)

    # If it doesn't match IG condition, make prediction
    else:
      pred = make_prediction(data[y],target_factor)
      return pred

   # Drop dataset if doesn't match depth or sample conditions
  else:
    pred = make_prediction(data[y],target_factor)
    return pred

  return subtree


max_depth = 5
min_samples_split = 20
min_information_gain  = 1e-5

randomTreesNumber = 10


generatedTrees = []
for _ in range(randomTreesNumber):

  randomSet = get_random_dataset(df)

  decisiones = train_tree(randomSet,'SalePrice',True, max_depth,min_samples_split,min_information_gain)
  generatedTrees.append(decisiones)
  print(_)



def clasificar_datos(observacion, arbol):
  question = list(arbol.keys())[0] 

  if question.split()[1] == '<=':

    if observacion[question.split()[0]] <= float(question.split()[2]):
      answer = arbol[question][0]
    else:
      answer = arbol[question][1]

  else:

    if observacion[question.split()[0]] in (question.split()[2]):
      answer = arbol[question][0]
    else:
      answer = arbol[question][1]

  # If the answer is not a dictionary
  if not isinstance(answer, dict):
    return answer
  else:
    return clasificar_datos(observacion, answer)


print("Test accuracy = ", 1-sum(count)/len(count))

count = []
for i, real_val in enumerate(df['SalePrice']):
  pred_val =0
  for tree in generatedTrees:
    pred_val += clasificar_datos(df.iloc[i,:], tree)
  pred_val = pred_val/len(generatedTrees)
  acc = abs((real_val-pred_val)/real_val)
  #print(acc)
  count.append(acc)
print("Train accuracy = ", 1-sum(count)/len(count))


