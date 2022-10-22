from __future__ import print_function
import matplotlib.pyplot as plt
import random
import numpy as np

    
ab = np.random.rand(1,2)
w = ab[0][0]
z = ab[0][1]
s = np.random.normal(0,1,100)

def noise(s):
    return np.random.choice(s)

#funciton create numbers with noise 
def func(x):
    n = noise(s)
    cal = w*x + z +n

    return cal

#where i create 100 number from 0-1
def Random():
    x_cor = []
    y_cor = []
    for t in range(101):
        k = random.uniform(0, 1)
        x = k / 100
        y = func(x)
        x_cor.append(x)
        y_cor.append(y)
   

    return x_cor,y_cor

def gradient_descent(x,y):
    
    m_curr = b_curr = 0  #initial guess
    limit_iteration = 1000
    iterations = 0
    n = len(x)
    learning_rate = 0.08
    last_cost = 0
    eva_fun = 0
    costs = []
    guesses = []

    
    for i in range(limit_iteration):
        guess = []
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        costs.append(cost)
        guess.append(m_curr)
        guess.append(b_curr)
        guesses.append(guess)
        eva_fun += 1
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        

        # if(abs(cost-last_cost) < 0.001):
        #     break
        last_cost = cost
        iterations = i 
        
  
    print ("exhaustive search : iteration={} ,cost={} ,evaluation of function={} ".format(iterations+1,last_cost,eva_fun))   
    num = np.linspace(0,0.01,50)
    plt.plot(num,m_curr*num+b_curr,label='exhaustive search')
    print()
 
def gradient_descent2(x,y):
    
    m_curr = b_curr = 0  #initial guess
    limit_iteration = 1000
    iterations = 0
    n = len(x)
    learning_rate = 0.08
    last_cost = 0
    eva_fun = 0
    costs = []
    guesses = []

    
    for i in range(limit_iteration):
        guess = []
        y_predicted = m_curr / 1 + (x* b_curr)
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        costs.append(cost)
        guess.append(m_curr)
        guess.append(b_curr)
        guesses.append(guess)
        eva_fun += 1
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        

        # if(abs(cost-last_cost) < 0.001):
        #     break
        last_cost = cost
        iterations = i 
        
  
    print ("exhaustive search with rational approximant : iteration={} ,cost={} ,evaluation of function={} ".format(iterations+1,last_cost,eva_fun))   
    num = np.linspace(0,0.01,50)
    plt.plot(num,m_curr / 1 + (num* b_curr),label='exhaustive search with rational approximant')
    print()



    

        
from scipy import optimize
from scipy import misc
from sympy import *


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

  
# Define the Gaussian function
def Gauss(x, A, B):
    y = A*x + B
    return y




from numpy.random import rand
from scipy import optimize
from scipy.optimize import minimize 




def Nelder_Mead(x,y):

    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
 
    array = []
    array.append(x)
    array.append(y)
    current_node= [1,1]
    n = len(x)
    
 
    def Fun(array):
        
        n = len(x)

        z = (1/n)*sum([(val)**2 for val in (y -(a * x + b))])
    
        result = z.subs([(a,array[0]),(b,array[1])])
        return result
 

 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(Fun, current_node, method='nelder-mead', tol=0.001)
    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = Fun(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    # print(result)

    x = np.linspace(0,0.01,50)
    # y = current_node[0] / (1+current_node[1]*x)
    y = (solution[0]*x + solution[1])
 
    plt.plot(x,y,label='Nelder_Mead')
    print('Nelder_Mead  :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
          .format(current_node,result['nit'],result['fun'], result['nfev'])) 
    print()
def Nelder_Mead2(x,y):

    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
 
    array = []
    array.append(x)
    array.append(y)
    current_node= [1,1]
    n = len(x)
    
 
    def Fun(array):
        
        n = len(x)


        z = (1/n)*sum([(val)**2 for val in (y -(a / (1+b*x)))])
        result = z.subs([(a,array[0]),(b,array[1])])
        return result
 

 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(Fun, current_node, method='nelder-mead', tol=0.001)
    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = Fun(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    # print(result)

    x = np.linspace(0,0.01,50)
    # y = current_node[0] / (1+current_node[1]*x)

    y = (solution[0] / 1 + (x * solution[1]))
    plt.plot(x,y,label='Nelder_Mead with rational approximant')
    print('Nelder_Mead with rational approximant :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
          .format(current_node,result['nit'],result['fun'], result['nfev']))  
    print()



def Gauss(x,y):

    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
 
    array = []
    array.append(x)
    array.append(y)
    current_node= [1,1]
    n = len(x)
    
 
    def Fun(array):
        
        n = len(x)

        z = (1/n)*sum([(val)**2 for val in (y -(a * x + b))])
        result = z.subs([(a,array[0]),(b,array[1])])
        return result
 

 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(Fun, current_node, method='CG', tol=0.001)
    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = Fun(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    # print(result)

    x = np.linspace(0,0.01,50)
    # y = current_node[0] / (1+current_node[1]*x)
    y = (solution[0]*x + solution[1])
    plt.plot(x,y,label='Gauss')
    print('Gauss  :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
          .format(current_node,result['nit'],result['fun'], result['nfev']))  
    print()
        
def Gauss2(x,y):

    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
 
    array = []
    array.append(x)
    array.append(y)
    current_node= [1,1]
    n = len(x)
    
 
    def Fun(array):
        
        n = len(x)

    
        z = (1/n)*sum([(val)**2 for val in (y -(a / (1+b*x)))])
        result = z.subs([(a,array[0]),(b,array[1])])
        return result
 

 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(Fun, current_node, method='CG', tol=0.001)
    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = Fun(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    # print(result)

    x = np.linspace(0,0.01,50)
    # y = current_node[0] / (1+current_node[1]*x)
    y = (solution[0] / 1 + (x * solution[1]))
    plt.plot(x,y,label='Gauss with rational approximant ')
    print('Gauss with rational approximant  :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
          .format(current_node,result['nit'],result['fun'], result['nfev']))  
    print()


def main():
    x_cor , y_cor = Random()
    plt.scatter(x_cor,y_cor)
    x_cor = np.array(x_cor)
    y_cor = np.array(y_cor)
    gradient_descent(x_cor, y_cor)
    Nelder_Mead(x_cor, y_cor)
    Gauss(x_cor, y_cor)
    plt.legend()
    plt.show()
    
    gradient_descent2(x_cor, y_cor)
    Nelder_Mead2(x_cor, y_cor)
    Gauss2(x_cor, y_cor)
    plt.scatter(x_cor,y_cor)
    
    plt.legend()
    plt.show()

      
if __name__ == "__main__":
    main()    

    
    
    