import random 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ab = np.random.rand(1,2)
w = ab[0][0]
z = ab[0][1]
x_cor = []
y_cor = []
s = np.random.normal(0,1,100)

def noise(s):
    return np.random.choice(s)

#funciton create numbers with noise 
def func(x):
    n = noise(s)
    cal = w*x + z +n

    return cal

#where i create 100 number from 0-1
def random():
    k = np.linspace(0,1,101)
    x = k / 100
    y = func(x)
    x_cor.append(x)
    y_cor.append(y)
    plt.scatter(x,y)

    return
    



def main():

    random()

    print(y_cor[0])
    
    [a, b], res1 = curve_fit(lambda x1,a,b: a*x1+b,  x_cor[0],  y_cor[0])
    #where i use api to find best fit line
    detail = curve_fit(lambda x1,a,b: a*x1+b,  x_cor[0],  y_cor[0],full_output=True)
    print(detail)
    y1 = a * x_cor[0] + b
    plt.subplot(221)
    plt.plot(x_cor[0], y_cor[0], 'b')
    plt.plot(x_cor[0], y1, 'r')
    plt.title("linearapproximant")
    print('detail')
    
    
    print('------------------------')


    
    
    [a, b], res1  = curve_fit(lambda x1,a,b: a/(1+b*x1) ,  x_cor[0],  y_cor[0])
    detail = curve_fit(lambda x1,a,b: a/(1+b*x1) ,  x_cor[0],  y_cor[0],full_output=True)
    y1 = a/1+b*x_cor[0]
    print(detail)
    plt.subplot(224)
    plt.plot(x_cor[0], y_cor[0], 'b')
    plt.plot(x_cor[0], y1, 'r')
    plt.title("rational approximant")
    


    

if __name__ == "__main__":
    main()





                 
