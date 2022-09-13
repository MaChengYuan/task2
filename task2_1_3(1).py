
from scipy import misc
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
 
# objective function
def func(x):
	return np.sin(1/x)


class gauss:
    counter = 0
    def __call__(self,a , b):
        
        start = a 
        end = b
        alpha = 0.1 # learning rate
        nb_max_iter = 100 # Nb max d'iteration
        eps = 0.0001 # stop condition
        
        x0 = start # start point
        y0 = func(x0)
        
        
        cond = eps + 10.0 # start with cond greater than eps (assumption)
        nb_iter = 0 
        tmp_y = y0
        while cond > eps and nb_iter < nb_max_iter and x0 >= start and x0 <= end:
            x0 = x0 - alpha * misc.derivative(func, x0)
            y0 = func(x0)
            gauss.counter += 1
            nb_iter = nb_iter + 1
            cond = abs( tmp_y - y0 )
            tmp_y = y0
        
        print('iteration is  : %.0f'%nb_iter)
        print ('lowest point %f,%f'%(x0,y0))
        print('evaluation of objective function is : %.0f'%gauss.counter)
        plt.scatter(x0, y0)
        plt.grid()
    
        plt.title("Gradient Descent Python (1d test)")


class nelder:
    counter = 0
    def __call__(self,a,b):
        nelder.counter += 1
        x0 = a
        ans = optimize.minimize(func, x0)
        minX = optimize.minimize(func, x0).x
        minY = func(minX)
        print(ans)
        plt.plot(minX , minY,'r*',label = '$lower point$')
        
        plt.title("Nelder-Mead")
        print("iteration is : %.f"%ans.nit)
        print('evaluation of objective function is : %.0f'%ans.nfev)
        plt.legend()    
        plt.grid()
        plt.show()
        

    

  
def main():
    a = 1e-6
    b = 1
    
    x = np.arange(a,b,1e-6)
    y = func(x)
    plt.subplot(121)
    plt.plot(x,y)    
    Nelder = nelder()
    Nelder(a, b)

    
    print('\n')
    
    
    plt.subplot(122)
    plt.plot(x,y)
    Gauss = gauss()
    Gauss(a, b)
    
    
 
      
  
if __name__ == "__main__":
    main()   


