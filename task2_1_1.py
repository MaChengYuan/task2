import numpy as np
import math
from  matplotlib import pyplot as plt
from scipy.optimize import minimize
from numpy.random import rand
from numpy.random import randn


class bisection:
    counter = 0
    def __call__(self,a,b):
     
        iter = 0
        while ((b-a) >= 0.001):
            iter = iter + 1 
            # Find middle point
            c = (a+b)/2
      
            
            if (equation(c) <= equation(a) and equation(c) >= equation(b)):
                bisection.counter += 4
                a = c
            elif(equation(c) <= equation(b) and equation(c) >= equation(a)):
                bisection.counter += 4
                b = c
            else:
                if(equation(a)>=equation(b)):
                    bisection.counter += 2
                    a = c
                else:
                    b = c
        print('iteration is  : %.0f' %iter) 
        print("The value of root is : ","%.4f"%c)
        
        plt.plot(c,equation(c),'r*',color='red')
        bisection.counter += 1
        print('evaluation of objective function is : %.0f'%bisection.counter)
        print('\n\n')
        plt.title("Bisection")  
     




def equation(x):
    return x**3 

class exhaustive:
    counter = 0 
    def __call__(self,a , b , n):
        x1 = a
        dx = (b-a)/n 
        x2 = x1 + dx
        x3 = x2 + dx
        iter = 0
        while (x3 <= b):
            iter = iter + 1 
            y1 = equation(x1)
            y2 = equation(x2)
            y3 = equation(x3)
            exhaustive.counter += 3 
            if y1>=y2 and y2<=y3:
                
                 print ('The minimum point lies between ',x1,' & ',x3)
                 plt.plot(x1,equation(x1),'r*',label = '$lower point$')
                 exhaustive.counter += 1
                 print('evaluation of objective function is : %.0f'%exhaustive.counter)
                 
              
                 break
            else:
                x1 = x2
                x2 = x3
                x3 = x2 + dx
                if(x3 > b):
                    print('bounday is minimun')
                    plt.plot(a,equation(a),'r*',label = '$lower point$')
                    exhaustive.counter+= 1
                    print('evaluation of objective function is : %.0f'%exhaustive.counter)
                    
                    break
        plt.title("Exhaustive")  
        print('iteration is  : %.0f' %iter) 
        print('\n\n')
class golden_section:
    counter = 0
    def __call__(self,a , b):
            iter = 0 
            gr = (math.sqrt(5)-1)/2
            start = a
            end = b
            x = np.arange(start,end,1e-6)
            y = equation(x) 
            golden_section.counter += 1
            while(abs(end-start) >= 0.001):
                iter += 1
               
                d = gr*(end-start)
                x1 = d+start
                x2 = end-d
               
                if(equation(x1) > equation(x2)):
                    golden_section.counter += 2
                    end = x1 
                elif (equation(x1)==equation(x2)):
                    golden_section.counter += 2
                    break
                else:
                    start = x2
        
        
            plt.plot(x,y,label='function')  
            plt.plot(start,equation(start),'r*',label = '$lower point$')
            golden_section.counter += 1
            print('evaluation of objective function is : %.0f'%golden_section.counter)
            print('iteration is  : %.0f' %iter) 
            plt.legend()
            print('\n\n')
            plt.title("Golden_section")  
           
           
            return a




  
def main():
    a = 1e-6
    b = 1
    n = 1000
    
    x = np.arange(a,b,1e-6)
    y = equation(x)
    plt.subplot(131)
    plt.plot(x,y)    
    Bisec = bisection()
    Bisec(a, b)
    

    plt.subplot(132)
    
    plt.plot(x,y)
    Exhaustive = exhaustive()
    Exhaustive(a,b,n)
    
    
    plt.subplot(133)

    plt.plot(x,y)
    Gold = golden_section()
    Gold(a,b)
    

      
  
if __name__ == "__main__":
    main()    


