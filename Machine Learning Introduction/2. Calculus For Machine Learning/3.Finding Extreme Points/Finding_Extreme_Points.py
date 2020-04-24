## 3. Differentiation ##

import matplotlib.pyplot as plt
import numpy as np
import sympy

h = sympy.symbols('h')

x = np.linspace(-5,6,110)
y = -2*x +3
#y =sympy.limit( (-2*x -h +3),h,0)



plt.plot(x,y)
plt.show()

## 6. Power Rule ##

import sympy
x = sympy.symbols('x')

slope_one = sympy.limit(5*(x**4),x,2)
slope_two = sympy.limit(9*(x**8),x,0)

slope_one =5*(2**4)
slope_two =9*(0**8)

## 7. Linearity Of Differentiation ##

slope_three = 5*(1**4) -1

slope_four = 3*(2**2) -2*2

## 8. Practicing Finding Extreme Values ##


crticial_points = [0,2/3]
rel_min = [2/3]
rel_max = [0]
