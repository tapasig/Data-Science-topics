## 3. Introduction to SymPy ##

#Import SymPy and declare x and y as SymPy symbols.

import sympy
x,y = sympy.symbols('x y')

y = x^2 + 1

y = 3*x
print(y)

## 4. Limits Using SymPy ##

import sympy
x2,y = sympy.symbols('x2 y')

y = -(x2**2) +3*x2 -1
limit_one = sympy.limit((y+1)/(x2-3), x2, 2.9)

print(limit_one)

## 5. Properties Of Limits I ##

import sympy
x,y = sympy.symbols('x y')

limit_two = sympy.limit((3*(x**2) + 3*x -3),x,1)
print(limit_two)

## 6. Properties Of Limits II ##

import sympy
x,y = sympy.symbols('x y')

limit_three = sympy.limit(x**3,x,-1)+2*sympy.limit(x**2,x,-1) - 10*sympy.limit(x,x,-1)

limit_three = sympy.limit((x**3+2*(x**2) - 10*x),x,-1)

print(limit_three)

## 7. Undefined Limit To Defined Limit ##

import sympy
x2, y = sympy.symbols('x2 y')

limit_four = sympy.limit((-x2**2 + 3*x2)/(x2 -3),x2,3)

print(limit_four)
