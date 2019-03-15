# coding: utf-8

#% Copyright (c) 2016 Alejandro de Haro (original - matlab code)
#% Copyright (c) 2018 Nguyen Ngoc Sang  

# % NACA Airfoil Generator
# % This function generates a set of points containing the coordinates of a
# % NACA airfoil from the 4 Digit Series, 5 Digit Series and 6 Series given
# % its number and, as additional features, the chordt, the number of points
# % to be calculated, spacing type (between linear and cosine spacing),
# % opened or closed trailing edge and the angle of attack of the airfoil.
# % It also plots the airfoil for further comprovation if it is the required
# % one by the user.
# %
# % INPUT DATA
# %   series --> NACA number (4, 5 or 6 digits)
# %
# % OPTIONAL INPUT DATA
# %   alpha --> Angle of attack (º) (0º default)
# %   c     --> Chord of airfoil (m) (1 m default)
# %   s     --> Number of points of airfoil (1000 default)
# %   cs    --> Linear or cosine spacing (0 or 1 respectively) (1 default)
# %   cte   --> Opened or closed trailing edge (0 or 1 respectively) (0 default)
# %
# % OUTPUT DATA
# %   x_e --> Extrados x coordinate of airfoil vector (m)
# %   x_i --> Intrados x coordinate of airfoil vector (m)
# %   y_e --> Extrados y coordinate of airfoil vector (m)
# %   y_i --> Intrados y coordinate of airfoil vector (m)

import numpy as np
from numpy import pi, sin, cos, tan, arctan, log, sign, linspace, zeros, append
import matplotlib.pyplot as plt   


def NACA(series, alpha=0.0, c=1.0, s=200, cs=1, cte=0):
    n = int(series)   #%NACA number
    nc = len(series)  #%number of digits (4, 5, 6)
    
    if cs==0:
        x = linspace(0,1, s)        #% X coordinate of airfoil (linear spacing)
    else:
        beta = linspace(0, pi, s)   #% Angle for cosine spacing
        x = (1.0 - cos(beta))/2     #% X coordinate of airfoil (cosine spacing) 
        
    t = (n%100)/100.        #% Maximum thickness as fraction of chord (two last digits)
    sym = 0                 #% Symetric airfoil variable
    alpha_deg = alpha       #% Angle of attack - degrees
    alpha = alpha/180*pi    #% Conversion of angle of attack from degrees to radians
    
    #%----------------------- VARIABLE PRELOCATION -----------------------------
    y_c = zeros(s)      #% Mean camber vector prelocation
    dyc_dx = zeros(s)   #% Mean camber fisrt derivative vector prelocation
    
    #%----------------------- THICKNESS CALCULATION ----------------------------
    if cte==1:  #% Thickness y coordinate with closed trailing edge
        y_t = t/0.2*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    else:       #% Thickness y coordinate with opened trailing edge
        y_t = t/0.2*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        
    #%----------------------- MEAN CAMBER 4 DIGIT SERIES CALCULATION -----------   
    if nc == 4:
        m = int(series[0])/100.     #% Maximum camber (1st digit)
        p = int(series[1])/10.      #% Location of maximum camber (2nd digit)
        if m == 0:
            if p == 0: sym = 1  # % Comprovation of symetric airfoil with two 0
            else: sym = 2       #% Comprovation of symetric airfoil with one 0
        #%----------------------- CAMBER ---------------------------------------
        for i in range(s):
            if x[i] < p:
                y_c[i] = m*x[i]/p**2*(2*p-x[i])+(1./2-x[i])*sin(alpha)
                dyc_dx[i]=2*m/p**2*(p-x[i])/cos(alpha)-tan(alpha)
            else:
                y_c[i] = m*(1-x[i])/(1-p)**2*(1+x[i]-2*p)+(1./2-x[i])*sin(alpha)
                dyc_dx[i] = 2*m/(1-p)**2*(p-x[i])/cos(alpha)-tan(alpha)
    #%----------------------- MEAN CAMBER 5 DIGIT SERIES CALCULATION -----------
    elif nc == 5:
        p = int(series[1])/20.  #% Location of maximum camber (2nd digit)
        rn = int(series[2])     #% Type of camber (3rd digit)
        # %----------------------- STANDARD CAMBER ------------------------------
        if rn == 0: 
            #% R constant calculation by interpolation
            r=3.33333333333212*p**3+0.700000000000909*p**2+1.19666666666638*p-0.00399999999996247
            #% K1 constant calculation by interpolation
            k1=1514933.33335235*p**4-1087744.00001147*p**3+286455.266669048*p**2-32968.4700001967*p+1420.18500000524
            #%----------------------- CAMBER -----------------------------------
            for i in range(s):
                if x[i]<r:
                    y_c[i] = k1/6*(x[i]**3-3*r*x[i]**2+r**2*(3-r)*x[i])+(1./2-x[i])*sin(alpha)
                    dyc_dx[i] = k1/6*(3*x[i]**2-6*r*x[i]+r**2*(3-r))/cos(alpha)-tan(alpha)
                else:
                    y_c[i] = k1*r**3/6*(1-x[i])+(1./2-x[i])*sin(alpha)
                    dyc_dx[i] = -k1*r**3/(6*cos(alpha))-tan(alpha)
        #%----------------------- REFLEXED CAMBER ------------------------------
        elif rn == 1:
            #% R constant calculation by interpolation
            r = 10.6666666666861*p**3-2.00000000001601*p**2+1.73333333333684*p-0.0340000000002413
            #% K1 constant calculation by interpolation
            k1 = -27973.3333333385*p**3+17972.8000000027*p**2-3888.40666666711*p+289.076000000022
            #% K1/K2 constant calculation by interpolation
            k2_k1 = 85.5279999999984*p**3-34.9828000000004*p**2+4.80324000000028*p-0.21526000000003
            #%----------------------- CAMBER -----------------------------------
            for i in range(s):
                if x[i]<r:
                    y_c[i] = k1/6*((x[i]-r)**3-k2_k1*(1-r)**3*x[i]-r**3*x[i]+r**3)+(1./2-x[i])*sin(alpha)
                    dyc_dx[i] = k1/6*(3*(x[i]-r)**2-k2_k1*(1-r)**3-r**3)/cos(alpha)-tan(alpha)
                else:
                    y_c[i] = k1/6*(k2_k1*(x[i]-r)**3-k2_k1*(1-r)**3*x[i]-r**3*x[i]+r**3)+(1./2-x[i])*sin(alpha)
                    dyc_dx[i] = k1/6*(3*k2_k1*(x[i]-r)**2-k2_k1*(1-r)**3-r**3)/cos(alpha)-tan(alpha)
        else:
            print('Incorrect NACA number. Third digit must be either 0 or 1')
            
    #%----------------------- MEAN CAMBER 6 DIGIT SERIES CALCULATION -----------     
    elif nc == 6:
        #%----------------------- MEAN CAMBER 6 DIGIT SERIES CALCULATION -----------
        #%----------------------- CONSTANTS ------------------------------------
        ser = int(series[0])        #% Number of series (1st digit)
        a = int(series[1])/10.      #% Chordwise position of minimum pressure (2nd digit)
        c_li = int(series[3])/10.   #% Design lift coefficient (4th digit)
        g = -1./(1-a)*(a**2*(1./2*log(a)-1./4)+1./4)            #% G constant calculation
        h = 1./(1-a)*(1./2*(1-a)**2*log(1-a)-1./4*(1-a)**2)+g   #% H constant calculation
        if ser == 6:
            for i in range(s):
                if (x[i]==0.) or (x[i]==a): x[i] += 1e-16
                elif (x[i]==1.): x[i] -= 1e-16
                
        #%----------------------- CAMBER ---------------------------------------
            y_c = c_li/(2.*pi*(a+1))*(1./(1-a)*(1./2*(a-x)**2*log(abs(a-x))-1./2*(1-x)**2*log(1-x)+1./4*(1-x)**2-1./4*(a-x)**2)-x*log(x)+g-h*x)+(1./2-x)*sin(alpha) #% Mean camber y coordinate
            dyc_dx = -(c_li*(h+log(x)-(x/2.-a/2.+(log(1-x)*(2*x-2))/2.+(log(abs(a-x))*(2*a-2*x))/2.+(sign(a-x)*(a-x)**2)/(2*abs(a-x)))/(a-1)+1))/(2*pi*(a+1)*cos(alpha))-tan(alpha) #% Mean camber first derivative
        else:
            print('NACA 6 Series must begin with 6') #% Error in 1st digit NACA 6 Series
    else:
        print('NACA ' + series + ' Series has not been yet implemented') #% Error of non-implemented NACA Series
    
    #%----------------------- FINAL CALCULATIONS -------------------------------
    theta = arctan(dyc_dx)      #% Angle for modifying x coordinate
    x=1./2-(1./2-x)*cos(alpha)  #% X coordinate rotation
    
    #%----------------------- COORDINATE ASSIGNATION ---------------------------
    x_e=(x-y_t*sin(theta))*c   #% X extrados coordinate
    x_i=(x+y_t*sin(theta))*c   #% X intrados coordinate
    y_e=(y_c+y_t*cos(theta))*c #% Y extrados coordinate
    y_i=(y_c-y_t*cos(theta))*c #% Y intrados coordinate
    
    
    x_a = append(x_e, x_i[::-1])  #% X airfoil
    y_a = append(y_e, y_i[::-1])  #% Y airfoi

    x_cc = x*c   #%X mean camber line
    y_cc = y_c*c #%Y mean camber line
    
    #%----------------------- NACA PLOT ----------------------------------------
    #fig = plt.figure(figsize=(15, 3))
    plt.plot(x_a, y_a, 'b') #% Airfoil
    plt.plot(x_cc,y_cc,'r') #% Mean camber line plot
    plt.plot([c/2*(1-cos(alpha)),c/2*(1+cos(alpha))],[c/2*sin(alpha),-c/2*sin(alpha)],'g')  #% Chord line plot
    plt.title('NACA ' + series + r' (%.1f' % alpha_deg  + u'\u00b0)')
    plt.legend(['Airfoil', 'Mean camber line', 'Chord line'])
    plt.xlabel('x')
    plt.xlabel('y')
    plt.axis('equal')    
    plt.xlim([-0.1, 1.1])
    plt.grid(True)
    plt.show()

    #%---------------------- Export coordinate ----------------------------------
    x_s = append(x_e, x_i) 
    y_s = append(y_e, y_i)
    np.savetxt('naca_'+series+'_geo.dat', zip(x_s, y_s))
    
def main():
    import os
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    parser = ArgumentParser( \
        formatter_class = RawDescriptionHelpFormatter, \
        description = dedent('''\
            Script to create NACA4, NACA5, NACA6 profiles
            If no argument is provided, a demo is displayed.
            '''), \
        epilog = dedent('''\
            Examples:
                Generate points for NACA profile 2412:
                    python {0} -p 2412 -a 2.5 -c 0.91 -n 200
                Ouput file:
                    naca_2412_geo.dat
            '''.format(os.path.basename(__file__))))
    parser.add_argument('-p','--profile', type = str,\
        help = 'NACA number (4, 5 or 6 digits). Example: "0012"')
    parser.add_argument('-a','--alpha', type = float, default = 0.0,\
        help = 'Angle of attack (º) (0º default)')
    parser.add_argument('-c','--chord', type = float, default = 1.0,\
        help = 'Chord of airfoil (m) (1 m default)')
    parser.add_argument('-n','--nbPoints', type = int, default = 200,\
        help = 'Number of points used to discretize chord (200 default)')
    parser.add_argument('-s','--cosine_spacing', type = int, default = 1,\
        help = 'Linear or cosine spacing (0 or 1 respectively) (1 default). '\
               'This option is recommended to have a smooth leading edge.')
    parser.add_argument('-o','--opened_edge', type = int, default = 0,\
        help = 'Opened or closed trailing edge (0 or 1 respectively) (0 default)')

    args = parser.parse_args()
    if args.profile is None:
        NACA("0012") #Demo
    else:
        NACA(args.profile, args.alpha, args.chord, args.nbPoints, args.cosine_spacing, args.opened_edge)
    
if __name__ == "__main__":
    main()        
        
