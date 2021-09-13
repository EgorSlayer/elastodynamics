from main import ME
import time as Time
import numpy as np


Lx = 10
Ly = 10
Lz = 800
dx = 5           #nm
dy = 5           #nm
dz = 5            #nm
dt = 10**3       #10^-18
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"


calc = ME()
MCP = {'MgO':[0,Lz/2],'CoFe':[Lz/2+1,Lz/2+2],'Pt':  [Lz/2+3,Lz-1]}
CoFe = {'CoFe':[0,Lz-1]}
MC = {'MgO':[0,Lz/2],'CoFe':[Lz/2+1,Lz-1]}
calc.init_structure(MC,Lx,Ly,Lz,dx,dy,dz,dt)

time = 2*10**9
every_print=10**4
count = 0
eps = 0.01

calc.init_el_BC(R_BD = 'Open',
                L_BD = 'Open',
                B_BD = 'Open',
                F_BD = 'Open',
                U_BD = 'Open',
                D_BD = 'Impulse')

calc.compile()

for t in range(time):

    calc.dynamics()

    if t % every_print == 0:


        screen = str(round(count)) + "k; time = " + str(round(t*dt*10**-9,5)) + " ns; eps = " + str(eps*100) + '%;'
        print(Time.asctime(), screen)


        layer = int(Lz/2)


        calc.plot_1D_z(directory + "/film/1D" + screen  +"1D.png")

        calc.save_eps(directory, count)


        ex = np.reshape(calc.eps_xx,(Lz,Ly,Lx))[int(Lz/2),int(Ly/2),int(Lx/2)]
        ey = np.reshape(calc.eps_yy,(Lz,Ly,Lx))[int(Lz/2),int(Ly/2),int(Lx/2)]
        print(ex,ey,eps/ex)
        count += int(every_print/1000)
