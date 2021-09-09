from main2 import ME
import time as Time
import numpy as np


Lx = 100
Ly = 100
Lz = 10
dx = 1.5           #nm
dy = 1.5           #nm
dz = 0.3            #nm
dt = 5*10**3       #10^-18
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"


calc = ME()
MCP = {'MgO':[0,Lz/2],'CoFe':[Lz/2+1,Lz/2+2],'Pt':  [Lz/2+3,Lz-1]}
CoFe = {'CoFe':[0,Lz-1]}
calc.init_structure(MCP,Lx,Ly,Lz,dx,dy,dz,dt,static=True)

time = 2*10**9
every_print=10**3
count = 0
eps = 0.01

calc.init_el_BC(R_BD = 'Free',
                L_BD = 'Free',
                B_BD = 'Free',
                F_BD = 'Free',
                U_BD = 'Free',
                D_BD = [eps,eps,0])

calc.compile()

for t in range(time):

    calc.dynamics()

    if t % every_print == 0:

        count += every_print/1000
        screen = str(round(count)) + "k; time = " + str(round(t*dt*10**-9,5)) + " ns; eps = " + str(eps*100) + '%;'
        print(Time.asctime(), screen)


        layer = int(Lz/2)
        calc.plot_xy_pl('eps_xx', layer, directory + "/film/epsxx" + screen  +"_layer= " + str(layer) + ".png")
        calc.plot_xy_pl('eps_yy', layer, directory + "/film/epsyy" + screen  +"_layer= " + str(layer) + ".png")
        calc.plot_xy_pl('eps_zz', layer, directory + "/film/epszz" + screen  +"_layer= " + str(layer) + ".png")
        calc.plot_1D_xy(directory + "/film/1D" + screen, [int(Lx/2),int(Ly/2)], layer)


        calc.plot_1D_z(directory + "/film/1D" + screen  +"1D.png")
        #calc.save_state(directory + "/init/state")


        ex = np.reshape(calc.eps_xx,(Lz,Ly,Lx))[0,int(Ly/2),int(Lx/2)]
        ey = np.reshape(calc.eps_yy,(Lz,Ly,Lx))[0,int(Ly/2),int(Lx/2)]
        print(ex,ey,eps/ex)
