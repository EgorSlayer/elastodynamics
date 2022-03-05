from main import ME
import time as Time
import numpy as np

Lx = 100
Ly = 100
Lz = 32
dx = 3.0            #nm
dy = 3.0           #nm
dz = 1.0            #nm
dt = 10**4       #10^-18


calc = ME()
calc.directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D/"
MCP = {'MgO':[0,Lz/2],'CoFe':[Lz/2+1,Lz/2+1],'Pt':  [Lz/2+2,Lz-1]}
MC  = {'MgO':[0,Lz-3],'CoFe':[Lz-2,Lz-1]}
MC  = {'MgO':[0,Lz/2],'CoFe':[Lz/2+1,Lz-1]}
#MC  = {'CoFe':[0,Lz/2],'MgO':[Lz/2+1,Lz-1]}
CoFe = {'CoFe':[0,Lz-1]}
state = f'one_skr_longHZ1{Lx},{Ly},{Lz},{dx},{dy},{dz}'
calc.init_structure(MC,Lx,Ly,Lz,dx,dy,dz,dt,static=True)

time = 10**8
every_print=10**4


#calc.load_state(f'/home/heisenberg/Desktop/НИР/ME/init/{state}')
#calc.load_state(f'/home/heisenberg/Desktop/НИР/ME/TXT/{state}0.9975ns')
calc.init_el_BC(R_BD = 'Free',
                L_BD = 'Free',
                B_BD = 'Free',
                F_BD = 'Free',
                U_BD = 'Free',
                D_BD = 'Fixed')


calc.compile()
#count = 1560
count = 0
for t in range(0, time, 1):

    calc.dynamics()

    if t % every_print == 0:
        time_count = str(round(t*dt*10**-9,5))

        screen = str(round(count)) + "k; time = " + time_count + " ns;"
        print(Time.asctime(), screen)

        layer = int(Lz/2+1)

        calc.plot_xy_pl('eps_xx', layer, count)
        calc.plot_xy_pl('eps_yy', layer, count)
        calc.plot_xy_pl('eps_zz', layer, count)
        calc.plot_1D_xy(count, [int(Lx/2),int(Ly/2)], layer)

        #calc.save_math(directory, count)
        calc.plot_1D_z(screen)
        #calc.save_state(f"/TXT/{state}{time_count}ns")
        #calc.save_state(f"/init/{state}")

        #calc.time_series(t*2*dt*10**-9)

        count += every_print/1000
