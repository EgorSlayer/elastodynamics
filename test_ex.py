from elastodynamics_vel import elasto

Lx = 100
Ly = 100
Lz = 40
dx = 3.5         #nm
dy = 3.5         #nm
dz = 0.25        #nm
dt = 10**3       #10^-18
Eps_yy= 0.02
Eps_xx= 0.02     #Abs deformations at sub
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 10  #GPa
c12 = 1   #GPa
c44 = 1   #GPa
rho = 4000         #kg/m3
Alpha = 0    #Damping
MAG = 0
el_calc = elasto()
el_calc.init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy,MAG)

time = 10**8
every_print=10**4
count = 0
for t in range(time):

    el_calc.dynamics()

    if t % every_print == 0:

        count += 1
        layer = 0
        el_calc.plot_xy_pl(layer, directory + "/film/" + "u" + str(count)  +" layer = " + str(layer) + ".png")
        el_calc.plot_1D_z(directory+"/film/z_strains" + str(count) + ".png")

        el_calc.save_eps(dir = directory, count = count)

        print(count)
