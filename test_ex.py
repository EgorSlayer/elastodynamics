from elastodynamics_vel import elasto

Lx = 300
Ly = 300
Lz = 5
dx = 1         #nm
dy = 1         #nm
dz = 1         #nm
dt = 10**4       #10^-18
Eps_yy= 0.01
Eps_xx= 0.01    #Abs deformations at sub
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 259   #GPa
c12 = 154   #GPa
c44 = 131   #GPa
rho = 8290         #kg/m3
Alpha = 10**15    #Damping
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
        layer = int(Lz/2)
        if Lx > 1 and Ly > 1:
            el_calc.plot_xy_pl('eps_zz', layer, directory + "/film/" + str(count)  +"_layer= " + str(layer) + "_eps_zz.png")
            el_calc.plot_xy_pl('eps_xx', layer, directory + "/film/" + str(count)  +"_layer= " + str(layer) + "_eps_xx.png")
            el_calc.plot_xy_pl('eps_yy', layer, directory + "/film/" + str(count)  +"_layer= " + str(layer) + "_eps_yy.png")
            el_calc.plot_1D_xy(directory + "/film/" + "u" + str(count)  +" layer = " + str(layer), point=[int(Lx/2),int(Ly/2)], layer=layer)


        if Lz > 1:
            el_calc.plot_1D_z(directory+"/film/z_strains" + str(count) + ".png")

        el_calc.save_eps(dir = directory, count = count)

        print(count)
