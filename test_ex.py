from elastodynamics_vel import elasto

Lx = 100
Ly = 100
Lz = 40
dx = 1         #nm
dy = 1        #nm
dz = 1           #nm
dt = 10**4       #10^-18
Eps_yy= 0.05
Eps_xx= 0.05
Eps_zz= 0.0      #Abs deformations at sub
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 259 * 10**9  #Pa
c12 = 154 * 10**9  #Pa
c44 = 131 * 10**9  #Pa
rho = 8290         #kg/m3
Alpha = 10**15     #Damping
MAG = 1


el_calc = elasto()
el_calc.init_boundaries(R_BD = 'Free', L_BD = 'Free',
                        B_BD = 'Free', F_BD = 'Free',
                        U_BD = 'Free',
                        D_BD = [c11 * Eps_xx + c12 * (Eps_yy + Eps_zz),
                                c11 * Eps_yy + c12 * (Eps_xx + Eps_zz),
                                c11 * Eps_zz + c12 * (Eps_yy + Eps_xx),
                                0,0,0])
el_calc.init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,MAG)


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
