from main_structure import elasto

Lx = 100
Ly = 100
Lz = 10
dx = 10           #nm
dy = 10           #nm
dz = 1           #nm
dt = 10**4       #10^-18
directory = "/home/neo/Desktop/НИР/ELASTIC/modeling3D"
Alpha = 10**16     #Damping
MAG = 0


el_calc = elasto()

el_calc.init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,MAG)
'''
el_calc.init_structure({'MgO':[0,Lz/2-1],
                        'CoFe':[Lz/2,Lz-1]})
'''
el_calc.init_structure({'CoFe':[0,Lz-1]})


el_calc.init_boundaries(R_BD = 'Free',
                        L_BD = 'Free',
                        B_BD = 'Free',
                        F_BD = 'Free',
                        U_BD = 'Free',
                        D_BD = [0.01,0.01,0])
el_calc.compile()

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
            #el_calc.plot_xy_pl('eps_xx', layer, directory + "/film/" + str(count)  +"_layer= " + str(layer) + "_eps_xx.png")
            #el_calc.plot_xy_pl('eps_yy', layer, directory + "/film/" + str(count)  +"_layer= " + str(layer) + "_eps_yy.png")
            #el_calc.plot_1D_xy(directory + "/film/" + "u" + str(count)  +" layer = " + str(layer), point=[int(Lx/2),int(Ly/2)], layer=layer)


        if Lz > 1:
            el_calc.plot_1D_z(directory+"/film/z_strains" + str(count) + ".png")

        el_calc.save_eps(dir = directory, count = count)

        print('как же заебало считать в '+ str(count)+ ' раз')
