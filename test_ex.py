from elastodynamics import dynamics, init_data,  plot_xy_pl, plot_1D_z

Lx = 100
Ly = 100
Lz = 3
dx = 2.5 #nm
dy = 2.5 #nm
dz = 2.5 #nm
dt = 0.0005 #ns
Eps_xx= 0.001
Eps_yy= 0.001
L = Lx * Ly * Lz
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 20 * 10**9
c12 = 1 * 10**9
c44 = 1 * 10**9
rho = 4000
Alpha = 2 * 10**14

data, consts = init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy)

time = 10**8
count = 0
for t in range(time):


    loc_out = dynamics(data, consts)

    data = loc_out[0]
    eps = loc_out[1]

    if t % 1000 == 0:
        count += 1

        '''
        eps['eps_xx'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xx.dat')
        eps['eps_yy'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yy.dat')
        eps['eps_zz'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_zz.dat')
        eps['eps_xy'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xy.dat')
        eps['eps_yz'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yz.dat')
        eps['eps_xz'].astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xz.dat')
        '''

        for layer in range(Lz):
            plot_xy_pl(eps['eps_zz'], layer, directory, count, consts)
        plot_1D_z(eps,directory,count, consts)


        print(count, "k", "time =", t * dt, 'ns')
