from elastodynamics import dynamics, init_data,  plot_xy_pl, plot_1D_z, save_data #_stabilization

Lx = 100
Ly = 100
Lz = 3
dx = 2.5      #m
dy = 2.5      #m
dz = 2.5      #m
dt = 1.82     #s
Eps_xx= 0.001 #Abs deformations at sub
Eps_yy= 0.001
L = Lx * Ly * Lz
directory = "dir"

c11 = 6   #Pa
c12 = 1   #Pa
c44 = 1   #Pa
rho = 4   #Pa
Alpha = 2 #Damping

data, consts = init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy)

time = 10**8
count = 0
for t in range(time):

    loc_out = dynamics(data, consts)

    data = loc_out[0]
    eps = loc_out[1]

    if t % 1 == 0:
        count += 1
        save_data(eps, directory, count)
        for layer in range(Lz):
            plot_xy_pl(eps['eps_zz'], layer, directory, count, consts)
        plot_1D_z(eps, directory, count, consts)

        print(count, "k", "time =", t * dt, 'ns')
