from elastodynamics import dynamics, init

Lx = 100
Ly = 100
Lz = 3
dx = 2.5
dy = 2.5
dz = 2.5
dt = 1.3
Eps_xx= 0.001
Eps_yy= 0.001
L = Lx * Ly * Lz
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 2
c12 = 1
c44 = 1
rho = 1
Alpha = 200

data, consts = init(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy)

time = 10**8
count = 0
for t in range(time):



    loc_out = dynamics(data, consts)

    data = loc_out[0]

    eps_xx = loc_out[1]
    eps_yy = loc_out[2]
    eps_zz = loc_out[3]
    eps_xy = loc_out[4]
    eps_yz = loc_out[5]
    eps_xz = loc_out[6]

    if t % 1000 == 0:
        count += 1

        eps_xx.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xx.dat')
        eps_yy.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yy.dat')
        eps_zz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_zz.dat')
        eps_xy.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xy.dat')
        eps_yz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yz.dat')
        eps_xz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xz.dat')

        print(count, "k", "time =", t * dt,"s; dt =", dt)
