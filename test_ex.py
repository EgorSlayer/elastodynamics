from main import dynamics, init_data

Lx = 100
Ly = 100
Lz = 3
dx = 2.5
dy = 2.5
dz = 2.5
dt = 2 # 10**-12
Eps_xx= 0.0001
Eps_yy= 0.0001
L = Lx * Ly * Lz
directory = "your_dir"


u1, u2, u3, v1, v2, v3, out_el = init_data(L)


time = 10**8
count = 0
for t in range(time):



    loc_out = dynamics(u1,u2,u3,v1,v2,v3,Lx,Ly,Lz,L,dx,dy,dz,dt,Eps_xx,Eps_yy)

    u1 = loc_out[0]
    u2 = loc_out[1]
    u3 = loc_out[2]
    v1 = loc_out[3]
    v2 = loc_out[4]
    v3 = loc_out[5]
    eps_xx = loc_out[6]
    eps_yy = loc_out[7]
    eps_zz = loc_out[8]
    eps_xy = loc_out[9]
    eps_yz = loc_out[10]
    eps_xz = loc_out[11]

    if t % 1 == 0:
        count += 1

        u1.astype('float32').tofile(directory + '/TXT/' + str(count) +'u1.dat')
        u2.astype('float32').tofile(directory + '/TXT/' + str(count) +'u2.dat')
        u3.astype('float32').tofile(directory + '/TXT/' + str(count) +'u3.dat')
        eps_xx.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xx.dat')
        eps_yy.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yy.dat')
        eps_zz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_zz.dat')
        eps_xy.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xy.dat')
        eps_yz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_yz.dat')
        eps_xz.astype('float32').tofile(directory + '/TXT/' + str(count) +'eps_xz.dat')

        print(count, "k", "time =", t * dt *10**-3, "ns; dt =", dt * 10**-12)
