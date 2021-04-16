from elastodynamics_vel import elasto

Lx = 100
Ly = 100
Lz = 5
dx = 2.5     #m
dy = 2.5      #m
dz = 2.5      #m
dt = 0.2        #s
Eps_yy= 0.0001
Eps_xx= 0.0001     #Abs deformations at sub
directory = "/home/heisenberg/Desktop/НИР/ELASTIC/modeling3D"

c11 = 5   #Pa
c12 = 1   #Pa
c44 = 1   #Pa
rho = 1   #Pa
Alpha = 0 #Damping

el_calc = elasto()
el_calc.init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy)

time = 10**8
count = 0
for t in range(time):

    el_calc.dynamics()

    if t % 10000 == 0:

        count += 1

        el_calc.get_eps()
        el_calc.save_data(dir = directory, count = count)

        print(count)
