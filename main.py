import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class elasto:

    def init_data(self,Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy):

        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.L  = Lx * Ly * Lz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Eps_xx = Eps_xx
        self.Eps_yy = Eps_yy
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self.rho = rho
        self.Alpha = Alpha

        self.coord_syst = '''

        const int lx =  '''+str(self.Lx)+''';
        const int ly =  '''+str(self.Ly)+''';
        const int lz =  '''+str(self.Lz)+''';
        int pl = lx * ly;
        int x = i % lx;
        int z = i / pl;
        int y = (i - z * pl)/ lx;

        bool left_bd  = x == 0;
        bool right_bd = x == lx-1;
        bool front_bd = y == 0;
        bool back_bd  = y == ly-1;
        bool down_bd  = z == 0;
        bool up_bd    = z == lz-1;

        bool x_bd = right_bd || left_bd;
        bool y_bd = back_bd  || front_bd;
        bool z_bd = up_bd    || down_bd;

        bool pochti_up_bd    = z == lz-2;
        bool pochti_right_bd = x == lx-2;
        bool pochti_left_bd  = x == 1;
        bool pochti_back_bd  = y == ly-2;
        bool pochti_front_bd = y == 1;
        bool pochti_x_bd = pochti_right_bd || pochti_left_bd;
        bool pochti_y_bd = pochti_back_bd  || pochti_front_bd;

        const float dt = '''+str(self.dt)+''';
        const float dx = '''+str(self.dx)+''';
        const float dy = '''+str(self.dy)+''';
        const float dz = '''+str(self.dz)+''';
        const float dt2 = native_powr(dt, 2);
        const float dx2 = native_powr(dx, 2);
        const float dy2 = native_powr(dy, 2);
        const float dz2 = native_powr(dz, 2);

        int l=i-1;
        int r=i+1;
        int f=i-lx;
        int b=i+lx;
        int u=i+pl;
        int d=i-pl;

        const float Alpha =    '''+str(self.Alpha)+''';
        const float c11 =      '''+str(self.c11)+''';
        const float c12 =      '''+str(self.c12)+''';
        const float c44 =      '''+str(self.c44)+''';
        float rho =      '''+str(self.rho)+''';
        const float Eps_xx =   '''+str(self.Eps_xx)+''';
        const float Eps_yy =   '''+str(self.Eps_yy)+''';
        const float B1 =  0*-2;
        const float B2 =  0*-1.5;
        '''

        # OpenCL elastic

        self.code_el = '''

        float shift(float pos, float neg) {
        float res = (pos + neg)/2;
        return res;
        }

        __kernel void update_velocity(
        __global float *vx,       __global float *vy,       __global float *vz,
        __global float *mx,       __global float *my,       __global float *mz,
        __global float *sigma_xx, __global float *sigma_yy, __global float *sigma_zz,
        __global float *sigma_xy, __global float *sigma_yz, __global float *sigma_xz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            float dsigmaxxdx = (sigma_xx[r]-sigma_xx[i])/dx;
            float dsigmayydy = (sigma_yy[b]-sigma_yy[i])/dy;
            float dsigmazzdz = (sigma_zz[u]-sigma_zz[i])/dz;

            float dsigmaxydx = (sigma_xy[i]-sigma_xy[l])/dx;
            float dsigmaxydy = (sigma_xy[i]-sigma_xy[f])/dy;

            float dsigmaxzdx = (sigma_xz[i]-sigma_xz[l])/dx;
            float dsigmaxzdz = (sigma_xz[i]-sigma_xz[d])/dz;

            float dsigmayzdy = (sigma_yz[i]-sigma_yz[f])/dy;
            float dsigmayzdz = (sigma_yz[i]-sigma_yz[d])/dz;

            if (up_bd) {
            dsigmazzdz = (0-sigma_zz[i])/dz;

            };

            if (down_bd) {
            dsigmaxzdz = (sigma_xz[i]-sigma_xz[i])/dz;
            dsigmayzdz = (sigma_yz[i]-sigma_xz[i])/dz;
            };

            if (back_bd) {
            dsigmayydy = (0-sigma_yy[i])/dy;
            };
            if (front_bd) {
            dsigmaxydy = (sigma_xy[i]-0)/dy;
            dsigmayzdy = (sigma_yz[i]-0)/dy;
            };
            if (right_bd) {
            dsigmaxxdx = (0-sigma_xx[i])/dx;
            };
            if (left_bd) {
            dsigmaxydx = (sigma_xy[i]-0)/dx;
            dsigmaxzdx = (sigma_xz[i]-0)/dx;
            };


            float dmxmxdx = (mx[r]*mx[r] - mx[l]*mx[l])/(2*dx);
            float dmymydy = (my[b]*my[b] - my[f]*my[f])/(2*dy);
            float dmzmzdz = (mz[u]*mz[u] - mz[d]*mz[d])/(2*dz);

            float dmxmydy = (mx[b]*my[b] - mx[f]*my[f])/(2*dy);
            float dmxmzdz = (mx[u]*mz[u] - mx[d]*mz[d])/(2*dz);
            float dmxmydx = (mx[r]*my[r] - mx[l]*my[l])/(2*dx);
            float dmymzdz = (my[u]*mz[u] - my[d]*mz[d])/(2*dz);
            float dmxmzdx = (mx[r]*mz[r] - mx[l]*mz[l])/(2*dx);
            float dmymzdy = (my[b]*mz[b] - my[f]*mz[f])/(2*dy);

            if (x_bd) {
            dmxmxdx = 0;
            dmxmydx = 0;
            dmxmzdx = 0;
            };

            if (y_bd) {
            dmymydy = 0;
            dmxmydy = 0;
            dmymzdy = 0;
            };

            if (z_bd) {
            dmzmzdz = 0;
            dmxmzdz = 0;
            dmymzdz = 0;
            };

            float MEx = B1*dmxmxdx + B2*(dmxmydy + dmxmzdz);
            float MEy = B1*dmymydy + B2*(dmxmydx + dmymzdz);
            float MEz = B1*dmzmzdz + B2*(dmxmzdx + dmymzdy);


            float rho_pos = rho;
            if (right_bd || back_bd || up_bd) {rho_pos = 10e23;};

            float new_vx = vx[i] + (dsigmaxxdx + dsigmaxydy + dsigmaxzdz + MEx)*dt*2/(rho+rho_pos);
            float new_vy = vy[i] + (dsigmaxydx + dsigmayydy + dsigmayzdz + MEy)*dt*2/(rho+rho_pos);
            float new_vz = vz[i] + (dsigmaxzdx + dsigmayzdy + dsigmazzdz + MEz)*dt*2/(rho+rho_pos);

            barrier(CLK_GLOBAL_MEM_FENCE);

            vx[i] = new_vx;
            vy[i] = new_vy;
            vz[i] = new_vz;
        };


        __kernel void update_strains(
        __global float *vx,       __global float *vy,       __global float *vz,
        __global float *sigma_xx, __global float *sigma_yy, __global float *sigma_zz,
        __global float *sigma_xy, __global float *sigma_yz, __global float *sigma_xz,
        __global float *eps_xx,   __global float *eps_yy,   __global float *eps_zz,
        __global float *eps_xy,   __global float *eps_yz,   __global float *eps_xz)

        {   int i = get_global_id(0);

            ''' + self.coord_syst + '''

            float dvxdx = (vx[i]-vx[l])/dx;
            float dvydx = (vy[r]-vy[i])/dx;
            float dvzdx = (vz[r]-vz[i])/dx;

            float dvxdy = (vx[b]-vx[i])/dy;
            float dvydy = (vy[i]-vy[f])/dy;
            float dvzdy = (vz[b]-vz[i])/dy;

            float dvxdz = (vx[u]-vx[i])/dz;
            float dvydz = (vy[u]-vy[i])/dz;
            float dvzdz = (vz[i]-vz[d])/dz;

            float small = 10e-23;


            if (up_bd) {
            dvxdz = (small-vx[i])/dz;
            dvydz = (small-vy[i])/dz;
            };
            if (down_bd) {
            dvzdz = (vz[i]-0)/dz;
            };
            if (back_bd) {
            dvxdy = (small-vx[i])/dy;
            dvzdy = (small-vz[i])/dy;
            };
            if (front_bd) {
            dvydy = (vy[i]-small)/dy;
            };
            if (right_bd) {
            dvydx = (small-vy[i])/dx;
            dvzdx = (small-vz[i])/dx;
            };
            if (left_bd) {
            dvxdx = (vx[i]-small)/dx;
            };


            float dsigmaxxdt = c11 * dvxdx + c12 * (dvydy + dvzdz);
            float dsigmayydt = c11 * dvydy + c12 * (dvxdx + dvzdz);
            float dsigmazzdt = c11 * dvzdz + c12 * (dvydy + dvxdx);

            float dsigmaxydt = c44 * (dvxdy+dvydx);
            float dsigmayzdt = c44 * (dvzdy+dvydz);
            float dsigmaxzdt = c44 * (dvxdz+dvzdx);

            float new_sigmaxx = sigma_xx[i] + dsigmaxxdt * dt;
            float new_sigmayy = sigma_yy[i] + dsigmayydt * dt;
            float new_sigmazz = sigma_zz[i] + dsigmazzdt * dt;
            float new_sigmaxy = sigma_xy[i] + dsigmaxydt * dt;
            float new_sigmayz = sigma_yz[i] + dsigmayzdt * dt;
            float new_sigmaxz = sigma_xz[i] + dsigmaxzdt * dt;

            if (down_bd) {
            new_sigmaxx = c11 * Eps_xx + c12 * (Eps_yy);
            new_sigmayy = c11 * Eps_yy + c12 * (Eps_xx);
            new_sigmazz = c12 * (Eps_xx + Eps_yy);

            new_sigmaxy = 0;
            new_sigmayz = 0;
            new_sigmaxz = 0;
            };

            barrier(CLK_GLOBAL_MEM_FENCE);

            sigma_xx[i] = new_sigmaxx;
            sigma_yy[i] = new_sigmayy;
            sigma_zz[i] = new_sigmazz;
            sigma_xy[i] = new_sigmaxy;
            sigma_yz[i] = new_sigmayz;
            sigma_xz[i] = new_sigmaxz;

            eps_xx[i] = dvxdx * dt;
            eps_yy[i] = dvydy * dt;
            eps_zz[i] = dvzdz * dt;
            eps_xy[i] = (dvxdy + dvydx) / 2 * dt;
            eps_yz[i] = (dvzdy + dvydz) / 2 * dt;
            eps_xz[i] = (dvxdz + dvzdx) / 2 * dt;
        };
        '''

        # build the Kernel
        self.prog = cl.Program(self.ctx, self.code_el).build()

        self.vx = np.zeros(self.L).astype(np.float32)
        self.vy = np.zeros(self.L).astype(np.float32)
        self.vz = np.zeros(self.L).astype(np.float32)

        self.sigmaxx = np.zeros(self.L).astype(np.float32)
        self.sigmayy = np.zeros(self.L).astype(np.float32)
        self.sigmazz = np.zeros(self.L).astype(np.float32)
        self.sigmaxy = np.zeros(self.L).astype(np.float32)
        self.sigmayz = np.zeros(self.L).astype(np.float32)
        self.sigmaxz = np.zeros(self.L).astype(np.float32)

        self.eps_xx = np.zeros(self.L).astype(np.float32)
        self.eps_yy = np.zeros(self.L).astype(np.float32)
        self.eps_zz = np.zeros(self.L).astype(np.float32)
        self.eps_xy = np.zeros(self.L).astype(np.float32)
        self.eps_yz = np.zeros(self.L).astype(np.float32)
        self.eps_xz = np.zeros(self.L).astype(np.float32)

        load = np.load("/home/heisenberg/Desktop/НИР/FM/math/CoFe_DMI/init/helical2100100.npy",allow_pickle=True)
        load_mag = load[0]
        Lmx = load_mag[0]
        Lmy = load_mag[1]
        Lmz = load_mag[2]

        self.mx = np.zeros(self.L).astype(np.float32)
        self.my = np.zeros(self.L).astype(np.float32)
        self.mz = np.zeros(self.L).astype(np.float32)

        for z in range(Lz):
            for y in range(Ly):
                for x in range(Lx):
                    i = z * Lx * Ly + y * Lx + x

                    if z == 1:
                        self.mx[i] = Lmx[i - Lx*Ly * z]
                        self.my[i] = Lmy[i - Lx*Ly * z]
                        self.mz[i] = Lmz[i - Lx*Ly * z]

        mf = cl.mem_flags

        self.mx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mx)
        self.my_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.my)
        self.mz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mz)

        self.vx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.vx)
        self.vy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.vy)
        self.vz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.vz)

        self.sigmaxx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmaxx)
        self.sigmayy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmayy)
        self.sigmazz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmazz)
        self.sigmaxy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmaxy)
        self.sigmayz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmayz)
        self.sigmaxz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.sigmaxz)

        self.eps_xx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xx)
        self.eps_yy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_yy)
        self.eps_zz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_zz)
        self.eps_xy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xy)
        self.eps_yz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_yz)
        self.eps_xz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xz)


    def dynamics(self):

        launch = self.prog.update_velocity(self.queue, self.vx.shape, None,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.mx_buf,      self.my_buf,      self.mz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf)
        launch.wait()


        launch = self.prog.update_strains(self.queue, self.vx.shape, None,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf,
        self.eps_xx_buf,  self.eps_yy_buf,  self.eps_zz_buf,
        self.eps_xy_buf,  self.eps_yz_buf,  self.eps_xz_buf)
        launch.wait()


    def plot_xy_pl(self, layer, dir, count):

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dx = self.dx
        dy = self.dy

        Xpos, Ypos = np.meshgrid(np.arange(0, dx * Lx, dx), np.arange(0, dy * Ly, dy))

        pl = np.reshape(self.eps_zz, (Lz, Ly, Lx))[layer]
        fig, ax = plt.subplots()
        plt.contourf(Xpos, Ypos, pl, cmap=plt.get_cmap('plasma'),
                     levels=MaxNLocator(nbins=100).tick_values(pl.min(), pl.max()))
        plt.colorbar(label=r"uxx", format='%.20f')
        ax.set_aspect('equal', 'box')
        plt.ticklabel_format(useOffset=False)
        plt.tick_params(labelleft=False)
        fig.set_size_inches(15, 15)
        plt.savefig(dir + "/film/"
                    + "u" + str(count)  +" layer = " + str(layer) + ".png", dpi=100)
        plt.close()

    def get_vs(self):

        cl.enqueue_copy(self.queue, self.vx, self.vx_buf)
        cl.enqueue_copy(self.queue, self.vy, self.vy_buf)
        cl.enqueue_copy(self.queue, self.vz, self.vz_buf)

    def get_eps(self):

        cl.enqueue_copy(self.queue, self.eps_xx, self.eps_xx_buf)
        cl.enqueue_copy(self.queue, self.eps_yy, self.eps_yy_buf)
        cl.enqueue_copy(self.queue, self.eps_zz, self.eps_zz_buf)
        cl.enqueue_copy(self.queue, self.eps_xy, self.eps_xy_buf)
        cl.enqueue_copy(self.queue, self.eps_yz, self.eps_yz_buf)
        cl.enqueue_copy(self.queue, self.eps_xz, self.eps_xz_buf)


    def plot_1D_z(self, dir, count):

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dz = self.dz

        a1 = np.reshape(self.eps_xx, (Lz, Ly, Lx))
        a2 = np.reshape(self.eps_yy, (Lz, Ly, Lx))
        a3 = np.reshape(self.eps_zz, (Lz, Ly, Lx))

        pl1 = []
        pl2 = []
        pl3 = []

        x = int(Lx/2)
        y = int(Ly/2)
        for z in range(Lz):
            pl1.append(a1[z][y][x]*10**2)
            pl2.append(a2[z][y][x]*10**2)
            pl3.append(a3[z][y][x]*10**2)

        pl1 = np.array(pl1)
        pl2 = np.array(pl2)
        pl3 = np.array(pl3)

        fig, ax = plt.subplots()
        t = np.arange(0, dz * Lz, dz)
        ax.plot(t, pl1,"b",   label="$\epsilon_{xx}$")
        ax.plot(t, pl2,"r--", label="$\epsilon_{yy}$")
        ax.plot(t, pl3, "g",  label="$\epsilon_{zz}$")
        ax.legend(loc='lower left')
        ax.set(xlabel='z coordinate (nm)', ylabel='Mechanical strains (%)')
        ax.grid()

        plt.savefig(dir+"/film/z_strains" + str(count) + ".png", dpi=100)
        plt.close()

    def save_data(self, dir, count):
        self.eps_xx.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xx.dat')
        self.eps_yy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yy.dat')
        self.eps_zz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_zz.dat')
        self.eps_xy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xy.dat')
        self.eps_yz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yz.dat')
        self.eps_xz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xz.dat')
