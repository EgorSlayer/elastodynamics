import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class elasto:

    def __init__(self):
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)


    def init_boundaries(self, R_BD, L_BD, B_BD, F_BD, U_BD, D_BD):

        self.R_BD = R_BD
        self.L_BD = L_BD
        self.B_BD = B_BD
        self.F_BD = F_BD
        self.U_BD = U_BD
        self.D_BD = D_BD

    def init_structure(self, interfaces):

        for material in interfaces:
            self.coord_syst += '''
            float c11 = NAN;
            float c12 = NAN;
            float c44 = NAN;
            float rho = NAN;

            float B1 = 0;
            float B2 = 0;
            ''' + 'if ((z >= ' + str(interfaces[material][0]) + ') && (z <= ' + str(interfaces[material][1]) + ''')) {
            c11 = c11'''+str(material)+''';
            c12 = c12'''+str(material)+''';
            c44 = c44'''+str(material)+''';
            rho = rho'''+str(material)+''';

            B1 =  B1'''+str(material)+''';
            B2 =  B2'''+str(material)+''';
            };
            '''

            self.bd_code_str +='if ((z == '+str(interfaces[material][0]) + ''') && !z_bd) {
            dvxdz = ((eps_m+1)*vx[i]-vx[i])/dz;
            dvydz = ((eps_m+1)*vy[i]-vy[i])/dz;
            };
            '''

            self.bd_code_str +='if ((z == '+str(interfaces[material][1]) + ''') && !z_bd) {
            dvxdz = (vx[i]-vx[i]/(eps_m+1))/dz;
            dvydz = (vy[i]-vy[i]/(eps_m+1))/dz;
            };
            '''

    def init_data(self,Lx,Ly,Lz,dx,dy,dz,dt,Alpha,MAG):

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.L  = Lx * Ly * Lz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Alpha = Alpha
        self.MAG = MAG

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

        const float dt = '''+str(self.dt)+'''e-18;
        const float dx = '''+str(self.dx)+'''e-9;
        const float dy = '''+str(self.dy)+'''e-9;
        const float dz = '''+str(self.dz)+'''e-9;
        const float dt2 = native_powr(dt, 2);
        const float dx2 = native_powr(dx, 2);
        const float dy2 = native_powr(dy, 2);
        const float dz2 = native_powr(dz, 2);

        bool pochti_up_bd    = z == lz-2;
        bool pochti_down_bd  = z == 1;
        bool pochti_right_bd = x == lx-2;
        bool pochti_left_bd  = x == 1;
        bool pochti_back_bd  = y == ly-2;
        bool pochti_front_bd = y == 1;
        bool pochti_x_bd = pochti_right_bd || pochti_left_bd;
        bool pochti_y_bd = pochti_back_bd  || pochti_front_bd;
        bool pochti_z_bd = pochti_up_bd    || pochti_down_bd;

        int l=i-1;
        int r=i+1;
        int f=i-lx;
        int b=i+lx;
        int u=i+pl;
        int d=i-pl;

        const float Alpha =    '''+str(self.Alpha)+''';

        const float small = 0;
        const float big = 1/small;
        const float rho_vac = 1;

        float sigmaxx_bd =NAN;
        float sigmayy_bd =NAN;
        float sigmazz_bd =NAN;
        float sigmaxz_bd =NAN;
        float sigmaxy_bd =NAN;
        float sigmayz_bd =NAN;

        float vx_bd =NAN;
        float vy_bd =NAN;
        float vz_bd =NAN;

        const float c11CoFe = 259e9;
        const float c12CoFe = 154e9;
        const float c44CoFe = 131e9;
        const float rhoCoFe = 8290;

        const float B1CoFe =  -2e6;
        const float B2CoFe =  -1.5e6;

        const float c11MgO = 273e9;
        const float c12MgO = 91e9;
        const float c44MgO = 141e9;
        const float rhoMgO = 3470;

        const float B1MgO = 0;
        const float B2MgO = 0;
        '''

        # boundaries for vel update

        self.bd_code_vel = '''
        if (up_bd) {
        dsigmazzdz = (-sigma_zz[i])/dz;
        };
        if (down_bd) {
        dsigmaxzdz = (sigma_xz[i])/dz;
        dsigmayzdz = (sigma_yz[i])/dz;
        };
        if (right_bd) {
        dsigmaxxdx = (-sigma_xx[i])/dx;
        };
        if (left_bd) {
        dsigmaxydx = (sigma_xy[i])/dx;
        dsigmaxzdx = (sigma_xz[i])/dx;
        };
        if (back_bd) {
        dsigmayydy = (-sigma_yy[i])/dy;
        };
        if (front_bd) {
        dsigmaxydy = (sigma_xy[i])/dy;
        dsigmayzdy = (sigma_yz[i])/dy;
        };
        '''

        # boundaries for strain update

        self.bd_code_str = '''

        if (up_bd) {
        dvxdz = (small-vx[i])/dz;
        dvydz = (small-vy[i])/dz;
        };
        if (down_bd) {
        dvzdz = (vz[i]-small)/dz;
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

        '''

        # upper bound managment


        if self.U_BD == 'Free':

            self.bd_code_vel += '''

            '''
            self.bd_code_str += '''
            if (up_bd && lz>1) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;




            sigmazz_bd = 0;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''
        elif self.U_BD == 'z_stressed':

            self.bd_code_vel += '''
            if (up_bd && lz>1) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (up_bd && lz>1) {
            dvzdz = 0;
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 1e7;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        # lower bound managment

        if self.D_BD == 'Free':

            self.bd_code_vel += '''

            '''
            self.bd_code_str += '''
            if (down_bd) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 0;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''
        elif self.D_BD == 'z_stressed':

            self.bd_code_vel += '''
            if (down_bd && lz>1) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (down_bd && lz>1) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 1e7;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''
        elif self.D_BD == 'substrate':

            self.bd_code_vel += '''
            if (down_bd && lz>1) {
            };
            '''
            self.bd_code_str += '''
            if (down_bd && lz>1) {

            sigmaxx_bd = c11 * 0.01 + c12 * 0.01;
            sigmayy_bd = sigmaxx_bd;
            sigmazz_bd = c12 * 0.02;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        elif self.D_BD == 'Andreys substrate':

            self.bd_code_vel += '''
            if (down_bd && lz>1) {
            };
            '''
            self.bd_code_str += '''
            if (down_bd && lz>1) {
            dvxdz = ((0.01 + 1)*vx[i] - vx[u])/dz;
            dvydz = ((0.01 + 1)*vy[i] - vy[u])/dz;

            sigmaxx_bd = c11 * 0.01 + c12 * 0.01;
            sigmayy_bd = sigmaxx_bd;
            sigmazz_bd = c12 * 0.02;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };


            '''

        elif self.D_BD == 'Fixed':

            self.bd_code_vel += '''
            if (down_bd && lz>1) {
            vx_bd = 0;
            vy_bd = 0;
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (down_bd && lz>1) {

            sigmaxx_bd = 0;
            sigmayy_bd = 0;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        # right bound managment

        if self.R_BD == 'Free':

            self.bd_code_vel += '''
            '''
            self.bd_code_str += '''
            if (right_bd) {
            dvxdx = -c12/c11*(dvydy+dvzdz);
            dvydx = -dvxdy;
            dvzdx = -dvxdz;
            };
            if (right_bd) {
            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        # left bound managment

        if self.L_BD == 'Free':

            self.bd_code_vel += '''

            '''
            self.bd_code_str += '''
            if (left_bd) {
            dvxdx = -c12/c11*(dvydy+dvzdz);
            dvydx = -dvxdy;
            dvzdx = -dvxdz;
            };
            if (left_bd) {
            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        # back bound managment

        if self.B_BD == 'Free':

            self.bd_code_vel += '''
            '''
            self.bd_code_str += '''
            if (back_bd) {
            dvydy = -c12/c11*(dvxdx+dvzdz);
            dvxdy = -dvydx;
            dvzdy = -dvzdz;
            };
            if (back_bd) {
            sigmayy_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        # front bound managment

        if self.F_BD == 'Free':

            self.bd_code_vel += '''
            '''
            self.bd_code_str += '''
            if (front_bd) {
            dvydy = -c12/c11*(dvxdx+dvzdz);
            dvxdy = -dvydx;
            dvzdy = -dvzdz;
            };
            if (front_bd) {
            sigmayy_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        self.bd_code_str += '''
        if (x_bd && y_bd) {
        dvxdx = -c12/(c11+c12)*(dvzdz);
        dvydy = -c12/(c11+c12)*(dvzdz);
        dvydx = 0;
        dvzdx = 0;
        dvxdy = 0;
        dvzdy = 0;
        };

        if (x_bd && z_bd && lz > 1) {
        dvxdx = -c12/(c11+c12)*(dvydy);
        dvzdz = -c12/(c11+c12)*(dvydy);
        dvydx = 0;
        dvzdx = 0;
        dvxdz = 0;
        dvydz = 0;
        };

        if (y_bd && z_bd && lz > 1) {
        dvydy = -c12/(c11+c12)*(dvxdx);
        dvzdz = -c12/(c11+c12)*(dvxdx);
        dvxdy = 0;
        dvzdy = 0;
        dvxdz = 0;
        dvydz = 0;
        };

        if (x_bd && y_bd && z_bd) {
        dvxdx = 0;
        dvydy = 0;
        dvzdz = 0;

        dvxdy = 0;
        dvzdy = 0;
        dvxdz = 0;
        dvydz = 0;
        dvzdx = 0;
        dvydx = 0;
        };
        '''

        # Magnetostriction

        self.mgstr_code = '''
        float dmxmxdx = (mx[r]*mx[r] - mx[i]*mx[i])/(dx);
        float dmymydy = (my[b]*my[b] - my[i]*my[i])/(dy);
        float dmzmzdz = (mz[u]*mz[u] - mz[i]*mz[i])/(dz);

        float dmxmydy = (mx[i]*my[i] - mx[f]*my[f])/(dy);
        float dmxmzdz = (mx[i]*mz[i] - mx[d]*mz[d])/(dz);
        float dmxmydx = (mx[i]*my[i] - mx[l]*my[l])/(dx);
        float dmymzdz = (my[i]*mz[i] - my[d]*mz[d])/(dz);
        float dmxmzdx = (mx[i]*mz[i] - mx[l]*mz[l])/(dx);
        float dmymzdy = (my[i]*mz[i] - my[f]*mz[f])/(dy);

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
        '''



        # OpenCL elastic

        self.code_el = '''

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

            ''' + self.bd_code_vel + '''

            float fx = 0;
            float fy = 0;
            float fz = 0;

            ''' + self.mgstr_code + '''

            float new_vx = vx[i] + (dsigmaxxdx + dsigmaxydy + dsigmaxzdz - Alpha * vx[i] + MEx + fx)*dt/reciep_rho;
            float new_vy = vy[i] + (dsigmaxydx + dsigmayydy + dsigmayzdz - Alpha * vy[i] + MEy + fy)*dt/reciep_rho;
            float new_vz = vz[i] + (dsigmaxzdx + dsigmayzdy + dsigmazzdz - Alpha * vz[i] + MEz + fz)*dt/reciep_rho;

            if (!isnan(vx_bd)) {new_vx = vx_bd;}
            if (!isnan(vy_bd)) {new_vy = vy_bd;}
            if (!isnan(vz_bd)) {new_vz = vz_bd;}

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
            float dvydy = (vy[i]-vy[f])/dy;
            float dvzdz = (vz[i]-vz[d])/dz;

            float dvydx = (vy[r]-vy[i])/dx;
            float dvzdx = (vz[r]-vz[i])/dx;

            float dvxdy = (vx[b]-vx[i])/dy;
            float dvzdy = (vz[b]-vz[i])/dy;

            float dvxdz = (vx[u]-vx[i])/dz;
            float dvydz = (vy[u]-vy[i])/dz;

            ''' + self.bd_code_str + '''

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

            if (!isnan(sigmaxx_bd)) {new_sigmaxx = sigmaxx_bd;}
            if (!isnan(sigmayy_bd)) {new_sigmayy = sigmayy_bd;}
            if (!isnan(sigmazz_bd)) {new_sigmazz = sigmazz_bd;}
            if (!isnan(sigmaxy_bd)) {new_sigmaxy = sigmaxy_bd;}
            if (!isnan(sigmayz_bd)) {new_sigmayz = sigmayz_bd;}
            if (!isnan(sigmaxz_bd)) {new_sigmaxz = sigmaxz_bd;}

            barrier(CLK_GLOBAL_MEM_FENCE);

            sigma_xx[i] = new_sigmaxx;
            sigma_yy[i] = new_sigmayy;
            sigma_zz[i] = new_sigmazz;
            sigma_xy[i] = new_sigmaxy;
            sigma_yz[i] = new_sigmayz;
            sigma_xz[i] = new_sigmaxz;

            float epsxx = ((c11 + c12)*sigma_xx[i]-c12*(sigma_zz[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));
            float epsyy = ((c11 + c12)*sigma_yy[i]-c12*(sigma_xx[i] + sigma_zz[i]))/((c11 - c12)*(c11 + 2*c12));
            float epszz = ((c11 + c12)*sigma_zz[i]-c12*(sigma_xx[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));

            eps_xx[i] = epsxx;
            eps_yy[i] = epsyy;
            eps_zz[i] = epszz;
            eps_xy[i] = 0;
            eps_yz[i] = 0;
            eps_xz[i] = 0;
        };
        '''


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

        if (self.Lx == 100 and self.Ly == 100) and self.MAG == 1:
            load = np.load("/home/heisenberg/Desktop/НИР/FM/math/CoFe_DMI/init/helical2100100.npy",allow_pickle=True)
            load_mag = load[0]
            Lmx = load_mag[0]
            Lmy = load_mag[1]
            Lmz = load_mag[2]
        else:
            Lmx = np.zeros(self.L).astype(np.float32)
            Lmy = np.zeros(self.L).astype(np.float32)
            Lmz = np.zeros(self.L).astype(np.float32)


        self.mx = np.zeros(self.L).astype(np.float32)
        self.my = np.zeros(self.L).astype(np.float32)
        self.mz = np.zeros(self.L).astype(np.float32)

        for z in range(Lz):
            for y in range(Ly):
                for x in range(Lx):
                    i = z * Lx * Ly + y * Lx + x

                    if z == int(Lz/2):
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

    def compile(self):
        # build the Kernel
        self.prog = cl.Program(self.ctx, self.code_el).build()

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


    def plot_xy_pl(self, stringer, layer, file_name):

        self.get_eps()

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dx = self.dx
        dy = self.dy

        Xpos, Ypos = np.meshgrid(np.arange(0, dx * Lx, dx), np.arange(0, dy * Ly, dy))

        if stringer == 'eps_xx':
            a = self.eps_xx
            lab = "$\epsilon_{xx}$"
        elif stringer == 'eps_yy':
            a = self.eps_yy
            lab = "$\epsilon_{yy}$"
        elif stringer == 'eps_zz':
            a = self.eps_zz
            lab = "$\epsilon_{zz}$"

        pl = np.reshape(a, (Lz, Ly, Lx))[layer]
        fig, ax = plt.subplots()
        plt.contourf(Xpos, Ypos, pl, cmap=plt.get_cmap('plasma'),
                     levels=MaxNLocator(nbins=100).tick_values(pl.min(), pl.max()))
        plt.colorbar(label=lab, format='%.20f')
        ax.set_aspect('equal', 'box')
        plt.ticklabel_format(useOffset=False)
        plt.tick_params(labelleft=False)
        fig.set_size_inches(15, 15)
        plt.savefig(file_name, dpi=100)
        plt.close()

    def plot_1D_xy(self, file_name, point, layer):

        self.get_eps()

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dx = self.dx
        dy = self.dy
        dz = self.dz

        a1 = 100*np.reshape(self.eps_xx, (Lz, Ly, Lx))
        a2 = 100*np.reshape(self.eps_yy, (Lz, Ly, Lx))
        a3 = 100*np.reshape(self.eps_zz, (Lz, Ly, Lx))

        pl1 = []
        pl2 = []
        pl3 = []

        z = layer
        x = point[0]

        for y in range(Ly):
            pl1.append(a1[z][y][x])
            pl2.append(a2[z][y][x])
            pl3.append(a3[z][y][x])

        pl1 = np.array(pl1)
        pl2 = np.array(pl2)
        pl3 = np.array(pl3)

        fig, ax = plt.subplots()
        t = np.arange(0, dy * Ly, dy)
        ax.plot(t, pl1,"b",   label="$\epsilon_{xx}$")
        ax.plot(t, pl2,"r--", label="$\epsilon_{yy}$")
        ax.plot(t, pl3, "g",  label="$\epsilon_{zz}$")
        ax.legend(loc='lower left')
        ax.set(xlabel='y coordinate (nm)', ylabel='Mechanical strains (%)')
        ax.grid()

        plt.savefig(file_name + "y.png", dpi=100)
        plt.close()

        pl1 = []
        pl2 = []
        pl3 = []

        y = point[1]

        for x in range(Lx):
            pl1.append(a1[z][y][x])
            pl2.append(a2[z][y][x])
            pl3.append(a3[z][y][x])

        pl1 = np.array(pl1)
        pl2 = np.array(pl2)
        pl3 = np.array(pl3)

        fig, ax = plt.subplots()
        t = np.arange(0, dx * Lx, dx)
        ax.plot(t, pl1,"b",   label="$\epsilon_{xx}$")
        ax.plot(t, pl2,"r--", label="$\epsilon_{yy}$")
        ax.plot(t, pl3, "g",  label="$\epsilon_{zz}$")
        ax.legend(loc='lower left')
        ax.set(xlabel='x coordinate (nm)', ylabel='Mechanical strains (%)')
        ax.grid()

        plt.savefig(file_name + "x.png", dpi=100)
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


    def plot_1D_z(self, file_name):

        self.get_eps()

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
        for z in range(0,Lz,1):
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

        plt.savefig(file_name, dpi=100)
        plt.close()

    def save_eps(self, dir, count):
        self.get_eps()
        self.eps_xx.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xx.dat')
        self.eps_yy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yy.dat')
        self.eps_zz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_zz.dat')
        self.eps_xy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xy.dat')
        self.eps_yz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yz.dat')
        self.eps_xz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xz.dat')

    def save_vs(self, dir, count):
        self.get_vs()
        self.vx.astype('float32').tofile(dir + '/TXT/' + str(count) +'vx.dat')
        self.vy.astype('float32').tofile(dir + '/TXT/' + str(count) +'vy.dat')
        self.vz.astype('float32').tofile(dir + '/TXT/' + str(count) +'vz.dat')

'''
            float B = 1/rho;

            if ((x_bd && lx > 1) || (y_bd && ly > 1) || (up_bd && lz > 1)) {
            B=0;
            };

            if (down_bd) {
            source_xx += c11 * Eps_xx + c12 * Eps_yy;
            source_yy += c11 * Eps_yy + c12 * Eps_xx;
            source_zz += c12 * (Eps_yy+Eps_xx);
            };

           float rc44 = c44;
           float bc44 = c44;
           float uc44 = c44;

           float brc44 = c44;
           float urc44 = c44;
           float ubc44 = c44;

           if (right_bd) {
           rc44 = small;
           brc44 = small;
           urc44 = small;
           };

           if (back_bd) {
           bc44 = small;
           brc44 = small;
           ubc44 = small;
           };

           if (up_bd) {
           uc44 = small;
           urc44 = small;
           ubc44 = small;
           };

           float muxy = 4/(1/c44+1/rc44+1/bc44+1/brc44);
           float muyz = 4/(1/c44+1/uc44+1/bc44+1/ubc44);
           float muxz = 4/(1/c44+1/rc44+1/uc44+1/urc44);

           if (right_bd) {
            epsxx = dt * (vx[i]-vx[l])/dx;
            };

            if (left_bd) {
            epsxx = dt * (vx[r]-vx[i])/dx;
            };

            if (back_bd) {
            epsyy = dt * (vy[i]-vy[f])/dy;
            };

            if (front_bd) {
            epsyy = dt * (vy[b]-vy[i])/dy;
            };

            if (up_bd) {
            epszz = dt * (vz[i]-vz[d])/dz;
            };

            if (down_bd) {
            epszz = dt * (vz[u]-vz[i])/dz;
            };

            if (lx == 1) {
            epsxx = dt * vx[i] / dx;
            };

            if (ly == 1) {
            epsyy = dt * vy[i] / dy;
            };

            if (lz == 1) {
            epszz = dt * vz[i] / dz;
            };


            if (down_bd) {
            new_sigmaxx = c11 * Eps_xx + c12 * Eps_yy;
            new_sigmayy = c11 * Eps_yy + c12 * Eps_xx;
            new_sigmazz = c12 * (Eps_xx + Eps_yy);
            new_sigmaxy = 0;
            new_sigmayz = 0;
            new_sigmaxz = 0;
            }


            float epsxx = ((c11 + c12)*sigma_xx[i]-c12*(sigma_zz[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));
            float epsyy = ((c11 + c12)*sigma_yy[i]-c12*(sigma_xx[i] + sigma_zz[i]))/((c11 - c12)*(c11 + 2*c12));
            float epszz = ((c11 + c12)*sigma_zz[i]-c12*(sigma_xx[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));

            float epsxx = sigma_xx[i];
            float epsyy = sigma_yy[i];
            float epszz = sigma_zz[i];


            if (down_bd) {
            new_sigmaxx = c11 * Eps_xx + c12 * (Eps_yy + Eps_zz);
            new_sigmayy = c11 * Eps_yy + c12 * (Eps_xx + Eps_zz);
            new_sigmazz = c11 * Eps_zz + c12 * (Eps_xx + Eps_yy);
            new_sigmaxy = 0;
            new_sigmayz = 0;
            new_sigmaxz = 0;
            };

            if ((x_bd && lx > 1) || (y_bd && ly > 1) || (z_bd && lz > 1))  {
            Bx=1e-27;
            By=1e-27;
            Bz=1e-27;
            };

            if ((right_bd && lx > 1) )  {
            Bx=1e-27;
            };

            if ((back_bd && ly > 1) )  {
            By=1e-27;
            };

            if ((up_bd && lz > 1) )  {
            Bz=1e-27;
            };

            /*
                        if (up_bd) {
                        dvxdz = (small-vx[i])/dz;
                        dvydz = (small-vy[i])/dz;
                        };
                        if (down_bd) {
                        dvzdz = (vz[i]-small)/dz;
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

            */

                if (x_bd && y_bd) {
                dvxdx = -c12/(c11+c12)*(dvzdz);
                dvydy = -c12/(c11+c12)*(dvzdz);
                dvydx = 0;
                dvzdx = 0;
                dvxdy = 0;
                dvzdy = 0;
                };

                if (x_bd && up_bd) {
                dvxdx = -c12/(c11+c12)*(dvydy);
                dvzdz = -c12/(c11+c12)*(dvydy);
                dvydx = 0;
                dvzdx = 0;
                dvxdz = 0;
                dvydz = 0;
                };

                if (y_bd && up_bd) {
                dvydy = -c12/(c11+c12)*(dvxdx);
                dvzdz = -c12/(c11+c12)*(dvxdx);
                dvxdy = 0;
                dvzdy = 0;
                dvxdz = 0;
                dvydz = 0;
                };

                if (x_bd && y_bd && up_bd) {
                dvxdx = 0;
                dvydy = 0;
                dvzdz = 0;

                dvxdy = 0;
                dvzdy = 0;
                dvxdz = 0;
                dvydz = 0;
                dvzdx = 0;
                dvydx = 0;
                };

'''
