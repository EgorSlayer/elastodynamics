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
        self.L =  Lx * Ly * Lz
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

        self.empty = np.zeros(self.L).astype(np.float32)

        self.coord_syst = '''

        const int len = '''+str(self.L)+''';
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
        int fl=i-lx-1;
        int fr=i-lx+1;
        int br=i+lx+1;
        int bl=i+lx-1;
        int dl=i-pl-1;
        int dr=i-pl+1;
        int ul=i+pl-1;
        int ur=i+pl+1;
        int db=i-pl+lx;
        int df=i-pl-lx;
        int uf=i+pl-lx;
        int ub=i+pl+lx;

        const float Alpha =    '''+str(self.Alpha)+''';
        const float c11 =      '''+str(self.c11)+''';
        const float c12 =      '''+str(self.c12)+''';
        const float c44 =      '''+str(self.c44)+''';
        const float rho =      '''+str(self.rho)+''';
        const float Eps_xx =   '''+str(self.Eps_xx)+''';
        const float Eps_yy =   '''+str(self.Eps_yy)+''';
        const float B1 =  -20.18e6;
        const float B2 =  -15.27e6;
        '''

        # OpenCL elastic

        self.code_el = '''

        float eps_ii(float pos, float here, float neg, float di)
            {
            float eps = (pos-here)/di;
            return eps;
            }

        float didj(float pos, float here, float neg, float dj)
            {
            float eps = (pos-neg)/(2*dj);
            return eps;
            }

        float eps_ij(float didj,float djdi)
            {
            float eps = (didj+djdi)/2;
            return eps;
            }

        float sigma_ii(float c11, float eps_ii, float c12, float eps_jj, float eps_kk)
            {
            float sigma = c11 * eps_ii + c12 * (eps_jj + eps_kk);
            return sigma;
            }

        float sigma_ij(float c44, float eps_ij)
            {
            float sigma = 2 * c44 * eps_ij;
            return sigma;
            }

        float dsigmaiidi(float pos,float here,float neg,float dj)
            {
            float eps = (pos-neg)/(2*dj);
            return eps;
            }

        float dsigmaijdj(float pos,float here,float neg,float dj)
            {
            float eps = (pos-neg)/(2*dj);
            return eps;
            }

        __kernel void get_strains(
        __global const float *u1, __global const float *u2, __global const float *u3,
        __global float *sigma_xx, __global float *sigma_yy, __global float *sigma_zz,
        __global float *sigma_xy, __global float *sigma_yz, __global float *sigma_xz,
        __global float *eps_xx,   __global float *eps_yy,   __global float *eps_zz,
        __global float *eps_xy,   __global float *eps_yz,   __global float *eps_xz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            eps_xx[i]=eps_ii(u1[r], u1[i], u1[l],dx);
            eps_yy[i]=eps_ii(u2[b], u2[i], u2[f],dy);
            eps_zz[i]=eps_ii(u3[u], u3[i], u3[d],dz);

            float dzdx =didj(u3[r], u3[i], u3[l], dx);
            float dydx =didj(u2[r], u2[i], u2[l], dx);
            float dxdy =didj(u1[b], u1[i], u1[f], dy);
            float dzdy =didj(u3[b], u3[i], u3[f], dy);
            float dxdz =didj(u1[u], u1[i], u1[d], dz);
            float dydz =didj(u2[u], u2[i], u2[d], dz);

            float dxdx=didj(u1[r], u1[i], u1[l],dx);
            float dydy=didj(u2[b], u2[i], u2[f],dy);
            float dzdz=didj(u3[u], u3[i], u3[d],dz);

            if (back_bd) {
            eps_yy[i]=(u2[i]-u2[f])/dy;
            dxdy =    (u1[i]-u1[f])/dy;
            dydy =    (u2[i]-u2[f])/dy;
            dzdy =    (u3[i],u3[f])/dy;
            };

            if (front_bd) {
            eps_yy[i]=(u2[b]-u2[i])/dy;
            dxdy =    (u1[b]-u1[i])/dy;
            dydy =    (u2[b]-u2[i])/dy;
            dzdy =    (u3[b]-u3[i])/dy;
            };

            if (right_bd) {
            eps_xx[i]=(u1[i]-u1[l])/dx;
            dxdx =    (u1[i]-u1[l])/dx;
            dzdx =    (u3[i]-u3[l])/dx;
            dydx =    (u2[i]-u2[l])/dx;
            };

            if (left_bd) {
            eps_xx[i]=(u1[r]-u1[i])/dx;
            dxdx =    (u1[r]-u1[i])/dx;
            dydx =    (u2[r]-u2[i])/dx;
            dzdx =    (u3[r]-u3[i])/dx;
            };

            if (down_bd) {
            eps_zz[i]=eps_ii(u3[u],u3[i], 0          ,dz);
            dxdz =      didj(u1[u],u1[i], Eps_xx*dx*x,dz);
            dydz =      didj(u2[u],u2[i], Eps_yy*dy*y,dz);
            dzdz =      didj(u3[u],u3[i], 0          ,dz);
            };

            if (up_bd) {
            eps_zz[i]=(u3[i]-u3[d])/dz;
            dxdz =    (u1[i]-u1[d])/dz;
            dydz =    (u2[i]-u2[d])/dz;
            dzdz =    (u3[i]-u3[d])/dz;
            };

            barrier(CLK_GLOBAL_MEM_FENCE);

            eps_xy[i]=eps_ij(dxdy, dydx);
            eps_yz[i]=eps_ij(dydz, dzdy);
            eps_xz[i]=eps_ij(dxdz, dzdx);

            if (x_bd) {
            sigma_xx[i] = 0;
            }
            else {
            sigma_xx[i] = sigma_ii(c11, eps_xx[i], c12, dydy, dzdz);
            };

            if (y_bd) {
            sigma_yy[i] = 0;
            }
            else {
            sigma_yy[i] = sigma_ii(c11, eps_yy[i], c12, dxdx, dzdz);
            };

            if (up_bd) {
            sigma_zz[i] = 0;
            }
            else {
            sigma_zz[i] = sigma_ii(c11, eps_zz[i], c12, dydy, dzdz);
            };

            if (x_bd || y_bd) {
            sigma_xy[i] = 0;
            }
            else{
            sigma_xy[i] = sigma_ij(c44, eps_xy[i]);
            };

            if (x_bd || up_bd) {
            sigma_xz[i] = 0;
            }
            else {
            sigma_xz[i] = sigma_ij(c44, eps_xz[i]);
            };

            if (y_bd || up_bd) {
            sigma_yz[i] = 0;
            }
            else {
            sigma_yz[i] = sigma_ij(c44, eps_yz[i]);
            };

        };



        __kernel void get_dstrains(
        __global const float *u1, __global const float *u2, __global const float *u3,
        __global float *dsigma_xx, __global float *dsigma_yy, __global float *dsigma_zz,
        __global float *dsigma_xy, __global float *dsigma_yz, __global float *dsigma_xz,
        __global float *eps_xx,   __global float *eps_yy,   __global float *eps_zz,
        __global float *eps_xy,   __global float *eps_yz,   __global float *eps_xz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            eps_xx[i]=eps_ii(u1[r], u1[i], u1[l], dx);
            eps_yy[i]=eps_ii(u2[b], u2[i], u2[f], dy);
            eps_zz[i]=eps_ii(u3[u], u3[i], u3[d], dz);

            float dzdx =didj(u3[r], u3[i], u3[l], dx);
            float dydx =didj(u2[r], u2[i], u2[l], dx);
            float dxdy =didj(u1[b], u1[i], u1[f], dy);
            float dzdy =didj(u3[b], u3[i], u3[f], dy);
            float dxdz =didj(u1[u], u1[i], u1[d], dz);
            float dydz =didj(u2[u], u2[i], u2[d], dz);

            float dxdx =didj(u1[r], u1[i], u1[l], dx);
            float dydy =didj(u2[b], u2[i], u2[f], dy);
            float dzdz =didj(u3[u], u3[i], u3[d], dz);

            if (back_bd) {
            eps_yy[i]=(u2[i]-u2[f])/dy;
            dxdy =    (u1[i]-u1[f])/dy;
            dydy =    (u2[i]-u2[f])/dy;
            dzdy =    (u3[i],u3[f])/dy;
            };

            if (front_bd) {
            eps_yy[i]=(u2[b]-u2[i])/dy;
            dxdy =    (u1[b]-u1[i])/dy;
            dydy =    (u2[b]-u2[i])/dy;
            dzdy =    (u3[b]-u3[i])/dy;
            };

            if (right_bd) {
            eps_xx[i]=(u1[i]-u1[l])/dx;
            dxdx =    (u1[i]-u1[l])/dx;
            dzdx =    (u3[i]-u3[l])/dx;
            dydx =    (u2[i]-u2[l])/dx;
            };

            if (left_bd) {
            eps_xx[i]=(u1[r]-u1[i])/dx;
            dxdx =    (u1[r]-u1[i])/dx;
            dydx =    (u2[r]-u2[i])/dx;
            dzdx =    (u3[r]-u3[i])/dx;
            };

            if (down_bd) {
            eps_zz[i]=eps_ii(u3[u],u3[i], 0          ,dz);
            dxdz =      didj(u1[u],u1[i], Eps_xx*dx*x,dz);
            dydz =      didj(u2[u],u2[i], Eps_yy*dy*y,dz);
            dzdz =      didj(u3[u],u3[i], 0          ,dz);
            };

            if (up_bd) {
            eps_zz[i]=(u3[i]-u3[d])/dz;
            dxdz =    (u1[i]-u1[d])/dz;
            dydz =    (u2[i]-u2[d])/dz;
            dzdz =    (u3[i]-u3[d])/dz;
            };

            barrier(CLK_GLOBAL_MEM_FENCE);

            eps_xy[i]=eps_ij(dxdy, dydx);
            eps_yz[i]=eps_ij(dydz, dzdy);
            eps_xz[i]=eps_ij(dxdz, dzdx);

            if (x_bd) {
            dsigma_xx[i] = 0;
            }
            else {
            dsigma_xx[i] = sigma_ii(c11, eps_xx[i], c12, dydy, dzdz);
            };

            if (y_bd) {
            dsigma_yy[i] = 0;
            }
            else {
            dsigma_yy[i] = sigma_ii(c11, eps_yy[i], c12, dxdx, dzdz);
            };

            if (up_bd) {
            dsigma_zz[i] = 0;
            }
            else {
            dsigma_zz[i] = sigma_ii(c11, eps_zz[i], c12, dydy, dzdz);
            };

            if (x_bd || y_bd) {
            dsigma_xy[i] = 0;
            }
            else{
            dsigma_xy[i] = sigma_ij(c44, eps_xy[i]);
            };

            if (x_bd || up_bd) {
            dsigma_xz[i] = 0;
            }
            else {
            dsigma_xz[i] = sigma_ij(c44, eps_xz[i]);
            };

            if (y_bd || up_bd) {
            dsigma_yz[i] = 0;
            }
            else {
            dsigma_yz[i] = sigma_ij(c44, eps_yz[i]);
            };

        };


        __kernel void get_magnetoelastic(
        __global const float *m1, __global const float *m2, __global const float *m3,
        __global float *MEx,      __global float *MEy,      __global float *MEz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            float dmxmxdx = (m1[r]*m1[r] - m1[l]*m1[l])/(2*dx);
            float dmymydy = (m2[b]*m2[b] - m2[f]*m2[f])/(2*dy);
            float dmzmzdz = (m3[u]*m3[u] - m3[d]*m3[d])/(2*dz);

            float dmxmydy = (m1[b]*m2[b] - m1[f]*m2[f])/(2*dy);
            float dmxmzdz = (m1[u]*m3[u] - m1[d]*m3[d])/(2*dz);
            float dmxmydx = (m1[r]*m2[r] - m1[l]*m2[l])/(2*dx);
            float dmymzdz = (m2[u]*m3[u] - m2[d]*m3[d])/(2*dz);
            float dmxmzdx = (m1[r]*m3[r] - m1[l]*m3[l])/(2*dx);
            float dmymzdy = (m2[b]*m3[b] - m2[f]*m3[f])/(2*dy);

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

            MEx[i] = B1*dmxmxdx + B2*(dmxmydy + dmxmzdz);
            MEy[i] = B1*dmymydy + B2*(dmxmydx + dmymzdz);
            MEz[i] = B1*dmzmzdz + B2*(dmxmzdx + dmymzdy);

        };


        __kernel void integrate(
        __global float *u1, __global float *u2, __global float *u3,
        __global float *v1, __global float *v2, __global float *v3,
        __global const float *sigma_xx,__global const float *sigma_yy,__global const float *sigma_zz,
        __global const float *sigma_xy,__global const float *sigma_yz,__global const float *sigma_xz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            float dsigmaxxdx = dsigmaiidi(sigma_xx[r],sigma_xx[i],sigma_xx[l],dx);
            float dsigmayydy = dsigmaiidi(sigma_yy[b],sigma_yy[i],sigma_yy[f],dy);
            float dsigmazzdz = dsigmaiidi(sigma_zz[u],sigma_zz[i],sigma_zz[d],dz);

            float dsigmaxydx = dsigmaijdj(sigma_xy[r],sigma_xy[i],sigma_xy[l],dx);
            float dsigmaxydy = dsigmaijdj(sigma_xy[b],sigma_xy[i],sigma_xy[f],dy);

            float dsigmaxzdx = dsigmaijdj(sigma_xz[r],sigma_xz[i],sigma_xz[l],dx);
            float dsigmaxzdz = dsigmaijdj(sigma_xz[u],sigma_xz[i],sigma_xz[d],dz);

            float dsigmayzdy = dsigmaijdj(sigma_yz[b],sigma_yz[i],sigma_yz[f],dy);
            float dsigmayzdz = dsigmaijdj(sigma_yz[u],sigma_yz[i],sigma_yz[d],dz);

            if (up_bd) {
            dsigmazzdz = (sigma_zz[i]-sigma_zz[d])/dz;
            dsigmaxzdz = (sigma_xz[i]-sigma_xz[d])/dz;
            dsigmayzdz = (sigma_yz[i]-sigma_yz[d])/dz;
            };

            if (down_bd) {
            dsigmazzdz = (sigma_zz[u] - sigma_zz[i])/dz;
            dsigmaxzdz = (sigma_xz[u] - sigma_xz[i])/dz;
            dsigmayzdz = (sigma_yz[u] - sigma_yz[i])/dz;
            };

            if (back_bd) {
            dsigmayydy = (sigma_yy[i]-sigma_yy[f])/dy;
            dsigmaxydy = (sigma_xy[i]-sigma_xy[f])/dy;
            dsigmayzdy = (sigma_yz[i]-sigma_yz[f])/dy;
            };

            if (front_bd) {
            dsigmayydy = (sigma_yy[b]-sigma_yy[i])/dy;
            dsigmaxydy = (sigma_xy[b]-sigma_xy[i])/dy;
            dsigmayzdy = (sigma_yz[b]-sigma_yz[i])/dy;
            };


            if (right_bd) {
            dsigmaxxdx = (sigma_xx[i]-sigma_xx[l])/dx;
            dsigmaxydx = (sigma_xy[i]-sigma_xy[l])/dx;
            dsigmaxzdx = (sigma_xz[i]-sigma_xz[l])/dx;
            };

            if (left_bd) {
            dsigmaxxdx = (sigma_xx[r]-sigma_xx[i])/dx;
            dsigmaxydx = (sigma_xy[r]-sigma_xy[i])/dx;
            dsigmaxzdx = (sigma_xz[r]-sigma_xz[i])/dx;
            };

            barrier(CLK_GLOBAL_MEM_FENCE);

            float U1f = (2*(dsigmaxxdx + dsigmaxydy + dsigmaxzdz)*dt2 + 4*rho*u1[i] + (Alpha*dt - 2*rho)*v1[i])/(Alpha*dt + 2*rho);
            float U2f = (2*(dsigmaxydx + dsigmayydy + dsigmayzdz)*dt2 + 4*rho*u2[i] + (Alpha*dt - 2*rho)*v2[i])/(Alpha*dt + 2*rho);
            float U3f = (2*(dsigmaxzdx + dsigmayzdy + dsigmazzdz)*dt2 + 4*rho*u3[i] + (Alpha*dt - 2*rho)*v3[i])/(Alpha*dt + 2*rho);

            barrier(CLK_GLOBAL_MEM_FENCE);

            v1[i] = u1[i];
            v2[i] = u2[i];
            v3[i] = u3[i];

            u1[i] = U1f;
            u2[i] = U2f;
            u3[i] = U3f;
        };
        '''

        # build the Kernel
        self.prog = cl.Program(self.ctx, self.code_el).build()

        self.eps_xx = self.empty
        self.eps_yy = self.empty
        self.eps_zz = self.empty
        self.eps_xy = self.empty
        self.eps_yz = self.empty
        self.eps_xz = self.empty

        mf = cl.mem_flags

        self.u1_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.u2_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.u3_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.v1_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.v2_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.v3_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)

        self.eps_xx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.eps_yy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.eps_zz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.eps_xy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.eps_yz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.eps_xz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)

        self.sigma_xx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.sigma_yy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.sigma_zz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.sigma_xy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.sigma_yz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)
        self.sigma_xz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.empty)


    def dynamics(self):

        launch = self.prog.get_strains(self.queue, self.empty.shape, None,
        self.u1_buf, self.u2_buf, self.u3_buf,
        self.sigma_xx_buf,self.sigma_yy_buf,self.sigma_zz_buf,
        self.sigma_xy_buf,self.sigma_yz_buf,self.sigma_xz_buf,
        self.eps_xx_buf,  self.eps_yy_buf,  self.eps_zz_buf,
        self.eps_xy_buf,  self.eps_yz_buf,  self.eps_xz_buf)
        launch.wait()

        launch = self.prog.integrate(self.queue, self.empty.shape, None,
        self.u1_buf, self.u2_buf, self.u3_buf,
        self.v1_buf, self.v2_buf, self.v3_buf,
        self.sigma_xx_buf,self.sigma_yy_buf,self.sigma_zz_buf,
        self.sigma_xy_buf,self.sigma_yz_buf,self.sigma_xz_buf)
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

    def get_eps(self):

        cl.enqueue_copy(self.queue, self.eps_xx, self.eps_xx_buf)
        cl.enqueue_copy(self.queue, self.eps_yy, self.eps_yy_buf)
        cl.enqueue_copy(self.queue, self.eps_zz, self.eps_zz_buf)
        cl.enqueue_copy(self.queue, self.eps_xy, self.eps_xy_buf)
        cl.enqueue_copy(self.queue, self.eps_yz, self.eps_yz_buf)
        cl.enqueue_copy(self.queue, self.eps_xz, self.eps_xz_buf)

        return {'eps_xx':self.eps_xx,'eps_yy': self.eps_yy,'eps_zz': self.eps_zz,
                'eps_xy':self.eps_xy,'eps_yz': self.eps_yz,'eps_xz': self.eps_xz}

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
