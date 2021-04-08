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

        float eps_ii(float pos, float neg, float di)
            {
            float eps = (pos-neg)/di;
            return eps;
            }

        float didj(float pos,float neg, float dj)
            {
            float eps = (pos-neg)/dj;
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


        __kernel void get_dstrains(
        __global const float *u1,   __global const float *u2,   __global const float *u3,
        __global float *dsigmaxxdx, __global float *dsigmayydy, __global float *dsigmazzdz,
        __global float *dsigmaxydx, __global float *dsigmayzdy, __global float *dsigmaxzdx,
        __global float *dsigmaxydy, __global float *dsigmayzdz, __global float *dsigmaxzdz,
        __global float *eps_xx,     __global float *eps_yy,     __global float *eps_zz,
        __global float *eps_xy,     __global float *eps_yz,     __global float *eps_xz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''
            eps_xx[i]=eps_ii(u1[r],u1[l],2*dx);
            eps_yy[i]=eps_ii(u2[b],u2[f],2*dy);
            eps_zz[i]=eps_ii(u3[u],u3[d],2*dz);

            float r_eps_xx=eps_ii(u1[r], u1[i],   dx);
            float l_eps_xx=eps_ii(u1[i], u1[l],   dx);
            float b_eps_xx=eps_ii(u1[br],u1[bl],2*dx);
            float f_eps_xx=eps_ii(u1[fr],u1[fl],2*dx);
            float u_eps_xx=eps_ii(u1[ur],u1[ul],2*dx);
            float d_eps_xx=eps_ii(u1[dr],u1[dl],2*dx);

            float b_eps_yy=eps_ii(u2[b], u2[i],   dy);
            float f_eps_yy=eps_ii(u2[i], u2[f],   dy);
            float r_eps_yy=eps_ii(u2[br],u2[fr],2*dy);
            float l_eps_yy=eps_ii(u2[bl],u2[fl],2*dy);
            float u_eps_yy=eps_ii(u2[ub],u2[uf],2*dy);
            float d_eps_yy=eps_ii(u2[db],u2[df],2*dy);

            float u_eps_zz=eps_ii(u3[u], u3[i],   dz);
            float d_eps_zz=eps_ii(u3[i], u3[d],   dz);
            float r_eps_zz=eps_ii(u3[ur],u3[dr],2*dz);
            float l_eps_zz=eps_ii(u3[ul],u3[dl],2*dz);
            float b_eps_zz=eps_ii(u3[ub],u3[db],2*dz);
            float f_eps_zz=eps_ii(u3[uf],u3[df],2*dz);

            float dzdx =didj(u3[r], u3[l],      2*dx);
            float rdzdx=didj(u3[r], u3[i],        dx);
            float ldzdx=didj(u3[i], u3[l],        dx);
            float udzdx=didj(u3[ur],u3[ul],     2*dx);
            float ddzdx=didj(u3[dr],u3[dl],     2*dx);

            float dydx =didj(u2[r], u2[l],2*dx);
            float rdydx=didj(u2[r], u2[i],dx);
            float ldydx=didj(u2[i], u2[l],dx);
            float bdydx=didj(u2[br],u2[bl],2*dx);
            float fdydx=didj(u2[fr],u2[fl],2*dx);

            float dxdy =didj(u1[b], u1[f],2*dy);
            float rdxdy=didj(u1[br],u1[fr],2*dy);
            float ldxdy=didj(u1[bl],u1[fl],2*dy);
            float bdxdy=didj(u1[b], u1[i],dy);
            float fdxdy=didj(u1[i], u1[f],dy);

            float dzdy =didj(u3[b], u3[f], 2*dy);
            float bdzdy=didj(u3[b], u3[i],   dy);
            float fdzdy=didj(u3[i], u3[f],   dy);
            float ddzdy=didj(u3[db],u3[df],2*dy);
            float udzdy=didj(u3[ub],u3[uf],2*dy);

            float dxdz =didj(u1[u], u1[d], 2*dz);
            float rdxdz=didj(u1[ur],u1[dr],2*dz);
            float ldxdz=didj(u1[ul],u1[dl],2*dz);
            float udxdz=didj(u1[u], u1[i],   dz);
            float ddxdz=didj(u1[i], u1[d]   ,dz);

            float dydz =didj(u2[u], u2[d], 2*dz);
            float bdydz=didj(u2[ub],u2[db],2*dz);
            float fdydz=didj(u2[uf],u2[df],2*dz);
            float udydz=didj(u2[u], u2[i],   dz);
            float ddydz=didj(u2[i], u2[d],   dz);

            if (back_bd) {
            eps_yy[i]=f_eps_yy;

            r_eps_yy=eps_ii(u2[r],u2[fr],dy);
            l_eps_yy=eps_ii(u2[l],u2[fl],dy);
            u_eps_yy=eps_ii(u2[u],u2[uf],dy);
            d_eps_yy=eps_ii(u2[d],u2[df],dy);

            dxdy =didj(u1[i],u1[f] ,dy);
            rdxdy=didj(u1[r],u1[fr],dy);
            ldxdy=didj(u1[l],u1[fl],dy);
            dzdy =didj(u3[i],u3[f] ,dy);
            ddzdy=didj(u3[d],u3[df],dy);
            udzdy=didj(u3[u],u3[uf],dy);
            };

            if (front_bd) {
            eps_yy[i]=b_eps_yy;

            r_eps_yy=eps_ii(u2[br],u2[r],dy);
            l_eps_yy=eps_ii(u2[bl],u2[l],dy);
            u_eps_yy=eps_ii(u2[ub],u2[u],dy);
            d_eps_yy=eps_ii(u2[db],u2[d],dy);

            dxdy =didj(u1[b], u1[i],dy);
            rdxdy=didj(u1[br],u1[r],dy);
            ldxdy=didj(u1[bl],u1[l],dy);
            dzdy =didj(u3[b], u3[i],dy);
            ddzdy=didj(u3[db],u3[d],dy);
            udzdy=didj(u3[ub],u3[u],dy);
            };

            if (right_bd) {
            eps_xx[i]=l_eps_xx;

            b_eps_xx=eps_ii(u1[b],u1[bl],dx);
            f_eps_xx=eps_ii(u1[f],u1[fl],dx);
            u_eps_xx=eps_ii(u1[u],u1[ul],dx);
            d_eps_xx=eps_ii(u1[d],u1[dl],dx);

            dzdx =didj(u3[i],u3[l], dx);
            udzdx=didj(u3[u],u3[ul],dx);
            ddzdx=didj(u3[d],u3[dl],dx);
            dydx =didj(u2[i],u2[l], dx);
            bdydx=didj(u2[b],u2[bl],dx);
            fdydx=didj(u2[f],u2[fl],dx);
            };

            if (left_bd) {
            eps_xx[i]=r_eps_xx;

            b_eps_xx=eps_ii(u1[br],u1[b],dx);
            f_eps_xx=eps_ii(u1[fr],u1[f],dx);
            u_eps_xx=eps_ii(u1[ur],u1[u],dx);
            d_eps_xx=eps_ii(u1[dr],u1[d],dx);

            dzdx =didj(u3[r], u3[i],dx);
            udzdx=didj(u3[ur],u3[u],dx);
            ddzdx=didj(u3[dr],u3[d],dx);
            dydx =didj(u2[r], u2[i],dx);
            bdydx=didj(u2[br],u2[b],dx);
            fdydx=didj(u2[fr],u2[f],dx);
            };

            if (down_bd) {

            eps_zz[i]=eps_ii(u3[u],u3[i],dz);

            d_eps_xx=Eps_xx;
            d_eps_yy=Eps_yy;

            d_eps_zz=eps_ii(u3[i], 0,    dz);
            r_eps_zz=eps_ii(u3[ur],u3[r],dz);
            l_eps_zz=eps_ii(u3[ul],u3[l],dz);
            b_eps_zz=eps_ii(u3[ub],u3[b],dz);
            f_eps_zz=eps_ii(u3[uf],u3[f],dz);

            ddzdx=0;
            ddzdy=0;

            dxdz =didj(u1[u], u1[i], dz);
            rdxdz=didj(u1[ur],u1[r], dz);
            ldxdz=didj(u1[ul],u1[l], dz);
            ddxdz=didj(u1[i], Eps_xx * dx * x, dz);

            dydz =didj(u2[u], u2[i], dz);
            bdydz=didj(u2[ub],u2[b], dz);
            fdydz=didj(u2[uf],u2[f], dz);
            ddydz=didj(u2[i], Eps_yy * dy * y, dz);
            };


            if (up_bd) {
            eps_zz[i]=eps_ii(u3[i],u3[d], dz);

            r_eps_zz=eps_ii(u3[r],u3[dr],dz);
            l_eps_zz=eps_ii(u3[l],u3[dl],dz);
            b_eps_zz=eps_ii(u3[b],u3[db],dz);
            f_eps_zz=eps_ii(u3[f],u3[df],dz);

            dxdz =didj(u1[i],u1[d], dz);
            rdxdz=didj(u1[r],u1[dr],dz);
            ldxdz=didj(u1[l],u1[dl],dz);
            dydz =didj(u2[i],u2[d], dz);
            bdydz=didj(u2[b],u2[db],dz);
            fdydz=didj(u2[f],u2[df],dz);
            };

            barrier(CLK_GLOBAL_MEM_FENCE);

            eps_xz[i]=eps_ij( dxdz, dzdx);
            float u_eps_xz=eps_ij(udxdz,udzdx);
            float d_eps_xz=eps_ij(ddxdz,ddzdx);
            float r_eps_xz=eps_ij(rdxdz,rdzdx);
            float l_eps_xz=eps_ij(ldxdz,ldzdx);

            eps_xy[i]=eps_ij( dxdy, dydx);
            float r_eps_xy=eps_ij(rdxdy,rdydx);
            float l_eps_xy=eps_ij(ldxdy,ldydx);
            float b_eps_xy=eps_ij(bdxdy,bdydx);
            float f_eps_xy=eps_ij(fdxdy,fdydx);

            eps_yz[i]=eps_ij( dydz, dzdy);
            float b_eps_yz=eps_ij(bdydz,bdzdy);
            float f_eps_yz=eps_ij(fdydz,fdzdy);
            float u_eps_yz=eps_ij(udydz,udzdy);
            float d_eps_yz=eps_ij(ddydz,ddzdy);


            barrier(CLK_GLOBAL_MEM_FENCE);

            float r_sigma_xx  = sigma_ii(c11,r_eps_xx,c12,r_eps_yy,r_eps_zz);
            float l_sigma_xx  = sigma_ii(c11,l_eps_xx,c12,l_eps_yy,l_eps_zz);
            float b_sigma_yy  = sigma_ii(c11,b_eps_yy,c12,b_eps_xx,b_eps_zz);
            float f_sigma_yy  = sigma_ii(c11,f_eps_yy,c12,f_eps_xx,f_eps_zz);
            float u_sigma_zz  = sigma_ii(c11,u_eps_zz,c12,u_eps_xx,u_eps_yy);
            float d_sigma_zz  = sigma_ii(c11,d_eps_zz,c12,d_eps_xx,d_eps_yy);

            float r_sigma_xy = sigma_ij(c44, r_eps_xy);
            float l_sigma_xy = sigma_ij(c44, l_eps_xy);
            float b_sigma_xy = sigma_ij(c44, b_eps_xy);
            float f_sigma_xy = sigma_ij(c44, f_eps_xy);

            float r_sigma_xz = sigma_ij(c44, r_eps_xz);
            float l_sigma_xz = sigma_ij(c44, l_eps_xz);
            float u_sigma_xz = sigma_ij(c44, u_eps_xz);
            float d_sigma_xz = sigma_ij(c44, d_eps_xz);

            float b_sigma_yz = sigma_ij(c44, b_eps_yz);
            float f_sigma_yz = sigma_ij(c44, f_eps_yz);
            float d_sigma_yz = sigma_ij(c44, d_eps_yz);
            float u_sigma_yz = sigma_ij(c44, u_eps_yz);

            dsigmaxxdx[i] = (r_sigma_xx-l_sigma_xx)/(2*dx);
            dsigmayydy[i] = (b_sigma_yy-f_sigma_yy)/(2*dy);
            dsigmazzdz[i] = (u_sigma_zz-d_sigma_zz)/(2*dz);

            dsigmaxydx[i] = (r_sigma_xy-l_sigma_xy)/(2*dx);
            dsigmaxydy[i] = (b_sigma_xy-f_sigma_xy)/(2*dy);

            dsigmaxzdx[i] = (r_sigma_xz-l_sigma_xz)/(2*dx);
            dsigmaxzdz[i] = (u_sigma_xz-d_sigma_xz)/(2*dz);

            dsigmayzdy[i] = (b_sigma_yz-f_sigma_yz)/(2*dy);
            dsigmayzdz[i] = (u_sigma_yz-d_sigma_yz)/(2*dz);

            if (up_bd) {
            dsigmazzdz[i] = (-d_sigma_zz)/dz;
            dsigmaxzdz[i] = (-d_sigma_xz)/dz;
            dsigmayzdz[i] = (-d_sigma_yz)/dz;

            dsigmaxzdx[i] = 0;
            dsigmayzdy[i] = 0;
            };

            if (pochti_up_bd) {
            dsigmazzdz[i] = (-d_sigma_zz)/(2*dz);
            dsigmaxzdz[i] = (-d_sigma_xz)/(2*dz);
            dsigmayzdz[i] = (-d_sigma_yz)/(2*dz);
            };

            if (back_bd) {
            dsigmayydy[i] = (-f_sigma_yy)/dy;
            dsigmaxydy[i] = (-f_sigma_xy)/dy;
            dsigmayzdy[i] = (-f_sigma_yz)/dy;

            dsigmaxydx[i] = 0;
            dsigmayzdz[i] = 0;
            };

            if (pochti_back_bd) {
            dsigmayydy[i] = (-f_sigma_yy)/(2*dy);
            dsigmaxydy[i] = (-f_sigma_xy)/(2*dy);
            dsigmayzdy[i] = (-f_sigma_yz)/(2*dy);
            };

            if (front_bd) {
            dsigmayydy[i] = (b_sigma_yy)/dy;
            dsigmaxydy[i] = (b_sigma_xy)/dy;
            dsigmayzdy[i] = (b_sigma_yz)/dy;

            dsigmaxydx[i] = 0;
            dsigmayzdz[i] = 0;
            };

            if (pochti_front_bd) {
            dsigmayydy[i] = (b_sigma_yy)/(2*dy);
            dsigmaxydy[i] = (b_sigma_xy)/(2*dy);
            dsigmayzdy[i] = (b_sigma_yz)/(2*dy);
            };

            if (right_bd) {
            dsigmaxxdx[i] = (-l_sigma_xx)/dx;
            dsigmaxydx[i] = (-l_sigma_xy)/dx;
            dsigmaxzdx[i] = (-l_sigma_xz)/dx;

            dsigmaxydy[i] = 0;
            dsigmaxzdz[i] = 0;
            };

            if (pochti_right_bd) {
            dsigmaxxdx[i] = (-l_sigma_xx)/(2*dx);
            dsigmaxydx[i] = (-l_sigma_xy)/(2*dx);
            dsigmaxzdx[i] = (-l_sigma_xz)/(2*dx);
            };

            if (left_bd) {
            dsigmaxxdx[i] = (r_sigma_xx)/dx;
            dsigmaxydx[i] = (r_sigma_xy)/dx;
            dsigmaxzdx[i] = (r_sigma_xz)/dx;

            dsigmaxydy[i] = 0;
            dsigmaxzdz[i] = 0;
            };

            if (pochti_left_bd) {
            dsigmaxxdx[i] = (r_sigma_xx)/(2*dx);
            dsigmaxydx[i] = (r_sigma_xy)/(2*dx);
            dsigmaxzdx[i] = (r_sigma_xz)/(2*dx);
            };

            if (up_bd && x_bd) {
            dsigmaxydy[i] = 0;
            dsigmaxzdz[i] = 0;
            dsigmaxzdx[i] = 0;
            dsigmayzdy[i] = 0;
            };

            if (up_bd && y_bd) {
            dsigmaxzdx[i] = 0;
            dsigmayzdy[i] = 0;
            dsigmaxydx[i] = 0;
            dsigmayzdz[i] = 0;
            };

            if (x_bd && y_bd) {
            dsigmaxydx[i] = 0;
            dsigmayzdz[i] = 0;
            dsigmaxydy[i] = 0;
            dsigmaxzdz[i] = 0;
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
        __global float *u1,         __global float *u2,         __global float *u3,
        __global float *v1,         __global float *v2,         __global float *v3,
        __global float *dsigmaxxdx, __global float *dsigmayydy, __global float *dsigmazzdz,
        __global float *dsigmaxydx, __global float *dsigmayzdy, __global float *dsigmaxzdx,
        __global float *dsigmaxydy, __global float *dsigmayzdz, __global float *dsigmaxzdz)

        {   int i = get_global_id(0);
            ''' + self.coord_syst + '''

            float U1f = (2*(dsigmaxxdx[i] + dsigmaxydy[i] + dsigmaxzdz[i])*dt2 + 4*rho*u1[i] + (Alpha*dt - 2*rho)*v1[i])/(Alpha*dt + 2*rho);
            float U2f = (2*(dsigmaxydx[i] + dsigmayydy[i] + dsigmayzdz[i])*dt2 + 4*rho*u2[i] + (Alpha*dt - 2*rho)*v2[i])/(Alpha*dt + 2*rho);
            float U3f = (2*(dsigmaxzdx[i] + dsigmayzdy[i] + dsigmazzdz[i])*dt2 + 4*rho*u3[i] + (Alpha*dt - 2*rho)*v3[i])/(Alpha*dt + 2*rho);

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

        self.u1 = np.zeros(self.L).astype(np.float32)
        self.u2 = np.zeros(self.L).astype(np.float32)
        self.u3 = np.zeros(self.L).astype(np.float32)
        self.v1 = np.zeros(self.L).astype(np.float32)
        self.v2 = np.zeros(self.L).astype(np.float32)
        self.v3 = np.zeros(self.L).astype(np.float32)

        self.eps_xx = np.zeros(self.L).astype(np.float32)
        self.eps_yy = np.zeros(self.L).astype(np.float32)
        self.eps_zz = np.zeros(self.L).astype(np.float32)
        self.eps_xy = np.zeros(self.L).astype(np.float32)
        self.eps_yz = np.zeros(self.L).astype(np.float32)
        self.eps_xz = np.zeros(self.L).astype(np.float32)

        self.dsigmaxxdx = np.zeros(self.L).astype(np.float32)
        self.dsigmayydy = np.zeros(self.L).astype(np.float32)
        self.dsigmazzdz = np.zeros(self.L).astype(np.float32)
        self.dsigmaxydx = np.zeros(self.L).astype(np.float32)
        self.dsigmaxydy = np.zeros(self.L).astype(np.float32)
        self.dsigmayzdy = np.zeros(self.L).astype(np.float32)
        self.dsigmayzdz = np.zeros(self.L).astype(np.float32)
        self.dsigmaxzdx = np.zeros(self.L).astype(np.float32)
        self.dsigmaxzdz = np.zeros(self.L).astype(np.float32)

        mf = cl.mem_flags

        self.u1_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.u1)
        self.u2_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.u2)
        self.u3_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.u3)
        self.v1_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.v1)
        self.v2_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.v2)
        self.v3_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.v3)

        self.eps_xx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xx)
        self.eps_yy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_yy)
        self.eps_zz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_zz)
        self.eps_xy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xy)
        self.eps_yz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_yz)
        self.eps_xz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.eps_xz)

        self.dsigmaxxdx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmaxxdx)
        self.dsigmayydy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmayydy)
        self.dsigmazzdz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmazzdz)
        self.dsigmaxydx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmaxydx)
        self.dsigmaxydy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmaxydy)
        self.dsigmayzdy_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmayzdy)
        self.dsigmayzdz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmayzdz)
        self.dsigmaxzdx_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmaxzdx)
        self.dsigmaxzdz_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dsigmaxzdz)


    def dynamics(self):

        launch = self.prog.get_dstrains(self.queue, self.u1.shape, None,
        self.u1_buf, self.u2_buf, self.u3_buf,
        self.dsigmaxxdx_buf,self.dsigmayydy_buf,self.dsigmazzdz_buf,
        self.dsigmaxydx_buf,self.dsigmayzdy_buf,self.dsigmaxzdx_buf,
        self.dsigmaxydy_buf,self.dsigmayzdz_buf,self.dsigmaxzdz_buf,
        self.eps_xx_buf,  self.eps_yy_buf,  self.eps_zz_buf,
        self.eps_xy_buf,  self.eps_yz_buf,  self.eps_xz_buf)
        launch.wait()

        launch = self.prog.integrate(self.queue, self.u1.shape, None,
        self.u1_buf, self.u2_buf, self.u3_buf,
        self.v1_buf, self.v2_buf, self.v3_buf,
        self.dsigmaxxdx_buf,self.dsigmayydy_buf,self.dsigmazzdz_buf,
        self.dsigmaxydx_buf,self.dsigmayzdy_buf,self.dsigmaxzdx_buf,
        self.dsigmaxydy_buf,self.dsigmayzdz_buf,self.dsigmaxzdz_buf)
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
