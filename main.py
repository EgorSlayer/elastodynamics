import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from math import cos, sin, pi
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt
from numpy.random import random, normal
from matplotlib.ticker import MaxNLocator
import telebot
import reikna.cluda as cluda
from reikna.fft import FFT

class ME:

    def __init__(self):
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)

    # initiate geometry and heterostructure films
    def init_structure(self,interfaces,Lx,Ly,Lz,dx,dy,dz,dt,static=False, T=0,Hz=0,pin=False,LDy=4,LDx=50):

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.L  = Lx * Ly * Lz

        self.Lzm = int(interfaces['CoFe'][1] - interfaces['CoFe'][0])+1
        self.Lm  = int(Lx * Ly * self.Lzm)
        self.specL =  (2 * Lx - 1) * (2 * Ly - 1) * (2 * self.Lzm - 1)

        if static == False:
            Alpha = 0.01             #Damping M
            Beta = 0                 #Damping EL
        else:
            Alpha = 1
            Beta = 10**16

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Alpha = Alpha
        self.Beta =  Beta
        self.T  = T      #K
        self.Hz = Hz     #A/m
        self.pin = pin
        self.LDx = LDx
        self.LDy = LDy

        self.mag_consts = '''

        const float Gamma = 1.76085e11;
        const float Mu0 = 1.25663e-6;
        const float bolzman = 1.38e-23;
        const float Alpha = ''' + str(self.Alpha)  + ''';
        const float Alpha2 = Alpha*Alpha;
        const float K1CoFe = -1560.6;
        const float KsCoFe = 1.18e-3;
        const float MsCoFe = 16e5;
        const float ACoFe =  2.5e-11;
        const float B1CoFe = -20.18e6;
        const float B2CoFe = -15.272e6;
        float D0CoFe = 2.26e-3;
        D0CoFe = 0.82e-11;
        float Omega = D0CoFe/ACoFe;

        const float T =  ''' + str(self.T)  + ''';
        const float Hz = ''' + str(self.Hz) + '''e5;
        '''

        self.shift = int(interfaces['CoFe'][0]*self.Lx*self.Ly)

        self.coord_syst_mag = '''

        const int lx = '''      + str(self.Lx) + ''';
        const int ly = '''      + str(self.Ly) + ''';
        const int lz = '''      + str(self.Lzm) + ''';
        const int speclen = ''' + str(self.specL) + ''';
        const int len = '''     + str(self.Lm) + ''';

        const int pl = lx * ly;
        int x = i % lx;
        int z = i / pl;
        int y = (i - z * pl)/ lx;
        int speci = z * (2 * ly - 1) * (2 * lx - 1) + y * (2 * lx - 1) + x ;


        const float dt = ''' + str(self.dt) + '''e-18;
        const float dx = ''' + str(self.dx) + '''e-9;
        const float dy = ''' + str(self.dy) + '''e-9;
        const float dz = ''' + str(self.dz) + '''e-9;
        const float dx2 =  dx*dx;
        const float dy2 =  dy*dy;
        const float dz2 =  dz*dz;

        int l=i-1;
        int r=i+1;
        int f=i-lx;
        int b=i+lx;
        int u=i+pl;
        int d=i-pl;

        '''

        self.coord_syst_el = '''

        const int lx = '''      + str(self.Lx) + ''';
        const int ly = '''      + str(self.Ly) + ''';
        const int lz = '''      + str(self.Lz) + ''';
        const int len = '''     + str(self.L) + ''';

        const int pl = lx * ly;
        int x = i % lx;
        int z = i / pl;
        int y = (i - z * pl)/ lx;

        const float dt = ''' + str(self.dt) + '''e-18;
        const float dx = ''' + str(self.dx) + '''e-9;
        const float dy = ''' + str(self.dy) + '''e-9;
        const float dz = ''' + str(self.dz) + '''e-9;
        const float dx2 =  dx*dx;
        const float dy2 =  dy*dy;
        const float dz2 =  dz*dz;

        int l=i-1;
        int r=i+1;
        int f=i-lx;
        int b=i+lx;
        int u=i+pl;
        int d=i-pl;

        '''

        bounds_el = '''

        const int LDx = ''' + str(self.LDx) + ''';
        const int LDy = ''' + str(self.LDy) + ''';

        bool left_bd =  (x == 0);
        bool right_bd = (x == lx-1);
        bool front_bd = (y == 0);
        bool back_bd =  (y == ly-1);
        bool down_bd =  (z == 0);
        bool up_bd =    (z == lz-1);

        //bool left_bd =  (x == 0    || ((y <= ly/2 + LDy/2) && (y >= ly/2 - LDy/2) && (x == LDx)));
        //bool right_bd = (x == lx-1);
        //bool front_bd = (y == 0    ||  ((x < LDx) && (y == ly/2 + LDy/2)));
        //bool back_bd =  (y == ly-1 ||  ((x < LDx) && (y == ly/2 - LDy/2)));
        //bool down_bd =  (z == 0);
        //bool up_bd =    (z == lz-1);

        bool x_bd = right_bd || left_bd;
        bool y_bd = back_bd  || front_bd;
        bool z_bd = up_bd    || down_bd;
        '''

        bounds_mag = bounds_el

        if pin:

            bounds_mag = '''

            const int LDx = ''' + str(self.LDx) + ''';
            const int LDy = ''' + str(self.LDy) + ''';

            bool left_bd =  (x == 0    || ((y <= ly/2 + LDy/2) && (y >= ly/2 - LDy/2) && (x == LDx)));
            bool right_bd = (x == lx-1);
            bool front_bd = (y == 0    ||  ((x < LDx) && (y == ly/2 + LDy/2)));
            bool back_bd =  (y == ly-1 ||  ((x < LDx) && (y == ly/2 - LDy/2)));
            bool down_bd =  (z == 0);
            bool up_bd =    (z == lz-1);

            bool x_bd = right_bd || left_bd;
            bool y_bd = back_bd || front_bd;
            bool z_bd = up_bd    || down_bd;
            '''

            bounds_el += '''
            const bool inside = ((x < LDx)  &  (y > ly/2 - LDy / 2) & (y < ly / 2 + LDy / 2));
            '''

            bounds_mag += '''
            const bool inside = ((x < LDx)  &  (y > ly/2 - LDy / 2) & (y < ly / 2 + LDy / 2));
            '''
        else:
            bounds_el += '''
            const bool inside = false;
            '''

            bounds_mag += '''
            const bool inside = false;
            '''

        self.coord_syst_mag += bounds_mag
        self.coord_syst_el += bounds_el

        self.coord_syst_el += '''

        float Alpha = '''+str(self.Beta)+''';

        const float small = 10e-19;
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

        const float B1CoFe =  -20e6;
        const float B2CoFe =  -15e6;

        const float aCoFe = 2.8;

        const float c11MgO = 273e9;
        const float c12MgO = 91e9;
        const float c44MgO = 141e9;
        const float rhoMgO = 3470;

        const float B1MgO = 0;
        const float B2MgO = 0;

        const float aMgO = 3.0;

        const float c11Pt = 303e9;
        const float c12Pt = 220e9;
        const float c44Pt = 54e9;
        const float rhoPt = 20600;

        const float B1Pt = 0;
        const float B2Pt = 0;

        const float aPt = 3.0;

        float time_val = time_val_arr[0];
        if (i==0){
        time_val_arr[0] = time_val_arr[0] + 1;
        };

        float c11 = NAN;
        float c12 = NAN;
        float c44 = NAN;
        float rho = NAN;

        float B1 = NAN;
        float B2 = NAN;

        float eps_m = NAN;
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

        # Magnetostriction

        self.mgstr_code = '''

        float dmxmxdx = (mx[r-'''+str(self.shift)+''']*mx[r-'''+str(self.shift)+'''] - mx[i-'''+str(self.shift)+''']*mx[i-'''+str(self.shift)+'''])/(dx);
        float dmymydy = (my[b-'''+str(self.shift)+''']*my[b-'''+str(self.shift)+'''] - my[i-'''+str(self.shift)+''']*my[i-'''+str(self.shift)+'''])/(dy);
        float dmzmzdz = (mz[u-'''+str(self.shift)+''']*mz[u-'''+str(self.shift)+'''] - mz[i-'''+str(self.shift)+''']*mz[i-'''+str(self.shift)+'''])/(dz);

        float dmxmydy = (mx[i-'''+str(self.shift)+''']*my[i-'''+str(self.shift)+'''] - mx[f-'''+str(self.shift)+''']*my[f-'''+str(self.shift)+'''])/(dy);
        float dmxmzdz = (mx[i-'''+str(self.shift)+''']*mz[i-'''+str(self.shift)+'''] - mx[d-'''+str(self.shift)+''']*mz[d-'''+str(self.shift)+'''])/(dz);
        float dmxmydx = (mx[i-'''+str(self.shift)+''']*my[i-'''+str(self.shift)+'''] - mx[l-'''+str(self.shift)+''']*my[l-'''+str(self.shift)+'''])/(dx);
        float dmymzdz = (my[i-'''+str(self.shift)+''']*mz[i-'''+str(self.shift)+'''] - my[d-'''+str(self.shift)+''']*mz[d-'''+str(self.shift)+'''])/(dz);
        float dmxmzdx = (mx[i-'''+str(self.shift)+''']*mz[i-'''+str(self.shift)+'''] - mx[l-'''+str(self.shift)+''']*mz[l-'''+str(self.shift)+'''])/(dx);
        float dmymzdy = (my[i-'''+str(self.shift)+''']*mz[i-'''+str(self.shift)+'''] - my[f-'''+str(self.shift)+''']*mz[f-'''+str(self.shift)+'''])/(dy);

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

        self.time_val_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(1).astype(np.float32))

        self.vx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vx)
        self.vy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vy)
        self.vz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vz)

        self.sigmaxx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxx)
        self.sigmayy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmayy)
        self.sigmazz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmazz)
        self.sigmaxy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxy)
        self.sigmayz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmayz)
        self.sigmaxz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxz)

        self.eps_xx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_xx)
        self.eps_yy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_yy)
        self.eps_zz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_zz)
        self.eps_xy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_xy)
        self.eps_yz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_yz)
        self.eps_xz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.eps_xz)

        for material in interfaces:

            self.coord_syst_el += 'if ((z >= ' + str(interfaces[material][0]) + ') & (z <= ' + str(interfaces[material][1]) + ''')) {
            c11 = c11'''+str(material)+''';
            c12 = c12'''+str(material)+''';
            c44 = c44'''+str(material)+''';
            rho = rho'''+str(material)+''';

            B1 =  B1'''+str(material)+''';
            B2 =  B2'''+str(material)+''';

            eps_m = (aMgO-aCoFe)/aCoFe;
            eps_m = 0;

            if (inside) {
            c11 = c11Pt;
            c12 = c12Pt;
            c44 = c44Pt;
            rho = big;
            };

            };
            '''

            #self.bd_code_str +='if ((z == '+str(interfaces[material][1]) + ''') & !z_bd) {
            #dvxdz = (vx[i]*eps_m)/dz;
            #dvydz = (vy[i]*eps_m)/dz;
            #};
            #'''

            #self.bd_code_vel +='if ((z == '+str(interfaces[material][1]) + ''') & !z_bd) {
            #dsigmazzdz = (c11CoFe*eps_zz[u]+c12CoFe*(2*eps_m+(eps_m+1)*(eps_xx[i]+eps_yy[i]))-sigma_zz[i])/dz;
            #};
            #'''

            if material == 'CoFe':

                self.mag_consts += '''

                float K1 = K1'''+str(material)+''';
                float Ks = Ks'''+str(material)+''';
                float Ms = Ms'''+str(material)+''';
                float A =  A''' +str(material)+''';
                float B1 = B1'''+str(material)+''';
                float B2 = B2'''+str(material)+''';
                float D0 = D0'''+str(material)+''';
                '''

    def init_el_BC(self, R_BD, L_BD, B_BD, F_BD, U_BD, D_BD):

        self.R_BD = R_BD
        self.L_BD = L_BD
        self.B_BD = B_BD
        self.F_BD = F_BD
        self.U_BD = U_BD
        self.D_BD = D_BD

        self.X_BD = None
        self.Y_BD = None
        self.Z_BD = None

        if R_BD == 'Free' and L_BD == 'Free':
            self.X_BD = 'Free'
        if B_BD == 'Free' and F_BD == 'Free':
            self.Y_BD = 'Free'
        if U_BD == 'Free' and D_BD == 'Free':
            self.Z_BD = 'Free'

        # upper bound managment


        if self.U_BD == 'Free':

            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 0;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.U_BD == 'Free2':

            self.bd_code_vel += '''
            if (up_bd & (lz>1)) {
            dsigmazzdz = -sigma_zz[i]/dz;
            };
            '''

            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            dvxdz = -dvzdx;
            dvydz = -dvzdy;
            };
            '''

        elif self.U_BD == 'z_stressed':

            self.bd_code_vel += '''
            if (up_bd & (lz>1)) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            //dvzdz = 0;
            //dvxdz = -dvzdx;
            //dvydz = -dvzdy;

            sigmazz_bd = 1e7;
            //sigmaxz_bd = 0;
            //sigmayz_bd = 0;
            };
            '''

        elif self.U_BD == 'z_stressed2':

            self.bd_code_vel += '''
            if (up_bd) & (lz>1)) {
            fz = 10e7/dz;
            };
            '''

            self.bd_code_str += '''
            if (up_bd) {
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            };
            '''
        elif self.U_BD == 'Absobtion':

            self.bd_code_vel += '''
            if (up_bd || z == lz-2 || z == lz-3 || z == lz-4 || z == lz-5) {
            Alpha = 1e23;
            };
            '''

        elif self.U_BD == 'Free_ab':

            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 0;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;

            Alpha = 1e23;
            };
            '''
        elif self.U_BD == 'Open':

            self.bd_code_vel += '''
            if (up_bd & (lz>1)) {
            dsigmazzdz = -sqrt(c11*rho)*(vz[i]-vz[d])/dz;
            };
            '''

            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            dvydz = -(sigma_yz[i]-sigma_yz[d])/dz/sqrt(c44*rho);
            dvxdz = -(sigma_xz[i]-sigma_xz[d])/dz/sqrt(c44*rho);
            };
            '''

        elif self.U_BD == 'Fixed':

            self.bd_code_vel += '''
            if (up_bd & (lz>1)) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (up_bd & (lz>1)) {

            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            };
            '''


        # lower bound managment

        if self.D_BD == 'Free':

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

        elif self.D_BD == 'Free2':

            self.bd_code_str += '''
            if (down_bd) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            };
            '''

            self.bd_code_vel += '''
            if (down_bd) {
            dsigmaxzdz = sigma_xz[i]/dz;
            dsigmayzdz = sigma_yz[i]/dz;
            };
            '''

        elif self.D_BD == 'z_stressed':

            self.bd_code_vel += '''
            if (down_bd & (lz>1)) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (down_bd & (lz>1)) {
            dvzdz = 0;
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            sigmazz_bd = 1e7;
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.D_BD == 'z_stressed2':

            self.bd_code_vel += '''
            if (z==1 & (lz>1)) {
            fz = -10e7/dz;
            };
            '''



        elif self.D_BD == 'PZN-PT':

            self.bd_code_str += '''

            if (down_bd) {
            sigmaxx_bd = 0.01*c11-0.01*c12;
            sigmayy_bd = -0.01*c11+0.01*c12;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        elif self.D_BD == 'Andreys substrate':

            self.bd_code_str += '''
            if (down_bd & (lz>1)) {
            dvxdz = vx[i]*(eps_m/(1-eps_m))/dz;
            dvydz = vy[i]*(eps_m/(1-eps_m))/dz;

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
            if (down_bd & (lz>1)) {
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (down_bd) {

            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            };
            '''

        elif len(self.D_BD) == 3:

            self.bd_code_vel += '''

            if (down_bd) {

            if (x==lx-2) {
            fx = (c11*'''+str(self.D_BD[0])+'''+c12*('''+str(self.D_BD[1])+'''+'''+str(self.D_BD[2])+'''))/dx;
            };
            if (x==1) {
            fx = -(c11*'''+str(self.D_BD[0])+'''+c12*('''+str(self.D_BD[1])+'''+'''+str(self.D_BD[2])+'''))/dx;
            };
            if (y==ly-2) {
            fy = (c11*'''+str(self.D_BD[1])+'''+c12*('''+str(self.D_BD[0])+'''+'''+str(self.D_BD[2])+'''))/dy;
            };
            if (y==1) {
            fy = -(c11*'''+str(self.D_BD[1])+'''+c12*('''+str(self.D_BD[0])+'''+'''+str(self.D_BD[2])+'''))/dy;
            };

            };


            '''

        elif self.D_BD == 'Emmiter':

            self.bd_code_str += '''

            if (down_bd) {

            sigmaxx_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmayy_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmazz_bd = 0.01*c11*sin(time_val*2*3.1415*dt*1e10);
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;

            };


            '''

        elif self.D_BD == 'Emmiter2.0':

            self.bd_code_vel += '''

            if (down_bd) {

            fz = -(c11*0.01*sin(time_val*2*3.1415*dt*1e8))/dz;

            if (x==lx-2) {
            fx = (c12*0.01*sin(time_val*2*3.1415*dt*1e8))/dx;
            };
            if (x==0) {
            fx = -(c12*0.01*sin(time_val*2*3.1415*dt*1e8))/dx;
            };
            if (y==ly-2) {
            fy = (c12*0.01*sin(time_val*2*3.1415*dt*1e8))/dy;
            };
            if (y==0) {
            fy = -(c12*0.01*sin(time_val*2*3.1415*dt*1e8))/dy;
            };

            };


            '''
        elif self.D_BD == 'Impulse':

            self.bd_code_str += '''

            if (down_bd) {

            const float imp_len = 0.1e-9;
            const int A = time_val*dt/imp_len;

            if (time_val*dt < imp_len) {

            sigmaxx_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmayy_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmazz_bd = 0.01*c11*sin(time_val*2*3.1415*dt*1e10);
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            }
            else {

            sigmaxx_bd = 0;
            sigmayy_bd = 0;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;

            }

            };


            '''

        elif self.D_BD == 'Source':

            self.bd_code_str += '''

            float source_xx = 0;
            float source_yy = 0;
            float source_zz = 0;
            float source_xy = 0;
            float source_yz = 0;
            float source_xz = 0;

            if (down_bd) {
            source_xx = 0.1*c11;
            source_yy = 0.1*c12;
            source_zz = 0.1*c12;
            source_xy = 0;
            source_yz = 0;
            source_xz = 0;
            };


            '''

        # right bound managment

        if self.R_BD == 'Free':

            self.bd_code_str += '''
            if (right_bd) {
            dvxdx = -c12/c11*(dvydy+dvzdz);
            dvydx = -dvxdy;
            dvzdx = -dvxdz;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        elif self.R_BD == 'Free2':

            self.bd_code_vel += '''
            if (right_bd & (lx>1)) {
            dsigmaxxdx = -sigma_xx[i]/dx;
            };
            '''

            self.bd_code_str += '''
            if (right_bd & (lx>1)) {
            dvydx = -dvxdy;
            dvzdx = -dvxdz;
            };
            '''

        elif self.R_BD == 'Open':

            self.bd_code_vel += '''
            if (right_bd & (lx>1)) {
            dsigmaxxdx = -sqrt(c11*rho)*(vx[i]-vx[l])/dx;
            };
            '''

            self.bd_code_str += '''
            if (right_bd & (lx>1)) {
            dvydx = -(sigma_xy[i]-sigma_xy[l])/dx/sqrt(c44*rho);
            dvzdx = -(sigma_xz[i]-sigma_xz[l])/dx/sqrt(c44*rho);
            };
            '''

        # left bound managment

        if self.L_BD == 'Free':

            self.bd_code_str += '''
            if (left_bd) {
            dvxdx = -c12/c11*(dvydy+dvzdz);
            dvydx = -dvxdy;
            dvzdx = -dvxdz;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        elif self.L_BD == 'Free2':



            self.bd_code_vel += '''
            if (left_bd & (lx>1)) {
            dsigmaxzdx = sigma_xz[i]/dx;
            dsigmaxydx = sigma_xy[i]/dx;
            };
            '''

            self.bd_code_str += '''
            if (left_bd & (lx>1)) {
            dvxdx = -c12/c11*(dvzdz+dvydy);
            };
            '''

        elif self.L_BD == 'Emmiter':

            self.bd_code_str += '''

            if (left_bd & (x != LDx)) {
            sigmaxx_bd = 0.01*c11*sin(time_val*2*3.1415*dt*1e10);
            sigmayy_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmazz_bd = 0.01*c12*sin(time_val*2*3.1415*dt*1e10);
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''
        elif self.L_BD == 'Open':

            self.bd_code_vel += '''
            if (left_bd & (lx>1)) {
            dsigmaxydx = sqrt(c11*rho)*(vy[r]-vy[i])/dx;
            dsigmaxzdx = sqrt(c11*rho)*(vz[r]-vz[i])/dx;
            };
            '''

            self.bd_code_str += '''
            if (left_bd & (lx>1)) {
            dvxdx = (sigma_xx[r]-sigma_xx[i])/dx/sqrt(c44*rho);
            };
            '''

        # back bound managment

        if self.B_BD == 'Free':

            self.bd_code_str += '''
            if (back_bd) {
            dvydy = -c12/c11*(dvxdx+dvzdz);
            dvxdy = -dvydx;
            dvzdy = -dvydz;

            sigmayy_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.B_BD == 'Free2':

            self.bd_code_vel += '''
            if (back_bd & (ly>1)) {
            dsigmayydy = -sigma_yy[i]/dy;
            };
            '''

            self.bd_code_str += '''
            if (back_bd & (ly>1)) {
            dvxdy = -dvydx;
            dvzdy = -dvydz;
            };
            '''

        elif self.B_BD == 'Open':

            self.bd_code_vel += '''
            if (back_bd & (ly>1)) {
            dsigmayydy = -sqrt(c11*rho)*(vy[i]-vy[f])/dy;
            };
            '''

            self.bd_code_str += '''
            if (back_bd & (ly>1)) {
            dvxdy = -(sigma_xy[i]-sigma_xy[f])/dy/sqrt(c44*rho);
            dvzdy = -(sigma_yz[i]-sigma_yz[f])/dy/sqrt(c44*rho);
            };
            '''

        # front bound managment

        if self.F_BD == 'Free':

            self.bd_code_str += '''
            if (front_bd) {
            dvydy = -c12/c11*(dvxdx+dvzdz);
            dvxdy = -dvydx;
            dvzdy = -dvydz;

            sigmayy_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.F_BD == 'Free2':

            self.bd_code_str += '''
            if (front_bd & (ly>1)) {
            dvydy = -c12/c11*(dvzdz+dvxdx);
            };
            '''

            self.bd_code_vel += '''
            if (front_bd & (ly>1)) {
            dsigmayzdy = sigma_yz[i]/dy;
            dsigmaxydy = sigma_xy[i]/dy;
            };
            '''

        elif self.F_BD == 'Open':

            self.bd_code_str += '''
            if (front_bd & (ly>1)) {
            dvydy = (sigma_yy[b]-sigma_yy[i])/dx/sqrt(c44*rho);
            };
            '''

            self.bd_code_vel += '''
            if (front_bd & (ly>1)) {
            dsigmaxydy = sqrt(c11*rho)*(vx[b]-vx[i])/dy;
            dsigmayzdy = sqrt(c11*rho)*(vz[b]-vz[i])/dy;
            };
            '''



        if self.F_BD == 'Free2' and self.R_BD == 'Free2':

            self.bd_code_str += '''
            if (front_bd & right_bd & (lx>1) & (ly>1)) {
            dvydy = -c12/(c11+c12)*(dvzdz);
            dvydx = 0;
            dvzdx = 0;
            };
            '''

            self.bd_code_vel += '''
            if (front_bd & right_bd) {
            dsigmayzdy = sigma_yz[i]/dy;
            dsigmaxydy = sigma_xy[i]/dy;
            dsigmaxxdx = -sigma_xx[i]/dx;
            };
            '''

        if self.F_BD == 'Free2' and self.L_BD == 'Free2':

            self.bd_code_str += '''
            if (front_bd & left_bd & (lx>1) & (ly>1)) {
            dvydy = -c12/(c11+c12)*(dvzdz);
            dvxdx = -c12/(c11+c12)*(dvzdz);
            };
            '''

            self.bd_code_vel += '''
            if (front_bd & left_bd & (lx>1) & (ly>1)) {
            dsigmayzdy = sigma_yz[i]/dy;
            dsigmaxydy = sigma_xy[i]/dy;
            dsigmaxzdx = sigma_xz[i]/dx;
            dsigmaxydx = sigma_xy[i]/dx;
            };
            '''

        if self.B_BD == 'Free2' and self.L_BD == 'Free2':

            self.bd_code_str += '''
            if (back_bd & left_bd & (lx>1) & (ly>1)) {
            dvxdx = -c12/(c11+c12)*(dvzdz);
            dvxdy = 0;
            dvzdy = 0;
            };
            '''

            self.bd_code_vel += '''
            if (back_bd & left_bd) {
            dsigmaxzdx = sigma_xz[i]/dx;
            dsigmaxydx = sigma_xy[i]/dx;
            dsigmayydy = -sigma_yy[i]/dy;
            };
            '''


        if self.X_BD == 'Free' and self.Y_BD == 'Free':
            self.bd_code_str += '''
            if (x_bd & y_bd) {
            dvxdx = -c12/(c11+c12)*(dvzdz);
            dvydy = -c12/(c11+c12)*(dvzdz);
            dvydx = 0;
            dvzdx = 0;
            dvxdy = 0;
            dvzdy = 0;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            sigmayy_bd = 0;
            sigmayz_bd = 0;

            };
            '''

        elif self.R_BD == 'Free' and self.Y_BD == 'Free':
            self.bd_code_str += '''
            if ('right_bd & y_bd') {
            dvxdx = -c12/(c11+c12)*(dvzdz);
            dvydy = -c12/(c11+c12)*(dvzdz);
            dvydx = 0;
            dvzdx = 0;
            dvxdy = 0;
            dvzdy = 0;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            sigmayy_bd = 0;
            sigmayz_bd = 0;

            };
            '''


        if self.X_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_str += '''

            if (x_bd & up_bd & (lz > 1)) {
            dvxdx = -c12/(c11+c12)*(dvydy);
            dvzdz = -c12/(c11+c12)*(dvydy);
            dvydx = 0;
            dvzdx = 0;
            dvxdz = 0;
            dvydz = 0;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            sigmazz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.R_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_str += '''

            if (right_bd & up_bd & (lz > 1)) {
            dvxdx = -c12/(c11+c12)*(dvydy);
            dvzdz = -c12/(c11+c12)*(dvydy);
            dvydx = 0;
            dvzdx = 0;
            dvxdz = 0;
            dvydz = 0;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            sigmazz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        if self.Y_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_str += '''

            if (y_bd & up_bd & (lz > 1)) {
            dvydy = -c12/(c11+c12)*(dvxdx);
            dvzdz = -c12/(c11+c12)*(dvxdx);
            dvxdy = 0;
            dvzdy = 0;
            dvxdz = 0;
            dvydz = 0;

            sigmayy_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmazz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

        if self.X_BD == 'Free' and self.Y_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_str += '''
            if (x_bd & y_bd & up_bd) {
            dvxdx = 0;
            dvydy = 0;
            dvzdz = 0;

            dvxdy = 0;
            dvzdy = 0;
            dvxdz = 0;
            dvydz = 0;
            dvzdx = 0;
            dvydx = 0;

            sigmaxx_bd = 0;
            sigmayy_bd = 0;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''
        elif self.R_BD == 'Free' and self.Y_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_str += '''
            if (right_bd & y_bd & up_bd) {
            dvxdx = 0;
            dvydy = 0;
            dvzdz = 0;

            dvxdy = 0;
            dvzdy = 0;
            dvxdz = 0;
            dvydz = 0;
            dvzdx = 0;
            dvydx = 0;

            sigmaxx_bd = 0;
            sigmayy_bd = 0;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

    # get data from gpu

    def get_vs(self):

        cl.enqueue_copy(self.queue, self.vx, self.vx_buf)
        cl.enqueue_copy(self.queue, self.vy, self.vy_buf)
        cl.enqueue_copy(self.queue, self.vz, self.vz_buf)
    def get_strains(self):

        cl.enqueue_copy(self.queue, self.sigmaxx, self.sigmaxx_buf)
        cl.enqueue_copy(self.queue, self.sigmayy, self.sigmayy_buf)
        cl.enqueue_copy(self.queue, self.sigmazz, self.sigmazz_buf)
        cl.enqueue_copy(self.queue, self.sigmaxy, self.sigmaxy_buf)
        cl.enqueue_copy(self.queue, self.sigmayz, self.sigmayz_buf)
        cl.enqueue_copy(self.queue, self.sigmaxz, self.sigmaxz_buf)
    def get_eps(self):

        cl.enqueue_copy(self.queue, self.eps_xx, self.eps_xx_buf)
        cl.enqueue_copy(self.queue, self.eps_yy, self.eps_yy_buf)
        cl.enqueue_copy(self.queue, self.eps_zz, self.eps_zz_buf)
        cl.enqueue_copy(self.queue, self.eps_xy, self.eps_xy_buf)
        cl.enqueue_copy(self.queue, self.eps_yz, self.eps_yz_buf)
        cl.enqueue_copy(self.queue, self.eps_xz, self.eps_xz_buf)

    # a better def of eps thru MEC  !
    # A great and terrible integrator lies here
    def compile(self):

        # OpenCL code of elastic and LLG integrators

        self.code = '''

        __kernel void update_velocity(
        __global float *time_val_arr,
        __global float *vx,       __global float *vy,       __global float *vz,
        __global float *sigma_xx, __global float *sigma_yy, __global float *sigma_zz,
        __global float *sigma_xy, __global float *sigma_yz, __global float *sigma_xz)

        {   int i = get_global_id(0);



            ''' + self.coord_syst_el + '''

            float dsigmaxxdx = (sigma_xx[r]-sigma_xx[i])/dx;
            float dsigmayydy = (sigma_yy[b]-sigma_yy[i])/dy;
            float dsigmazzdz = (sigma_zz[u]-sigma_zz[i])/dz;

            float dsigmaxydx = (sigma_xy[i]-sigma_xy[l])/dx;
            float dsigmaxydy = (sigma_xy[i]-sigma_xy[f])/dy;

            float dsigmaxzdx = (sigma_xz[i]-sigma_xz[l])/dx;
            float dsigmaxzdz = (sigma_xz[i]-sigma_xz[d])/dz;

            float dsigmayzdy = (sigma_yz[i]-sigma_yz[f])/dy;
            float dsigmayzdz = (sigma_yz[i]-sigma_yz[d])/dz;

            float fx = 0;
            float fy = 0;
            float fz = 0;

            ''' + self.bd_code_vel + '''

            float new_vx = vx[i] + (dsigmaxxdx + dsigmaxydy + dsigmaxzdz - Alpha * vx[i] + fx)*dt/rho;
            float new_vy = vy[i] + (dsigmaxydx + dsigmayydy + dsigmayzdz - Alpha * vy[i] + fy)*dt/rho;
            float new_vz = vz[i] + (dsigmaxzdx + dsigmayzdy + dsigmazzdz - Alpha * vz[i] + fz)*dt/rho;

            if (!isnan(vx_bd)) {new_vx = vx_bd;}
            if (!isnan(vy_bd)) {new_vy = vy_bd;}
            if (!isnan(vz_bd)) {new_vz = vz_bd;}

            barrier(CLK_GLOBAL_MEM_FENCE);

            vx[i] = new_vx;
            vy[i] = new_vy;
            vz[i] = new_vz;


        };


        __kernel void update_strains(
        __global float *time_val_arr,
        __global float *vx,       __global float *vy,       __global float *vz,
        __global float *sigma_xx, __global float *sigma_yy, __global float *sigma_zz,
        __global float *sigma_xy, __global float *sigma_yz, __global float *sigma_xz,
        __global float *eps_xx,   __global float *eps_yy,   __global float *eps_zz,
        __global float *eps_xy,   __global float *eps_yz,   __global float *eps_xz)

        {   int i = get_global_id(0);

            ''' + self.coord_syst_el + '''

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

            float epsxx = ((c11 + c12)*sigma_xx[i]-c12*(sigma_zz[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));
            float epsyy = ((c11 + c12)*sigma_yy[i]-c12*(sigma_xx[i] + sigma_zz[i]))/((c11 - c12)*(c11 + 2*c12));
            float epszz = ((c11 + c12)*sigma_zz[i]-c12*(sigma_xx[i] + sigma_yy[i]))/((c11 - c12)*(c11 + 2*c12));
            float epsxy = sigma_xy[i]/2/c44;
            float epsyz = sigma_yz[i]/2/c44;
            float epsxz = sigma_xz[i]/2/c44;

            eps_xx[i] = epsxx;
            eps_yy[i] = epsyy;
            eps_zz[i] = epszz;
            eps_xy[i] = epsxy;
            eps_yz[i] = epsyz;
            eps_xz[i] = epsxz;

            sigma_xx[i] = new_sigmaxx;
            sigma_yy[i] = new_sigmayy;
            sigma_zz[i] = new_sigmazz;
            sigma_xy[i] = new_sigmaxy;
            sigma_yz[i] = new_sigmayz;
            sigma_xz[i] = new_sigmaxz;


        };
        '''

        # build the Kernel
        self.prog = cl.Program(self.ctx, self.code).build()

    def dynamics(self):

        launch = self.prog.update_strains(self.queue, self.vx.shape, None,
        self.time_val_buf,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf,
        self.eps_xx_buf,  self.eps_yy_buf,  self.eps_zz_buf,
        self.eps_xy_buf,  self.eps_yz_buf,  self.eps_xz_buf)
        launch.wait()



        launch = self.prog.update_velocity(self.queue, self.vx.shape, None,
        self.time_val_buf,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf)
        launch.wait()


    def save_eps(self, dir, count):
        self.get_eps()
        self.eps_xx.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xx.dat')
        self.eps_yy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yy.dat')
        self.eps_zz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_zz.dat')
        self.eps_xy.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xy.dat')
        self.eps_yz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_yz.dat')
        self.eps_xz.astype('float32').tofile(dir + '/TXT/' + str(count) +'eps_xz.dat')
    def save_state(self, filename):

        self.get_vs()
        self.get_strains()

        np.save(filename + "_vx_.npy", self.vx)
        np.save(filename + "_vy_.npy", self.vy)
        np.save(filename + "_vz_.npy", self.vz)

        np.save(filename + "_sigmaxx_.npy", self.sigmaxx)
        np.save(filename + "_sigmayy_.npy", self.sigmayy)
        np.save(filename + "_sigmazz_.npy", self.sigmazz)
        np.save(filename + "_sigmaxy_.npy", self.sigmaxy)
        np.save(filename + "_sigmayz_.npy", self.sigmayz)
        np.save(filename + "_sigmaxz_.npy", self.sigmaxz)


    def load_state(self, filename):

        self.vx = np.load(filename + "_vx_.npy")
        self.vy = np.load(filename + "_vy_.npy")
        self.vz = np.load(filename + "_vz_.npy")

        self.sigmaxx = np.load(filename + "_sigmaxx_.npy")
        self.sigmayy = np.load(filename + "_sigmayy_.npy")
        self.sigmazz = np.load(filename + "_sigmazz_.npy")
        self.sigmaxy = np.load(filename + "_sigmaxy_.npy")
        self.sigmayz = np.load(filename + "_sigmayz_.npy")
        self.sigmaxz = np.load(filename + "_sigmaxz_.npy")

        self.vx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vx)
        self.vy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vy)
        self.vz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.vz)

        self.sigmaxx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxx)
        self.sigmayy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmayy)
        self.sigmazz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmazz)
        self.sigmaxy_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxy)
        self.sigmayz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmayz)
        self.sigmaxz_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.sigmaxz)


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
