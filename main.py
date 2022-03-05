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
from scipy.fft import fft, fftfreq

class ME:

    def __init__(self):

        self.time_mx = []
        self.time_my = []
        self.time_mz = []
        self.time_epsxx = []
        self.time_epsyy = []
        self.time_epszz = []
        self.time = []
        self.comp = 0
    # initiate geometry and heterostructure
    def init_structure(self,interfaces,Lx,Ly,Lz,dx,dy,dz,dt,static=False):

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.mag_mat = 'CoFe'

        if static == False:

            Beta = 1000                 #Damping EL
        else:

            Beta = 10**16

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.Beta =  Beta

        self.interfaces = interfaces


        self.coord_syst_el = '''

        const int lx = get_global_size(2);
        const int ly = get_global_size(1);
        const int lz = get_global_size(0);

        const int pl = lx * ly;
        int x = get_global_id(2);
        int y = get_global_id(1);
        int z = get_global_id(0);
        int i = x + y * lx + z * pl;

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

        self.coord_syst_el += '''



        bool left_bd =  (x == 0);
        bool right_bd = (x == lx-1);
        bool front_bd = (y == 0);
        bool back_bd =  (y == ly-1);
        bool down_bd =  (z == 0);
        bool up_bd =    (z == lz-1);

        bool x_bd = right_bd || left_bd;
        bool y_bd = back_bd  || front_bd;
        bool z_bd = up_bd    || down_bd;

        bool pochti_left_bd =  (x == 1);
        bool pochti_right_bd = (x == lx-2);
        bool pochti_front_bd = (y == 1);
        bool pochti_back_bd =  (y == ly-2);
        bool pochti_down_bd =  (z == 1);
        bool pochti_up_bd =    (z == lz-2);
        '''


        self.coord_syst_el += '''

        float Alpha = '''+str(self.Beta)+''';

        const float small = 0;
        const float big = 1/small;
        const float rho_vac = 1;
        float rhoz;

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

        const float c11FeGa = 162e9;
        const float c12FeGa = 124e9;
        const float c44FeGa = 126e9;
        const float rhoFeGa = 7800;

        const float B1FeGa =  -9e6;
        const float B2FeGa =  -8e6;

        const float aFeGa = 3.0;

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

        #self.bd_code_vel = ''''''

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



        for material in interfaces:

            self.coord_syst_el += 'if ((z >= ' + str(interfaces[material][0]) + ') & (z <= ' + str(interfaces[material][1]) + ''')) {
            c11 = c11'''+str(material)+''';
            c12 = c12'''+str(material)+''';
            c44 = c44'''+str(material)+''';
            rho = rho'''+str(material)+''';
            rhoz = rho;

            B1 =  B1'''+str(material)+''';
            B2 =  B2'''+str(material)+''';

            eps_m = (aMgO-aCoFe)/aCoFe;
            eps_m = 0;

            };
            '''


        self.vx = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.vy = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.vz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)

        self.sigmaxx = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.sigmayy = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.sigmazz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.sigmaxy = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.sigmayz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.sigmaxz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)

        self.eps_xx = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.eps_yy = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.eps_zz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.eps_xy = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.eps_yz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)
        self.eps_xz = np.zeros((self.Lz,self.Ly,self.Lx)).astype(np.float32)

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

        # misfits
        m = 0
        interfaces = self.interfaces
        K = list(interfaces.keys())
        for material in interfaces:

            if m!=0:

                #displacements
                self.bd_code_vel += '''
                if ((z == '''+str(interfaces[material][0]) + ''') & !(x_bd | y_bd)) {
                dsigmaxzdz -= -c44'''+K[m-1]+'''*((vx[i]*dt+(x-lx/2)*dx)*(1-a'''+K[m]+'''/a'''+K[m-1]+''')/dz)/dz;
                dsigmayzdz -= -c44'''+K[m-1]+'''*((vy[i]*dt+(y-ly/2)*dy)*(1-a'''+K[m]+'''/a'''+K[m-1]+''')/dz)/dz;
                };
                '''

            if m!=len(K)-1:

                self.bd_code_vel  +='if ((z == '+str(interfaces[material][1]) + ''')) {
                rhoz = (rho'''+ K[m] +'''+ rho'''+ K[m+1] + ''')/2;
                };
                '''

                self.bd_code_str += '''
                if ((z == '''+str(interfaces[material][1]) + ''') & !(x_bd | y_bd)) {
                dvxdz -= vx[u]*(1 - a'''+K[m+1]+'''/a'''+K[m]+''')/dz;
                dvydz -= vy[u]*(1 - a'''+K[m+1]+'''/a'''+K[m]+''')/dz;

                //dvxdz -= ((vx[u]+(x-lx/2)*dx/dt)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dz;
                //dvydz -= ((vy[u]+(y-ly/2)*dy/dt)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dz;
                };
                '''

                self.bd_code_vel += '''
                if ((z == '''+str(interfaces[material][1]) + ''') & !(x_bd | y_bd)) {
                dsigmaxzdz -= c44*((vx[u]*dt+(x-lx/2)*dx)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dz;
                dsigmayzdz -= c44*((vy[u]*dt+(y-ly/2)*dy)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dz;

                dsigmaxzdx -= c44*((vx[u]*dt-vx[u-1]*dt+dx)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dx;
                dsigmayzdy -= c44*((vy[u]*dt-vy[u-1]*dt+dy)*(1-a'''+K[m+1]+'''/a'''+K[m]+''')/dz)/dy;

                };
                '''

            m+=1

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

            //sigmazz_bd = 0;
            //sigmaxz_bd = 0;
            //sigmayz_bd = 0;
            };
            '''

            self.bd_code_vel += '''
            if (up_bd) {
            dsigmazzdz = 0;
            dsigmaxzdz = (-sigma_xz[d])/dz;
            dsigmayzdz = (-sigma_yz[d])/dz;

            dsigmaxzdx = 0;
            dsigmayzdy = 0;
            };

            if (pochti_up_bd) {
            dsigmazzdz = -sigma_zz[i]/dz;
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



        elif self.U_BD == 'Free2':

            self.bd_code_str += '''
            if (up_bd & (lz>1)) {
            c11 = small;
            c12 = small;
            c44 = small;
            rho = small;

            sigmazz_bd = -sigma_zz[d];
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
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
            sigmaxz_bd = 0;
            sigmayz_bd = 0;
            };
            '''

        elif self.U_BD == 'z_stressed2':

            self.bd_code_vel += '''
            if ((up_bd) & (lz>1)) {
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


        elif self.U_BD == 'Fixed':

            self.bd_code_vel += '''
            if (down_bd & (lz>1)) {
            vx_bd = 0;
            vy_bd = 0;
            vz_bd = 0;
            };
            '''
            self.bd_code_str += '''
            if (down_bd) {

            sigmaxx_bd = 0;
            sigmayy_bd = 0;
            sigmazz_bd = 0;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            };
            '''


        # lower bound managment

        if self.D_BD == 'Free':

            self.bd_code_str += '''
            if (down_bd) {
            dvzdz = -c12/c11*(dvxdx+dvydy);
            dvxdz = -dvzdx;
            dvydz = -dvzdy;

            //sigmazz_bd = 0;
            //sigmaxz_bd = 0;
            //sigmayz_bd = 0;
            };
            '''

            self.bd_code_vel += '''
            if (down_bd) {
            dsigmazzdz = (sigma_zz[u])/dz;
            dsigmaxzdz = 0;
            dsigmayzdz = 0;

            dsigmaxzdx = 0;
            dsigmayzdy = 0;
            };

            if (pochti_down_bd) {
            dsigmaxzdz = sigma_xz[i]/dz;
            dsigmayzdz = sigma_yz[i]/dz;
            };
            '''

        elif self.D_BD == 'Open':

            self.bd_code_vel += '''
            if (down_bd & (lx>1)) {
            dsigmaxzdz = sqrt(c11*rho)*(vx[u]-vx[i])/dz;
            dsigmayzdz = sqrt(c11*rho)*(vy[u]-vy[i])/dz;
            };
            '''

            self.bd_code_str += '''
            if (down_bd & (lx>1)) {
            dvzdz = (sigma_zz[u]-sigma_zz[i])/dz/sqrt(c44*rho);
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
            vx_bd = 0;
            vy_bd = 0;
            //vz_bd = 0;
            };
            '''

        elif self.D_BD == 'Fixed2':

            self.bd_code_vel += '''
            if (down_bd & (lz>1)) {
            vz_bd = 0;
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

            sigmaxx_bd = 0.0001*c12*sin(time_val*2*3.1415*dt*2.8*1e9);
            sigmayy_bd = 0.0001*c12*sin(time_val*2*3.1415*dt*2.8*1e9);
            sigmazz_bd = 0.0001*c11*sin(time_val*2*3.1415*dt*2.8*1e9);
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;

            };
            '''

        elif self.D_BD == 'Impulse':

            self.bd_code_str += '''

            if (down_bd) {

            const float imp_len = 1e-11;

            if (time_val*dt < imp_len) {

            sigmaxx_bd = 0.001*c12;
            sigmayy_bd = 0.001*c12;
            sigmazz_bd = 0.001*c11;
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

        elif self.D_BD == 'Bipolar_Impulse':

            self.bd_code_str += '''

            if (down_bd) {

            const float imp_len = 1e-11;
            const float amp = 10*0.5*1e-3/6;
            const float dur = 0.02;
            const float center = 4*dur;

            if (2*time_val*dt*1e9 < dur) {

            sigmaxx_bd = amp*exp(-(-center+2*time_val*dt*1e9)*(-center+2*time_val*dt*1e9)/2/dur/dur)*(-center+2*time_val*dt*1e9)/dur/dur*c12;
            sigmayy_bd = amp*exp(-(-center+2*time_val*dt*1e9)*(-center+2*time_val*dt*1e9)/2/dur/dur)*(-center+2*time_val*dt*1e9)/dur/dur*c12;
            sigmazz_bd = amp*exp(-(-center+2*time_val*dt*1e9)*(-center+2*time_val*dt*1e9)/2/dur/dur)*(-center+2*time_val*dt*1e9)/dur/dur*c11;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            }

            else {

            dvzdz = (sigma_zz[u]-sigma_zz[i])/dz/sqrt(c44*rho);
            };
            };
            '''

            self.bd_code_vel += '''

            if (down_bd) {

            const float imp_len = 0.1e-9;

            if (time_val*dt >= imp_len) {

            //vx_bd = 0;
            //vy_bd = 0;
            //vz_bd = 0;

            }
            else {
            dsigmaxzdz = sqrt(c11*rho)*(vx[u]-vx[i])/dz;
            dsigmayzdz = sqrt(c11*rho)*(vy[u]-vy[i])/dz;
            };
            };

            '''

        elif self.D_BD == 'Impulses':

            self.bd_code_str += '''

            if (down_bd) {

            const float imp_len = 1e-11;
            bool imp = True

            if (time_val*dt % imp_len == 0) {
            imp = !imp;

            if (imp) {

            sigmaxx_bd = 0.01*c12;
            sigmayy_bd = 0.01*c12;
            sigmazz_bd = 0.01*c11;
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;
            }
            else {

            dvzdz = (sigma_zz[u]-sigma_zz[i])/dz/sqrt(c44*rho);

            }
            };

            '''

            self.bd_code_vel += '''

            if (down_bd) {

            const float imp_len = 0.1e-9;

            bool imp = True

            if (time_val*dt % imp_len == 0) {
            imp = !imp;

            if (imp) {

            vx_bd = 0;
            vy_bd = 0;
            vz_bd = 0;

            }
            else {
            dsigmaxzdz = sqrt(c11*rho)*(vx[u]-vx[i])/dz;
            dsigmayzdz = sqrt(c11*rho)*(vy[u]-vy[i])/dz;

            }
            };

            '''





        elif self.D_BD == '1D_Emmiter':

            self.bd_code_str += '''

            if (down_bd) {

            sigmaxx_bd = 0.0001*c12*sin(time_val*2*3.1415*dt*2.74e9);
            sigmayy_bd = 0.0001*c12*sin(time_val*2*3.1415*dt*2.74e9);
            sigmazz_bd = 0.0001*c11*sin(time_val*2*3.1415*dt*2.74e9);
            sigmaxy_bd = 0;
            sigmayz_bd = 0;
            sigmaxz_bd = 0;

            };
            '''

        # right bound managment

        if self.R_BD == 'Free':

            self.bd_code_str += '''
            if (right_bd) {
            dvxdx = -c12/c11*(dvydy+dvzdz);
            dvydx = -dvxdy;
            dvzdx = -dvxdz;

            //sigmaxx_bd = 0;
            //sigmaxy_bd = 0;
            //sigmaxz_bd = 0;
            };
            '''

            #self.bd_code_vel += '''
            '''if (right_bd) {
            dsigmaxxdx = 0;
            dsigmaxydx = (-sigma_xy[l])/dx;
            dsigmaxzdx = (-sigma_xz[l])/dx;

            dsigmaxydy = 0;
            dsigmaxzdz = 0;
            };

            if (pochti_right_bd) {
            dsigmaxxdx = -sigma_xx[i]/dx;
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
            dvydx = - dvxdy;
            dvzdx = - dvxdz;

            sigmaxx_bd = 0;
            sigmaxy_bd = 0;
            sigmaxz_bd = 0;
            };
            '''

            #self.bd_code_vel += '''
            '''if (left_bd) {
            dsigmaxxdx = (sigma_xx[r])/dx;
            dsigmaxydx = 0;
            dsigmaxzdx = 0;

            dsigmaxydy = 0;
            dsigmaxzdz = 0;
            };

            if (pochti_left_bd) {
            dsigmaxydx = sigma_xy[i]/dx;
            dsigmaxzdx = sigma_xz[i]/dx;
            };
            '''

        elif self.L_BD == 'Free2':

            self.bd_code_str += '''
            if (left_bd & (lx>1)) {
            c11 = small;
            rho = small;

            sigmaxx_bd = -sigma_xx[r];
            sigmaxz_bd = 0;
            sigmaxy_bd = 0;
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

            //sigmayy_bd = 0;
            //sigmaxy_bd = 0;
            //sigmayz_bd = 0;
            };
            '''

            self.bd_code_vel += '''
            if (back_bd) {
            dsigmayydy = 0;
            dsigmaxydy = (-sigma_xy[f])/dy;
            dsigmayzdy = (-sigma_yz[f])/dy;

            dsigmaxydx = 0;
            dsigmayzdz = 0;

            };

            if (pochti_back_bd) {
            dsigmayydy = -sigma_yy[i]/dy;
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

            //sigmayy_bd = 0;
            //sigmaxy_bd = 0;
            //sigmayz_bd = 0;
            };
            '''

            self.bd_code_vel += '''
            if (front_bd) {
            dsigmayydy = (sigma_yy[b])/dy;
            dsigmaxydy = 0;
            dsigmayzdy = 0;

            dsigmaxydx = 0;
            dsigmayzdz = 0;
            };

            if (pochti_front_bd) {
            dsigmaxydy = sigma_xy[i]/dy;
            dsigmayzdy = sigma_yz[i]/dy;
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

            self.bd_code_vel += '''
            if (front_bd & (ly>1)) {
            dsigmaxydy = sqrt(c11*rho)*(vx[b]-vx[i])/dy;
            dsigmayzdy = sqrt(c11*rho)*(vz[b]-vz[i])/dy;
            };
            '''

            self.bd_code_str += '''
            if (front_bd & (ly>1)) {
            dvydy = (sigma_yy[b]-sigma_yy[i])/dx/sqrt(c44*rho);
            };
            '''

        if self.U_BD == 'Open':

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

        if self.D_BD == 'Open':

            self.bd_code_vel += '''
            if (down_bd & (lx>1)) {
            dsigmaxzdz = sqrt(c11*rho)*(vx[u]-vx[i])/dz;
            dsigmayzdz = sqrt(c11*rho)*(vy[u]-vy[i])/dz;
            };
            '''

            self.bd_code_str += '''
            if (down_bd & (lx>1)) {
            dvzdz = (sigma_zz[u]-sigma_zz[i])/dz/sqrt(c44*rho);
            };
            '''



        if self.X_BD == 'Free' and self.Y_BD == 'Free':
            self.bd_code_vel += '''
            if (x_bd & y_bd) {
            dsigmaxydx = 0;
            dsigmaxydy = 0;
            dsigmayzdy = 0;
            dsigmayzdz = 0;
            dsigmaxzdx = 0;
            dsigmaxzdz = 0;
            };
            '''

        if self.X_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_vel += '''
            if (x_bd & up_bd) {
            dsigmaxydx = 0;
            dsigmaxydy = 0;
            dsigmaxzdx = 0;

            dsigmaxydy = 0;
            dsigmaxzdz = 0;

            };
            '''
        if self.Y_BD == 'Free' and self.U_BD == 'Free':
            self.bd_code_vel += '''
            if (x_bd & up_bd) {
            dsigmaxydx = 0;
            dsigmaxydy = 0;
            dsigmaxzdx = 0;

            dsigmaxydy = 0;
            dsigmaxzdz = 0;

            };
            '''

        #elif self.R_BD == 'Free' and self.Y_BD == 'Free':
        #    self.bd_code_str += '''
        #    if (right_bd & y_bd) {
        #    dvxdx = -c12/(c11+c12)*(dvzdz);
        #    dvydy = -c12/(c11+c12)*(dvzdz);
        #    dvydx = 0;
        #    dvzdx = 0;
        #    dvxdy = 0;
        #    dvzdy = 0;

        #    sigmaxx_bd = 0;
        #    sigmaxy_bd = 0;
        #    sigmaxz_bd = 0;
        #    sigmayy_bd = 0;
        #    sigmayz_bd = 0;

        #    };
        #    '''


        #if self.X_BD == 'Free' and self.U_BD == 'Free':
        #    self.bd_code_str += '''
#
#            if (x_bd & up_bd & (lz > 1)) {
#            dvxdx = -c12/(c11+c12)*(dvydy);
#            dvzdz = -c12/(c11+c12)*(dvydy);
#            dvydx = 0;
#            dvzdx = 0;
#            dvxdz = 0;
#            dvydz = 0;

#            sigmaxx_bd = 0;
#            sigmaxy_bd = 0;
#            sigmaxz_bd = 0;
#            sigmazz_bd = 0;
#            sigmayz_bd = 0;
#            };
#            '''

#        elif self.R_BD == 'Free' and self.U_BD == 'Free':
#            self.bd_code_str +=
#            '''

#            if (right_bd & up_bd & (lz > 1)) {
#            dvxdx = -c12/(c11+c12)*(dvydy);
#            dvzdz = -c12/(c11+c12)*(dvydy);
#            dvydx = 0;
#            dvzdx = 0;
#            dvxdz = 0;
#            dvydz = 0;

#            sigmaxx_bd = 0;
#            sigmaxy_bd = 0;
#            sigmaxz_bd = 0;
#            sigmazz_bd = 0;
#            sigmayz_bd = 0;
#            };
#            '''

        #if self.Y_BD == 'Free' and self.U_BD == 'Free':
        #    self.bd_code_str +=
#            '''

#            if (y_bd & up_bd & (lz > 1)) {
#           dvydy = -c12/(c11+c12)*(dvxdx);
#          dvzdz = -c12/(c11+c12)*(dvxdx);
#          dvxdy = 0;
#           dvzdy = 0;
#           dvxdz = 0;
#           dvydz = 0;

#           sigmayy_bd = 0;
#           sigmaxy_bd = 0;
#           sigmayz_bd = 0;
#            sigmazz_bd = 0;
#           sigmaxz_bd = 0;
#            };
#            '''

        #if self.X_BD == 'Free' and self.Y_BD == 'Free' and self.U_BD == 'Free':
            #self.bd_code_str +=
#            '''
#            if (x_bd & y_bd & up_bd) {
#            dvxdx = 0;
#            dvydy = 0;
#            dvzdz = 0;

#            dvxdy = 0;
#            dvzdy = 0;
#            dvxdz = 0;
#            dvydz = 0;
#            dvzdx = 0;
#            dvydx = 0;

#            sigmaxx_bd = 0;
#            sigmayy_bd = 0;
#            sigmazz_bd = 0;
#            sigmaxy_bd = 0;
#            sigmayz_bd = 0;
#            sigmaxz_bd = 0;
#            };
#            '''
        #elif self.R_BD == 'Free' and self.Y_BD == 'Free' and self.U_BD == 'Free':
            #self.bd_code_str += '''
            #if (right_bd & y_bd & up_bd) {
            #dvxdx = 0;
            #dvydy = 0;
            #dvzdz = 0;

            #dvxdy = 0;
            #vzdy = 0;
            #dvxdz = 0;
            #dvydz = 0;
            #dvzdx = 0;
            #dvydx = 0;

            #sigmaxx_bd = 0;
            #sigmayy_bd = 0;
            #sigmazz_bd = 0;
            #sigmaxy_bd = 0;
            #sigmayz_bd = 0;
            #sigmaxz_bd = 0;
            #};
            #'''



    # get data from gpu

    def get_vs(self):

        self.vx = self.vx_buf.get()
        self.vy = self.vy_buf.get()
        self.vz = self.vz_buf.get()
    def get_strains(self):

        self.sigmaxx = self.sigmaxx_buf.get()
        self.sigmayy = self.sigmayy_buf.get()
        self.sigmazz = self.sigmazz_buf.get()
        self.sigmaxz = self.sigmaxz_buf.get()
        self.sigmaxy = self.sigmaxy_buf.get()
        self.sigmayz = self.sigmayz_buf.get()
    def get_eps(self):
        self.eps_xx = self.eps_xx_buf.get()
        self.eps_yy = self.eps_yy_buf.get()
        self.eps_zz = self.eps_zz_buf.get()
        self.eps_xz = self.eps_xz_buf.get()
        self.eps_xy = self.eps_xy_buf.get()
        self.eps_yz = self.eps_yz_buf.get()

    # a better def of eps thru MEC  !
    # A great and terrible integrator lies here
    def compile(self):

        if self.comp > 0:
            self.get_vs()
            self.get_strains()
        self.comp += 1

        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        #self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread(self.queue)
        print(cl.get_platforms()[0].get_devices())#[0].max_compute_units

        self.time_val_buf = cl_array.to_device(self.queue, np.zeros(1).astype(np.float32))

        self.vx_buf = cl_array.to_device(self.queue, self.vx)
        self.vy_buf = cl_array.to_device(self.queue, self.vy)
        self.vz_buf = cl_array.to_device(self.queue, self.vz)

        self.sigmaxx_buf = cl_array.to_device(self.queue, self.sigmaxx)
        self.sigmayy_buf = cl_array.to_device(self.queue, self.sigmayy)
        self.sigmazz_buf = cl_array.to_device(self.queue, self.sigmazz)
        self.sigmaxy_buf = cl_array.to_device(self.queue, self.sigmaxy)
        self.sigmayz_buf = cl_array.to_device(self.queue, self.sigmayz)
        self.sigmaxz_buf = cl_array.to_device(self.queue, self.sigmaxz)

        self.eps_xx_buf = cl_array.to_device(self.queue, self.eps_xx)
        self.eps_yy_buf = cl_array.to_device(self.queue, self.eps_yy)
        self.eps_zz_buf = cl_array.to_device(self.queue, self.eps_zz)
        self.eps_xy_buf = cl_array.to_device(self.queue, self.eps_xy)
        self.eps_yz_buf = cl_array.to_device(self.queue, self.eps_yz)
        self.eps_xz_buf = cl_array.to_device(self.queue, self.eps_xz)


        # OpenCL code of elastic and LLG integrators

        self.code = '''

        #include <pyopencl-complex.h>

        KERNEL void update_velocity(
        GLOBAL_MEM float *time_val_arr,
        GLOBAL_MEM float *vx,       GLOBAL_MEM float *vy,       GLOBAL_MEM float *vz,
        GLOBAL_MEM float *sigma_xx, GLOBAL_MEM float *sigma_yy, GLOBAL_MEM float *sigma_zz,
        GLOBAL_MEM float *sigma_xy, GLOBAL_MEM float *sigma_yz, GLOBAL_MEM float *sigma_xz,
        GLOBAL_MEM float *eps_xx,   GLOBAL_MEM float *eps_yy,   GLOBAL_MEM float *eps_zz,
        GLOBAL_MEM float *eps_xy,   GLOBAL_MEM float *eps_yz,   GLOBAL_MEM float *eps_xz)

        {   ''' + self.coord_syst_el + '''

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



            if (lx==1) {
            dsigmaxxdx = 0;
            dsigmaxydx = 0;
            dsigmaxzdx = 0;
            };

            if (ly==1) {
            dsigmayydy = 0;
            dsigmaxydy = 0;
            dsigmayzdy = 0;
            };

            if (lz==1) {
            dsigmazzdz = 0;
            dsigmayzdz = 0;
            dsigmaxzdz = 0;
            };

            float new_vx = vx[i] + (dsigmaxxdx + dsigmaxydy + dsigmaxzdz - Alpha * vx[i] + fx)*dt/rho*2;
            float new_vy = vy[i] + (dsigmaxydx + dsigmayydy + dsigmayzdz - Alpha * vy[i] + fy)*dt/rho*2;
            float new_vz = vz[i] + (dsigmaxzdx + dsigmayzdy + dsigmazzdz - Alpha * vz[i] + fz)*dt/rhoz*2;

            if (!isnan(vx_bd)) {new_vx = vx_bd;}
            if (!isnan(vy_bd)) {new_vy = vy_bd;}
            if (!isnan(vz_bd)) {new_vz = vz_bd;}

            vx[i] = new_vx;
            vy[i] = new_vy;
            vz[i] = new_vz;

            barrier(CLK_GLOBAL_MEM_FENCE);

        };



        KERNEL void update_strains(
        GLOBAL_MEM float *time_val_arr,
        GLOBAL_MEM float *vx,       GLOBAL_MEM float *vy,       GLOBAL_MEM float *vz,
        GLOBAL_MEM float *sigma_xx, GLOBAL_MEM float *sigma_yy, GLOBAL_MEM float *sigma_zz,
        GLOBAL_MEM float *sigma_xy, GLOBAL_MEM float *sigma_yz, GLOBAL_MEM float *sigma_xz)

        {
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

            if (lx==1) {
            dvxdx = 0;
            dvydx = 0;
            dvzdx = 0;
            };

            if (ly==1) {
            dvxdy = 0;
            dvydy = 0;
            dvzdy = 0;
            };

            if (lz==1) {
            dvxdz = 0;
            dvydz = 0;
            dvzdz = 0;
            };


            float dsigmaxxdt = c11 * dvxdx + c12 * (dvydy + dvzdz);
            float dsigmayydt = c11 * dvydy + c12 * (dvxdx + dvzdz);
            float dsigmazzdt = c11 * dvzdz + c12 * (dvydy + dvxdx);

            float dsigmaxydt = c44 * (dvxdy+dvydx);
            float dsigmayzdt = c44 * (dvzdy+dvydz);
            float dsigmaxzdt = c44 * (dvxdz+dvzdx);

            float new_sigmaxx = sigma_xx[i] + dsigmaxxdt * dt*2;
            float new_sigmayy = sigma_yy[i] + dsigmayydt * dt*2;
            float new_sigmazz = sigma_zz[i] + dsigmazzdt * dt*2;
            float new_sigmaxy = sigma_xy[i] + dsigmaxydt * dt*2;
            float new_sigmayz = sigma_yz[i] + dsigmayzdt * dt*2;
            float new_sigmaxz = sigma_xz[i] + dsigmaxzdt * dt*2;

            if (!isnan(sigmaxx_bd)) {new_sigmaxx = sigmaxx_bd;}
            if (!isnan(sigmayy_bd)) {new_sigmayy = sigmayy_bd;}
            if (!isnan(sigmazz_bd)) {new_sigmazz = sigmazz_bd;}
            if (!isnan(sigmaxy_bd)) {new_sigmaxy = sigmaxy_bd;}
            if (!isnan(sigmayz_bd)) {new_sigmayz = sigmayz_bd;}
            if (!isnan(sigmaxz_bd)) {new_sigmaxz = sigmaxz_bd;}




            //float ux = vx[i]*dt;
            //float uy = vy[i]*dt;
            //float uz = vz[i]*dt;



            sigma_xx[i] = new_sigmaxx;
            sigma_yy[i] = new_sigmayy;
            sigma_zz[i] = new_sigmazz;
            sigma_xy[i] = new_sigmaxy;
            sigma_yz[i] = new_sigmayz;
            sigma_xz[i] = new_sigmaxz;

            barrier(CLK_GLOBAL_MEM_FENCE);


        };

        '''

        # build the Kernel
        self.prog = self.thr.compile(self.code)

    def dynamics(self):


        self.prog.update_strains(
        self.time_val_buf,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf,
        global_size=(self.Lz,self.Ly,self.Lx))


        self.proc_to_kill = self.prog.update_velocity(
        self.time_val_buf,
        self.vx_buf,      self.vy_buf,      self.vz_buf,
        self.sigmaxx_buf, self.sigmayy_buf, self.sigmazz_buf,
        self.sigmaxy_buf, self.sigmayz_buf, self.sigmaxz_buf,
        self.eps_xx_buf,  self.eps_yy_buf,  self.eps_zz_buf,
        self.eps_xy_buf,  self.eps_yz_buf,  self.eps_xz_buf,
        global_size=(self.Lz,self.Ly,self.Lx))

    def save_math(self,dir, count):
        self.get_eps()

        self.eps_xx.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_xx.dat')
        self.eps_yy.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_yy.dat')
        self.eps_zz.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_zz.dat')
        self.eps_xy.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_xy.dat')
        self.eps_yz.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_yz.dat')
        self.eps_xz.astype('float32').tofile(dir + '/MATH/' + str(count) +'eps_xz.dat')

    def save_state(self, screen):
        self.get_vs()
        self.get_strains()
        #self.get_eps()


        np.save(f"{self.directory}{screen}_sigmaxx_.npy", self.sigmaxx)
        np.save(f"{self.directory}{screen}_sigmayy_.npy", self.sigmayy)
        np.save(f"{self.directory}{screen}_sigmazz_.npy", self.sigmazz)
        np.save(f"{self.directory}{screen}_sigmaxy_.npy", self.sigmaxy)
        np.save(f"{self.directory}{screen}_sigmayz_.npy", self.sigmayz)
        np.save(f"{self.directory}{screen}_sigmaxz_.npy", self.sigmaxz)

        np.save(f"{self.directory}{screen}_epsxx_.npy", self.eps_xx)
        np.save(f"{self.directory}{screen}_epsyy_.npy", self.eps_yy)
        np.save(f"{self.directory}{screen}_epszz_.npy", self.eps_zz)
        np.save(f"{self.directory}{screen}_epsxy_.npy", self.eps_xy)
        np.save(f"{self.directory}{screen}_epsyz_.npy", self.eps_yz)
        np.save(f"{self.directory}{screen}_epsxz_.npy", self.eps_xz)
    def load_state(self, filename, zero_el = False):

        self.vx = np.load(filename + "_vx_.npy")
        self.vy = np.load(filename + "_vy_.npy")
        self.vz = np.load(filename + "_vz_.npy")

        self.sigmaxx = np.load(filename + "_sigmaxx_.npy")
        self.sigmayy = np.load(filename + "_sigmayy_.npy")
        self.sigmazz = np.load(filename + "_sigmazz_.npy")
        self.sigmaxy = np.load(filename + "_sigmaxy_.npy")
        self.sigmayz = np.load(filename + "_sigmayz_.npy")
        self.sigmaxz = np.load(filename + "_sigmaxz_.npy")

    def plot_xy_pl(self, stringer, layer, count):

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

        pl = a[layer]
        fig, ax = plt.subplots()
        plt.contourf(Xpos, Ypos, pl, cmap=plt.get_cmap('plasma'),
                     levels=MaxNLocator(nbins=100).tick_values(pl.min(), pl.max()))
        plt.colorbar(label=lab, format='%.20f')
        ax.set_aspect('equal', 'box')
        plt.ticklabel_format(useOffset=False)
        plt.tick_params(labelleft=False)
        font = 15

        plt.title(f'Deformations {lab}')

        plt.xlabel("Coordinate x (nm)", fontsize=font)
        plt.ylabel("Coordinate y (nm)", fontsize=font)
        fig.set_size_inches(8, 8)
        plt.savefig(self.directory + f"/film/{stringer}{count}_layer={layer}.png", dpi=100)
        plt.close()
    def plot_1D_xy(self, screen, point, layer):

        self.get_eps()

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dx = self.dx
        dy = self.dy
        dz = self.dz

        a1 = 100*self.eps_xx
        a2 = 100*self.eps_yy
        a3 = 100*self.eps_zz

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

        plt.savefig(self.directory + f"/film/y{screen}.png", dpi=100)
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

        plt.savefig(self.directory + f"/film/x{screen}.png", dpi=100)
        plt.close()
    def plot_1D_z(self, screen):

        self.get_eps()

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dz = self.dz

        a1 = self.eps_xx
        a2 = self.eps_yy
        a3 = self.eps_zz

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

        plt.savefig(self.directory + f"/film/z{screen}.png", dpi=100)
        plt.close()

    def plot_1D_z_shear(self, file_name):

        self.get_eps()

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz

        dz = self.dz

        a1 = self.eps_zz
        a2 = self.eps_xz
        a3 = self.eps_yz

        pl1 = []
        pl2 = []
        pl3 = []

        x = int(Lx/2)
        y = int(Ly/2)
        for z in range(0,Lz,1):
            pl1.append(a1[z][y][x]*10**2*0)
            pl2.append(a2[z][y][x]*10**2)
            pl3.append(a3[z][y][x]*10**2)

        pl1 = np.array(pl1)
        pl2 = np.array(pl2)
        pl3 = np.array(pl3)

        fig, ax = plt.subplots()
        t = np.arange(0, dz * Lz, dz)
        ax.plot(t, pl1,"b",   label="$\epsilon_{zz}$")
        ax.plot(t, pl2,"r--", label="$\epsilon_{xz}$")
        ax.plot(t, pl3, "g",  label="$\epsilon_{yz}$")
        ax.legend(loc='lower left')
        ax.set(xlabel='z coordinate (nm)', ylabel='Mechanical strains (%)')
        ax.grid()

        plt.savefig(file_name, dpi=100)
        plt.close()
