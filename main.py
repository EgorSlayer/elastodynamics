import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)



def init_data(Lx,Ly,Lz,dx,dy,dz,dt,Alpha,c11,c12,c44,rho,Eps_xx,Eps_yy):

    L = Lx * Ly * Lz
    c11_arr = np.full(L, c11).astype(np.float32)
    c12_arr = np.full(L, c12).astype(np.float32)
    c44_arr = np.full(L, c44).astype(np.float32)
    rho_arr = np.full(L, rho).astype(np.float32)
    out = np.zeros(12 * L).astype(np.float32)


    consts = {'Lx' : np.float32(Lx),'Ly' : np.float32(Ly),'Lz' : np.float32(Lz),'L' : np.float32(L),
              'dx' : np.float32(dx),'dy' : np.float32(dy),'dz' : np.float32(dz),'dt':np.float32(dt),
    'c11':c11_arr,'c12':c12_arr,'c44':c44_arr,'rho':rho_arr,
    'out':out,'Eps_xx':np.float32(Eps_xx),'Eps_yy':np.float32(Eps_yy),'Alpha':np.float32(Alpha)}

    u = np.zeros(L).astype(np.float32)

    data = {"u1":u, "u2":u, "u3":u, 'v1':u, "v2":u, 'v3':u}

    return  data, consts



# OpenCL elastic

code_el = """

float eps_ii(float pos,float neg,float di)
    {
    float eps = native_divide(pos-neg,di);
    return eps;
    }

float didj(float pos,float neg,float dj)
    {
    float eps = native_divide(pos-neg,dj);
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


__kernel void my_el(
__global const float *u1, __global const float *u2, __global const float *u3,
__global const float *v1, __global const float *v2, __global const float *v3,
__global const float *c11, __global const float *c12, __global const float *c44, __global const float *rho,
const float Lx, const float Ly, const float Lz, const float L, const float Dt, const float Alpha,
const float Dx, const float Dy, const float Dz, const float Eps_xx,  const float Eps_yy,
__global float *out)

{   int i = get_global_id(0);
    int len = L;

    if (i <= len-1) {
    int lx = Lx;
    int ly = Ly;
    int lz = Lz;
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

    float dt = Dt * native_powr(10, -9);
    float dx = Dx * native_powr(10, -9);
    float dy = Dy * native_powr(10, -9);
    float dz = Dz * native_powr(10, -9);
    float dt2 = native_powr(dt, 2);
    float dx2 = native_powr(dx, 2);
    float dy2 = native_powr(dy, 2);
    float dz2 = native_powr(dz, 2);

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

    barrier(CLK_GLOBAL_MEM_FENCE);

    float eps_xx=eps_ii(u1[r],u1[l],2*dx);
    float eps_yy=eps_ii(u2[b],u2[f],2*dy);
    float eps_zz=eps_ii(u3[u],u3[d],2*dz);

    float r_eps_xx=eps_ii(u1[r], u1[i],dx);
    float l_eps_xx=eps_ii(u1[i], u1[l],dx);
    float b_eps_xx=eps_ii(u1[br],u1[bl],2*dx);
    float f_eps_xx=eps_ii(u1[fr],u1[fl],2*dx);
    float u_eps_xx=eps_ii(u1[ur],u1[ul],2*dx);
    float d_eps_xx=eps_ii(u1[dr],u1[dl],2*dx);

    float b_eps_yy=eps_ii(u2[b], u2[i],dy);
    float f_eps_yy=eps_ii(u2[i], u2[f],dy);
    float r_eps_yy=eps_ii(u2[br],u2[fr],2*dy);
    float l_eps_yy=eps_ii(u2[bl],u2[fl],2*dy);
    float u_eps_yy=eps_ii(u2[ub],u2[uf],2*dy);
    float d_eps_yy=eps_ii(u2[db],u2[df],2*dy);

    float u_eps_zz=eps_ii(u3[u], u3[i],dz);
    float d_eps_zz=eps_ii(u3[i], u3[d],dz);
    float r_eps_zz=eps_ii(u3[ur],u3[dr],2*dz);
    float l_eps_zz=eps_ii(u3[ul],u3[dl],2*dz);
    float b_eps_zz=eps_ii(u3[ub],u3[db],2*dz);
    float f_eps_zz=eps_ii(u3[uf],u3[df],2*dz);

    float dzdx =didj(u3[r], u3[l] ,2*dx);
    float rdzdx=didj(u3[r], u3[i]   ,dx);
    float ldzdx=didj(u3[i], u3[l]   ,dx);
    float udzdx=didj(u3[ur],u3[ul],2*dx);
    float ddzdx=didj(u3[dr],u3[dl],2*dx);

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
    eps_yy=f_eps_yy;

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
    eps_yy=eps_ii(u2[b],u2[i],dy);

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
    eps_xx=eps_ii(u1[i],u1[l],dx);

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
    eps_xx=eps_ii(u1[r],u1[i],dx);

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

    eps_zz=eps_ii(u3[u],u3[i],dz);

    d_eps_xx=Eps_xx;
    d_eps_yy=Eps_yy;

    d_eps_zz=eps_ii(u3[i], 0,  dz);
    r_eps_zz=eps_ii(u3[ur],0,2*dz);
    l_eps_zz=eps_ii(u3[ul],0,2*dz);
    b_eps_zz=eps_ii(u3[ub],0,2*dz);
    f_eps_zz=eps_ii(u3[uf],0,2*dz);

    ddzdx=0;
    ddzdy=0;

    dxdz =didj(u1[u], u1[i],                 dz);
    rdxdz=didj(u1[ur],Eps_xx * dx * (x+1), 2*dz);
    ldxdz=didj(u1[ul],Eps_xx * dx * (x-1), 2*dz);
    ddxdz=didj(u1[i], Eps_xx * dx * x,       dz);

    dydz =didj(u2[u], u2[i],                 dz);
    bdydz=didj(u2[ub],Eps_yy * dy * (y+1), 2*dz);
    fdydz=didj(u2[uf],Eps_yy * dy * (y-1), 2*dz);
    ddydz=didj(u2[i], Eps_yy * dy *  y,      dz);
    };

    if (up_bd) {
    eps_zz  =eps_ii(u3[i],u3[d], dz);

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

    float   eps_xz=eps_ij( dxdz, dzdx);
    float u_eps_xz=eps_ij(udxdz,udzdx);
    float d_eps_xz=eps_ij(ddxdz,ddzdx);
    float r_eps_xz=eps_ij(rdxdz,rdzdx);
    float l_eps_xz=eps_ij(ldxdz,ldzdx);

    float   eps_xy=eps_ij( dxdy, dydx);
    float r_eps_xy=eps_ij(rdxdy,rdydx);
    float l_eps_xy=eps_ij(ldxdy,ldydx);
    float b_eps_xy=eps_ij(bdxdy,bdydx);
    float f_eps_xy=eps_ij(fdxdy,fdydx);

    float   eps_yz=eps_ij( dydz, dzdy);
    float b_eps_yz=eps_ij(bdydz,bdzdy);
    float f_eps_yz=eps_ij(fdydz,fdzdy);
    float u_eps_yz=eps_ij(udydz,udzdy);
    float d_eps_yz=eps_ij(ddydz,ddzdy);


    barrier(CLK_GLOBAL_MEM_FENCE);

    float r_sigma_xx  = sigma_ii(c11[r],r_eps_xx,c12[r],r_eps_yy,r_eps_zz);
    float l_sigma_xx  = sigma_ii(c11[l],l_eps_xx,c12[l],l_eps_yy,l_eps_zz);
    float b_sigma_yy  = sigma_ii(c11[b],b_eps_yy,c12[b],b_eps_xx,b_eps_zz);
    float f_sigma_yy  = sigma_ii(c11[f],f_eps_yy,c12[f],f_eps_xx,f_eps_zz);
    float u_sigma_zz  = sigma_ii(c11[u],u_eps_zz,c12[u],u_eps_xx,u_eps_yy);
    float d_sigma_zz  = sigma_ii(c11[d],d_eps_zz,c12[d],d_eps_xx,d_eps_yy);

    float r_sigma_xy = sigma_ij(c44[r], r_eps_xy);
    float l_sigma_xy = sigma_ij(c44[l], l_eps_xy);
    float b_sigma_xy = sigma_ij(c44[b], b_eps_xy);
    float f_sigma_xy = sigma_ij(c44[f], f_eps_xy);

    float r_sigma_xz = sigma_ij(c44[r], r_eps_xz);
    float l_sigma_xz = sigma_ij(c44[l], l_eps_xz);
    float u_sigma_xz = sigma_ij(c44[u], u_eps_xz);
    float d_sigma_xz = sigma_ij(c44[d], d_eps_xz);

    float b_sigma_yz = sigma_ij(c44[b], b_eps_yz);
    float f_sigma_yz = sigma_ij(c44[f], f_eps_yz);
    float d_sigma_yz = sigma_ij(c44[d], d_eps_yz);
    float u_sigma_yz = sigma_ij(c44[u], u_eps_yz);

    if (down_bd) {
    d_sigma_zz = sigma_ii(c11[i], d_eps_zz, c12[i],d_eps_xx,d_eps_yy);
    d_sigma_xz = sigma_ij(c44[i], d_eps_xz);
    d_sigma_yz = sigma_ij(c44[i], d_eps_yz);
    };

    float dsigmaxxdx = (r_sigma_xx-l_sigma_xx)/(2*dx);
    float dsigmayydy = (b_sigma_yy-f_sigma_yy)/(2*dy);
    float dsigmazzdz = (u_sigma_zz-d_sigma_zz)/(2*dz);

    float dsigmaxydx = (r_sigma_xy-l_sigma_xy)/(2*dx);
    float dsigmaxydy = (b_sigma_xy-f_sigma_xy)/(2*dy);

    float dsigmaxzdx = (r_sigma_xz-l_sigma_xz)/(2*dx);
    float dsigmaxzdz = (u_sigma_xz-d_sigma_xz)/(2*dz);

    float dsigmayzdy = (b_sigma_yz-f_sigma_yz)/(2*dy);
    float dsigmayzdz = (u_sigma_yz-d_sigma_yz)/(2*dz);

    if (up_bd) {
    dsigmazzdz = (-d_sigma_zz)/dz;
    dsigmaxzdz = (-d_sigma_xz)/dz;
    dsigmayzdz = (-d_sigma_yz)/dz;

    dsigmaxzdx = 0;
    dsigmayzdy = 0;
    };

    if (pochti_up_bd) {
    dsigmazzdz = (-d_sigma_zz)/(2*dz);
    dsigmaxzdz = (-d_sigma_xz)/(2*dz);
    dsigmayzdz = (-d_sigma_yz)/(2*dz);
    };

    if (back_bd) {
    dsigmayydy = (-f_sigma_yy)/dy;
    dsigmaxydy = (-f_sigma_xy)/dy;
    dsigmayzdy = (-f_sigma_yz)/dy;

    dsigmaxydx = 0;
    dsigmayzdz = 0;
    };

    if (pochti_back_bd) {
    dsigmayydy = (-f_sigma_yy)/(2*dy);
    dsigmaxydy = (-f_sigma_xy)/(2*dy);
    dsigmayzdy = (-f_sigma_yz)/(2*dy);
    };

    if (front_bd) {
    dsigmayydy = (b_sigma_yy)/dy;
    dsigmaxydy = (b_sigma_xy)/dy;
    dsigmayzdy = (b_sigma_yz)/dy;

    dsigmaxydx = 0;
    dsigmayzdz = 0;
    };

    if (pochti_front_bd) {
    dsigmayydy = (b_sigma_yy)/(2*dy);
    dsigmaxydy = (b_sigma_xy)/(2*dy);
    dsigmayzdy = (b_sigma_yz)/(2*dy);
    };

    if (right_bd) {
    dsigmaxxdx = (-l_sigma_xx)/dx;
    dsigmaxydx = (-l_sigma_xy)/dx;
    dsigmaxzdx = (-l_sigma_xz)/dx;

    dsigmaxydy = 0;
    dsigmaxzdz = 0;
    };

    if (pochti_right_bd) {
    dsigmaxxdx = (-l_sigma_xx)/(2*dx);
    dsigmaxydx = (-l_sigma_xy)/(2*dx);
    dsigmaxzdx = (-l_sigma_xz)/(2*dx);
    };

    if (left_bd) {
    dsigmaxxdx = (r_sigma_xx)/dx;
    dsigmaxydx = (r_sigma_xy)/dx;
    dsigmaxzdx = (r_sigma_xz)/dx;

    dsigmaxydy = 0;
    dsigmaxzdz = 0;
    };

    if (pochti_left_bd) {
    dsigmaxxdx = (r_sigma_xx)/(2*dx);
    dsigmaxydx = (r_sigma_xy)/(2*dx);
    dsigmaxzdx = (r_sigma_xz)/(2*dx);
    };

    barrier(CLK_GLOBAL_MEM_FENCE);

    out[i]           = (2*(dsigmaxxdx + dsigmaxydy + dsigmaxzdz)*dt2 + 4*rho[i]*u1[i] + (Alpha*dt - 2*rho[i])*v1[i])/(Alpha*dt + 2*rho[i]);
    out[i + len]     = (2*(dsigmaxydx + dsigmayydy + dsigmayzdz)*dt2 + 4*rho[i]*u2[i] + (Alpha*dt - 2*rho[i])*v2[i])/(Alpha*dt + 2*rho[i]);
    out[i + 2 * len] = (2*(dsigmaxzdx + dsigmayzdy + dsigmazzdz)*dt2 + 4*rho[i]*u3[i] + (Alpha*dt - 2*rho[i])*v3[i])/(Alpha*dt + 2*rho[i]);

    out[i + 3 * len] = u1[i];
    out[i + 4 * len] = u2[i];
    out[i + 5 * len] = u3[i];

    out[i + 6 * len]  = eps_xx;
    out[i + 7 * len]  = eps_yy;
    out[i + 8 * len]  = eps_zz;
    out[i + 9 * len]  = eps_xy;
    out[i + 10 * len] = eps_yz;
    out[i + 11 * len] = eps_xz;
};
};
"""

# build the Kernel
prog = cl.Program(ctx, code_el).build()

def dynamics(data, consts):
    mf = cl.mem_flags
    u1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['u1'].astype(np.float32))
    u2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['u2'].astype(np.float32))
    u3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['u3'].astype(np.float32))
    v1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['v1'].astype(np.float32))
    v2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['v2'].astype(np.float32))
    v3_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data['v3'].astype(np.float32))

    c11_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=consts['c11'].astype(np.float32))
    c12_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=consts['c12'].astype(np.float32))
    c44_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=consts['c44'].astype(np.float32))
    rho_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=consts['rho'].astype(np.float32))

    out_el = consts['out']
    out_el_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=out_el.astype(np.float32))

    launch = prog.my_el(queue, data['u1'].shape, None,
    u1_buf, u2_buf, u3_buf,
    v1_buf, v2_buf, v3_buf,
    c11_buf, c12_buf, c44_buf, rho_buf,
    consts['Lx'], consts['Ly'], consts['Lz'], consts['L'], consts['dt'], consts['Alpha'],
    consts['dx'], consts['dy'], consts['dz'], consts['Eps_xx'], consts['Eps_yy'],
    out_el_buf)
    launch.wait()

    cl.enqueue_copy(queue, out_el, out_el_buf)

    L = int(consts['L'])

    return ({'u1' : out_el[0: L], 'u2' : out_el[L: (2 * L)], 'u3' : out_el[(2 * L): (3 * L)],
            'v1' : out_el[(3 * L): (4 * L)], 'v2' : out_el[(4 * L): (5 * L)], 'v3' : out_el[(5 * L): (6 * L)]},

            {'eps_xx':out_el[(6 * L): (7 * L)],'eps_yy': out_el[(7 * L): (8 * L)],'eps_zz': out_el[(8 * L): (9 * L)],
            'eps_xy':out_el[(9 * L): (10 * L)],'eps_yz': out_el[(10 * L): (11 * L)],'eps_xz': out_el[(11 * L): (12 * L)]})



def plot_xy_pl(a, layer, dir, count, consts):

    Lx = int(consts['Lx'])
    Ly = int(consts['Ly'])
    Lz = int(consts['Lz'])

    dx = consts['dx']
    dy = consts['dy']

    Xpos, Ypos = np.meshgrid(np.arange(0, dx * Lx, dx), np.arange(0, dy * Ly, dy))

    pl = np.reshape(a, (Lz, Ly, Lx))[layer]
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


def plot_1D_z(data,dir,count, consts):

    Lx = int(consts['Lx'])
    Ly = int(consts['Ly'])
    Lz = int(consts['Lz'])

    dz = int(consts['dz'])

    a1 = np.reshape(data['eps_xx'], (Lz, Ly, Lx))
    a2 = np.reshape(data['eps_yy'], (Lz, Ly, Lx))
    a3 = np.reshape(data['eps_zz'], (Lz, Ly, Lx))


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
    ax.plot(t, pl1,"b", label="$\epsilon_{xx}$")
    ax.plot(t, pl2,"r--", label="$\epsilon_{yy}$")
    ax.plot(t, pl3, "g", label="$\epsilon_{zz}$")
    ax.legend(loc='lower left')
    ax.set(xlabel='z coordinate (nm)', ylabel='Mechanical strains (%)')
    ax.grid()

    plt.savefig(dir+"/film/z_strains" + str(count) + ".png", dpi=100)
    plt.close()

def save_data(data, dir):
    np.save(data, dir+ '/TXT/')
