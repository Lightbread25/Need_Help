import numpy as np
import matplotlib.pyplot as plt
import time
import camb
from camb import model
import matplotlib.pyplot as plt
import numpy as np
import pdb

h = 0.6774
omb = 0.0486
Om = 0.3089
omc = Om - omb #0.261 
w=-1
ns = 0.965
sigma8 = 0.8159

As = 2e-9 # fiducial amplitude guess to start with
pars =  camb.set_params(H0=h*100, ombh2=omb * (h**2), omch2=omc *(h**2)
                        , w = w, ns=ns, As=As, omk = 0.0)
pars.set_matter_power(redshifts=[0.], kmax=2.0)
results = camb.get_results(pars)
s8_fid = results.get_sigma8_0()
# now set correct As using As \propto sigma8**2.


pars.InitPower.set_params(As=As*sigma8**2/s8_fid**2, ns=ns)
pars.set_for_lmax(2500,lens_potential_accuracy=0)
pars.set_matter_power(redshifts=[0,0.2,0.4,0.6,0.8,1,2,4], kmax=2.0)
PK = camb.get_matter_power_interpolator(pars, nonlinear=False, 
        hubble_units=True, k_hunit=True, kmax=2.0 ,extrap_kmax = 10)
results = camb.get_results(pars)


def Radius(Mass, Omega_M = 0.31):
    '''
    Mass to Radius transform evaluated at z = 0 

    Mass input needs to be in units of h^-1 Msun

    Returns R in units of h^-1 Mpc
    '''
    rho_c_0 = 2.775e11 #h^2 M_sun Mpc^-3
    bar_rho_m = Omega_M * rho_c_0 
    R = np.power(np.divide((3*Mass) , (4 *np.pi*bar_rho_m)), (1/3)) #h^-1 Mpc
    return R


def sigma_R(R,redshift):
    '''
    Radius input needs to be in units of h^-1 Mpc
    '''
    k = np.linspace(1e-4,10,5000)
    dk = (k[2:]-k[:-2]) /2
    ks = k[1:-1]

    def Top_Hat(x):
        """
        Defining the Bessel Function j_1(x)
        """
        return 3*( (np.sin(x)/(x**3)) - (np.cos(x)/(x**2)) )
    # pdb.set_trace()

    kR = np.tensordot(R,ks,axes = 0)

    if isinstance(redshift,float) or isinstance(redshift,int):
        f = (1/(2*(np.pi**2)))*PK.P(redshift,ks,grid = False)* (ks**2) * ( ((Top_Hat(kR))**2 ))
        s_2 = np.tensordot(f,dk,axes = 1)
        sigma = np.sqrt(s_2)
    else:
        # Precompute Top_Hat values for all kR
        top_hat_values = np.array([Top_Hat(kr) for kr in kR])

        # Precompute the factor outside the loops
        factor = 1 / (2 * np.pi**2)

        # Initialize sigma array
        sigma = np.zeros((len(redshift), len(R)))

        # Vectorize over redshift
        for i, z in enumerate(redshift):
            pk_values = PK.P(z, ks, grid=False)
            # Vectorize over kR
            f = factor * pk_values * ks**2 * (top_hat_values[:, None]**2)
            s_2 = np.tensordot(f,dk,axes = 1)[:, 0]
            sigma[i, :] = np.sqrt(s_2)
    return sigma
 

def dlns_dlnM(Mass,redshift,step = 1e-8):
    R_left = Radius(np.exp(np.log(Mass)-step))
    R_right = Radius(np.exp(np.log(Mass)+step))
    left_side = np.log(sigma_R(R_left,redshift))
    right_side = np.log(sigma_R(R_right,redshift))
    derivative = (right_side - left_side) / (2*step)
    return np.abs(derivative)
    # pdb.set_trace()

def nu_f_nu(nu,redshift):
    if not isinstance(redshift,float) or isinstance(redshift,int):
        redshift = redshift[:, np.newaxis]
    alpha = 0.368
    beta = 0.589 * np.power(1+redshift,0.20)
    gamma = 0.864 * np.power(1+redshift,-0.01)
    phi =  -0.729 * np.power( 1+ redshift, -0.08)
    eta =  -0.243 * np.power(1+redshift, 0.27)
    f_nu = alpha * (1+ np.power(beta*nu,-2*phi)) * np.power(nu,2*eta) * np.exp((-gamma*(nu**2))/2)
    return nu * f_nu



def SPT_f(sigma,redshift):
    if not isinstance(redshift,float) or isinstance(redshift,int):
        redshift = redshift[:, np.newaxis]
    A0 = 0.175
    a0 = 1.53
    b0 = 2.55
    c0 = 1.19
    A = A0 *(1+redshift) ** (-0.012)
    a = a0 *(1+redshift) ** (-0.040) 
    b = b0 *(1+redshift) ** (-0.194)
    c = c0 *(1+redshift) ** (-0.021)
    f = A *(np.power(sigma/b,-a) +1)* np.exp((-c/(np.power(sigma,2))))
    return f


def bias(nu,delta_m=200):
    y = np.log10(delta_m)
    a = 0.44*y -0.88
    b = 1.5
    c = 2.4
    A = 1 + 0.24*y*np.exp(-np.power((4/y),(4)))
    B = 0.183
    C = 0.019 + 0.107*y + 0.19 * np.exp(-np.power((4/y),(4)))
    fraction = np.divide((np.power(nu,a)), ( np.power(nu,a) + np.power(1.686,a) ))
    bias = 1 - A*fraction + B * np.power(nu,(b)) + C * np.power(nu,(c))
    return bias

def Tinker_2010(Mass,redshift,dn_dlnm = False):
    '''
    Calculates the Tinker 2010 mass function and the corresponding bias
    '''
    Omega_M = omb + omc

    rho_c_0 = 2.775e11 #h^2 M_sun Mpc^-3
    bar_rho_m = Omega_M * rho_c_0  #* np.power((1+redshift),3)
    Mass_fraction = np.divide(bar_rho_m,Mass)
    R = Radius(Mass, Omega_M= Omega_M)
    sigma = sigma_R(R, redshift)

    nu = 1.686/sigma
    # pdb.set_trace()
    unbiased_dn_dM = dlns_dlnM(Mass,redshift) * Mass_fraction * nu_f_nu(nu,redshift)  
    bias_dn_dM = bias(nu,delta_m=200) * unbiased_dn_dM

    if dn_dlnm == False:
        unbiased_dn_dM *= 1/Mass
        bias_dn_dM *= 1/Mass


    return unbiased_dn_dM , bias_dn_dM




def Bocquet_2016(Mass,redshift,dn_dlnm = False):
    '''
    Calculates the Bocquet 2016 mass function 
    and the corresponding Tinker 2010 bias
    '''
    Omega_M = omb + omc

    rho_c_0 = 2.775e11 #h^2 M_sun Mpc^-3
    bar_rho_m = Omega_M * rho_c_0  #* np.power((1+redshift),3)
    Mass_fraction = np.divide(bar_rho_m,Mass)
    R = Radius(Mass, Omega_M= Omega_M)
    sigma = sigma_R(R, redshift)

    nu = 1.686/sigma
    # pdb.set_trace()
    unbiased_dn_dM = dlns_dlnM(Mass,redshift) * Mass_fraction * SPT_f(sigma,redshift)  
    bias_dn_dM = bias(nu,delta_m=200) * unbiased_dn_dM

    if dn_dlnm == False:
        unbiased_dn_dM *= 1/Mass
        bias_dn_dM *= 1/Mass


    return unbiased_dn_dM , bias_dn_dM





def b_eff(Min_Mass,redshift):
    min = np.log(float(Min_Mass))
    lnM = np.linspace(min,np.log(1e16),num=2000)
    dlnM = (lnM[2:]-lnM[:-2])/2
    lnM = lnM[1:-1]
    M = np.exp(lnM)

    Unbiased_M, Biased_M = Tinker_2010(M,redshift,dn_dlnm=True)
    dem = np.dot(Unbiased_M,dlnM)
    num = np.dot(Biased_M,dlnM)
    f = num/dem
    return f

def dN_dz(redshift,Min_Mass,sky_perc = 0.4,Method = "Tinker"):
    sky_coverage = sky_perc * 4 * np.pi *(180/np.pi)**2
    min = np.log(float(Min_Mass))
    lnM = np.linspace(min,np.log(1e16),num=2000)
    dlnM = (lnM[2:]-lnM[:-2])/2
    lnM = lnM[1:-1]
    M = np.exp(lnM)
    # M = np.logspace(min,16,num= 2000)
    # dM = (M[2:]-M[:-2]) /2
    # M = M[1:-1]
    if Method == "Tinker":
        Unbiased_M, _ = Tinker_2010(M,redshift,dn_dlnm=True)
    elif Method == "Bocquet":
        Unbiased_M, _ = Bocquet_2016(M,redshift,dn_dlnm=True)
    Int = np.dot(Unbiased_M,dlnM)
    chi = results.comoving_radial_distance(redshift) *h
    Hz = results.h_of_z(redshift) / h
    CVE =  (chi**2)/Hz
    Omega_Sky = sky_coverage * np.power(np.pi/180,2)
    f = Omega_Sky * CVE * Int 
    # pdb.set_trace()
    return f

def N(z_min,z_max,min_mass,sky_perc = 0.4,Method = 'Tinker'):
    '''
    For N from z=0.2 to z = 4 with minimum mass of 1e13 h^-1 Msun
    Total number is 35215169.700532876 approx to 35215170
    '''
    z = np.linspace(z_min,z_max,num= 2000)
    dz = (z[2:]-z[:-2]) /2
    zs = z[1:-1]
    n = dN_dz(zs,min_mass,sky_perc =sky_perc,Method=Method)
    # pdb.set_trace()
    f = np.dot(n,dz)
    return f





'''
Testing zone in case anything goes wrong
'''


# start = time.time()

# redshift = np.linspace(0,5)

# M = np.logspace(13,17,num= 100)
# print(Tinker_2010(M,redshift)[0])
# f = dN_dz(redshift,1e13)/35215170
# Mass = 1e14

# plt.plot(redshift,f)
# plt.show()
# DN_DZ = dN_dz(0.2,1e13)
# TOTAL = N(0.2,0.4,1e13)


# print(DN_DZ)
# print(TOTAL) #35215170 #.700532876
# print(DN_DZ/TOTAL)

# print(b_eff(1e14,0.0))


# M = np.logspace(13,17,num= 2000)
# dM = (M[2:]-M[:-2]) /2
# M = M[1:-1]


# Unbiased_M, Biased_M = Tinker_2010(M,0.0)
# dem = np.dot(Unbiased_M,dM)
# num = np.dot(Biased_M,dM)
# # pdb.set_trace()
# print((num/dem))

# M = np.logspace(12,16)
# R = Radius(M)
# s = sigma_R(R,0.0)
# plt.plot(np.log10(M),bias(1.686/s))
# plt.show()




# start = time.time()
# Mass = np.array([1e13,1e14,1e15]) #np.logspace(11,20,num= 50)

# plt.plot((Mass),Tinker_2010(Mass,0.0,dn_dlnm= True)[0],'b',label='z=0,Tinker')
# # plt.plot((Mass),Tinker_2010(Mass,0.0,dn_dlnm= True)[1],'r',label='z=0,biased')
# plt.plot((Mass),Tinker_2010(Mass,1.0,dn_dlnm= True)[0],'c',label='z=1,Tinker')
# plt.plot((Mass),Tinker_2010(Mass,2.0,dn_dlnm= True)[0],'g',label='z=2,Tinker')
# plt.plot((Mass),Tinker_2010(Mass,4.0,dn_dlnm= True)[0],'y',label='z=4,Tinker')


# plt.plot((Mass),Bocquet_2016(Mass,0.0,dn_dlnm= True)[0],'y',label='z=0, Bocquet')
# # plt.plot((Mass),Bocquet_2016(Mass,0.0,dn_dlnm= True)[1],'r',label='z=0,biased')
# plt.plot((Mass),Bocquet_2016(Mass,1.0,dn_dlnm= True)[0],'b',label='z=1')
# plt.plot((Mass),Bocquet_2016(Mass,2.0,dn_dlnm= True)[0],'c',label='z=2')
# plt.plot((Mass),Bocquet_2016(Mass,4.0,dn_dlnm= True)[0],'g',label='z=4')

# # end = time.time()
# # print('time spent:', end - start)
# plt.plot(redshift,Bocquet_2016(Mass,redshift,dn_dlnm=True)[0])

# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1e13,1e16)
# plt.ylim(1e-10,1e-3)
# plt.title('Comparision of Tinker 2010 mass function of M200m Halo at z=0')
# plt.title('Comparison of Bocquet 2016 mass function of M200m Halo at z=0')
# plt.xlabel('Mass of halo ' + r"$log_{10}(M_{200m} h^{-1}M_{\odot})$")
# plt.ylabel(r'$\ln{\sigma^{-1}}$')
# plt.ylabel(r'$dn/dlnM (h^3Mpc^{-3} )$')
# plt.legend()
# plt.grid()
# plt.show()

# def test(redshift):
#     chi = results.comoving_radial_distance(redshift) *h
#     Hz = results.h_of_z(redshift) * h
#     CVE =  (chi**2)/Hz
#     return CVE


# plt.plot(redshift,test(redshift)/44)
# plt.yscale('log')
# plt.show()
