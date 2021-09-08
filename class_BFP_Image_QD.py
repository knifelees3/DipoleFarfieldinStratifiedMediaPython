
"""
BFP Image Calculations of QD in Multi-layered structure

This is to calculate the BFP image for a given material and given
kx,ky distribution. Once kx,ky are given, then the pattern can be calclated.

# The description of this class can be found in the file "ReadMe.md"

Author: Zhaohua Tian
email: knifelees3@hotmail.com
web page: knifelees3.githun.io
"""


# Import commerical package
import numpy as np
import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Import my own package
import Fun_BFP_Image


# ______________________________________________________________________________
# The definition of the Class
class BFP_Image_QD:
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # **********************************Initialize Parameters*****************
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self, Eplist, dl, nUp, nDn, p0, WL0, POSD):

        # Optical Constant
        self.epsilon0 = 8.854187817620389850537e-12
        self.mu0 = 1.2566370614359172953851e-6
        self.const_c = np.sqrt(1 / self.epsilon0 / self.mu0)

        # Main input parameters
        self.Eplist = Eplist
        self.dl = dl
        self.p0 = p0
        self.nUp = nUp
        self.nDn = nDn
        self.WL0 = WL0
        self.POSD = POSD

        # Test the input
        error_count = self.__check_input(Eplist, dl, nUp, nDn)
        if error_count == 0:
            # pass
            # f1 = open("process.txt", "a+")
            # f1.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            #          ':The Basic Parameters Have Been Initialized!!! \n')
            # f1.close
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                  ':The Basic Parameters Have Been Initialized!!!')
        else:
            # f1 = open("process.txt", "a+")
            # f1.write(datetime.datetime.now().strftime(
            #     '%Y-%m-%d %H:%M:%S') + ': WARNING: The parameters are not coorect, the results maybe not reliable!!! \n')
            # f1.close
            print(datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + ': WARNING: The parameters are not coorect, the results maybe not reliable!!!')

        # Other parameters can be derived from the input parameters

        # number of layers
        self.num_layer = nUp + nDn + 1
        # number of coodinates
        self.num_dl = nUp + nDn
        # free space wave vector
        self.k0 = 2 * np.pi / WL0
        # Upper length to calculate the far field
        self.dUpFar = WL0 * 500
        # Lower length to calculate the far field
        self.dDnFar = -WL0 * 600
        # the wavevector in each layer
        self.kl = self.k0 * np.sqrt(Eplist)
        # the angular frequency
        self.omega = 2 * np.pi / self.WL0 * self.const_c

        # parameters that should be added laterly
        # The grid
        self.kx, self.ky, self.klz = 0, 0, 0
        # The size
        self.num_kx, self.num_ky = 0, 0
        # Green function initialize
        self.GreenSUp, self.GreenPUp, self.GreenSDn, self.GreenPDn = 0, 0, 0, 0
        # The angle list will be used
        self.theta_Up, self.theta_Dn = 0, 0


# Check the correctness of the initialization

    def __check_input(self, Eplist, dl, nUp, nDn):
        size_Eplist = len(Eplist)
        size_dl = len(dl)

        error_count = 0
        if size_dl + 1 != size_Eplist:
            error_count = error_count + 1

        if nUp + nDn + 1 != size_Eplist:
            error_count = error_count + 1

        return error_count


# *******************************************************************
# ########################Functions Part#############################
# ___________________________________________________________________

# Part I Bacis Function Part
# *******************************************************************
# ___________________________________________________________________

# To calculate the R_{s/p} in different interface


    def Cal_RSP(self, klz):
        num_dl = self.num_dl
        Eplist = self.Eplist
        dl = self.dl
        nUp = self.nUp
        nDn = self.nDn

        RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21 = \
            Fun_BFP_Image.Cal_RSP(num_dl, Eplist, dl, nUp, nDn, klz)

        return RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21


# To calculate the Green function

    def Cal_Green(self, RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21, kx, ky, klz):
        nUp = self.nUp
        nDn = self.nDn
        num_layer = self.num_layer
        dl = self.dl
        POSD = self.POSD

        dUpFar = self.dUpFar
        dDnFar = self.dDnFar

        GreenSUp_Far, GreenPUp_Far, GreenSDn_Far, GreenPDn_Far = \
            Fun_BFP_Image.Cal_Green(nUp, nDn, num_layer, dl, POSD, dUpFar, dDnFar,
                                    RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21, kx, ky, klz)

        return GreenSUp_Far, GreenPUp_Far, GreenSDn_Far, GreenPDn_Far

# To calculate the electric field when the Green function is given
    def Cal_Elec_Field(self, GreenSUp, GreenPUp, GreenSDn, GreenPDn):
        p0 = self.p0
        omega = self.omega
        mu0 = self.mu0

        ESUp_Far, EPUp_Far, ESDn_Far, EPDn_Far = Fun_BFP_Image.Cal_Elec_Field(
            GreenSUp, GreenPUp, GreenSDn, GreenPDn,
            p0, omega, mu0)

        return ESUp_Far, EPUp_Far, ESDn_Far, EPDn_Far

# To calculate the emission pattern for a given elecric field
    def Cal_Pattern(self, ESUp, EPUp, ESDn, EPDn, theta_Up, theta_Dn):

        epsilon0 = self.epsilon0
        mu0 = self.mu0
        num_layer = self.num_layer
        Eplist = self.Eplist

        PatternUp, PatternDn = Fun_BFP_Image.Cal_Pattern(
            Eplist, epsilon0, mu0, num_layer, ESUp, EPUp, ESDn, EPDn, theta_Up, theta_Dn)

        nPatternUp = PatternUp / np.max(PatternUp)
        nPatternDn = PatternDn / np.max(PatternDn)
        return nPatternUp, nPatternDn


# Part II Pattern Function Part
# ****************************************************************************
# ____________________________________________________________________________
    # _____________________________________________________________________________________________________
    # To calculate the green function for a given kx,ky list which can be used laterly

    def Cal_Green_List(self, kx, ky):
        self.kx, self.ky = kx, ky
        self.num_kx, self.num_ky = kx.shape

        klz = np.zeros((self.num_kx, self.num_ky,
                        self.num_layer), dtype=complex)
        for l in range(self.num_layer):
            klz[:, :, l] = np.sqrt(self.kl[l]**2 - self.kx**2 - self.ky**2)
        self.klz = klz

        self.theta_Up = np.arccos(self.klz[:, :, self.num_layer - 1] /
                                  self.kl[self.num_layer - 1])
        self.theta_Dn = np.arccos(self.klz[:, :, 0] / self.kl[0])

        GreenSUp, GreenPUp, GreenSDn, GreenPDn = np.zeros((3, 3, self.num_kx, self.num_ky), dtype=complex), np.zeros(
            (3, 3, self.num_kx, self.num_ky), dtype=complex), np.zeros((3, 3, self.num_kx, self.num_ky), dtype=complex), np.zeros((3, 3, self.num_kx, self.num_ky), dtype=complex)

        for l in range(self.num_kx):
            for m in range(self.num_ky):
                RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21 = self.Cal_RSP(
                    self.klz[l, m, :])
                GreenSUp[:, :, l, m], GreenPUp[:, :, l, m], GreenSDn[:, :, l, m], GreenPDn[:, :, l, m] = self.Cal_Green(
                    RSUp, RPUp, RSDn, RPDn, RS12, RP12, RS21, RP21, self.kx[l, m], self.ky[l, m], self.klz[l, m, :])
        self.GreenSUp, self.GreenPUp, self.GreenSDn, self.GreenPDn = GreenSUp, GreenPUp, GreenSDn, GreenPDn

        print(datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + ': The Green Function Has Been Prepared')
        return 0


    # For a given p1
    # The upper and lower pattern will both be simulated
    def Cal_Pattern_List_QD_p1(self, p1):

        ESUp = np.zeros((self.num_kx, self.num_ky, 3), dtype=complex)
        EPUp = np.zeros((self.num_kx, self.num_ky, 3), dtype=complex)
        ESDn = np.zeros((self.num_kx, self.num_ky, 3), dtype=complex)
        EPDn = np.zeros((self.num_kx, self.num_ky, 3), dtype=complex)

        for l in range(3):
            ESUp[:, :, l] = 1j * self.omega * self.mu0 * \
                (self.GreenSUp[l, 0, :, :] * p1[0] + self.GreenSUp[l,
                                                                   1, :, :] * p1[1] + self.GreenSUp[l, 2, :, :] * p1[2])
            EPUp[:, :, l] = 1j * self.omega * self.mu0 * \
                (self.GreenPUp[l, 0, :, :] * p1[0] + self.GreenPUp[l,
                                                                   1, :, :] * p1[1] + self.GreenPUp[l, 2, :, :] * p1[2])
            ESDn[:, :, l] = 1j * self.omega * self.mu0 * \
                (self.GreenSDn[l, 0, :, :] * p1[0] + self.GreenSDn[l,
                                                                   1, :, :] * p1[1] + self.GreenSDn[l, 2, :, :] * p1[2])
            EPDn[:, :, l] = 1j * self.omega * self.mu0 * \
                (self.GreenPDn[l, 0, :, :] * p1[0] + self.GreenPDn[l,
                                                                   1, :, :] * p1[1] + self.GreenPDn[l, 2, :, :] * p1[2])


        # Cal the emission pattern
        PatternUpS = (np.abs(ESUp[:, :, 0])**2 + np.abs(ESUp[:, :, 1])**2 + np.abs(ESUp[:, :, 2])**2) * np.abs(np.cos(self.theta_Up[:, :]))**2
        PatternUpP = (np.abs(EPUp[:, :, 0])**2 + np.abs(EPUp[:, :, 1])**2 + np.abs(EPUp[:, :, 2])**2) * np.abs(np.cos(self.theta_Up[:, :]))**2
        PatternDnS = (np.abs(ESDn[:, :, 0])**2 + np.abs(ESDn[:, :, 1])**2 + np.abs(ESDn[:, :, 2])**2) * np.abs(np.cos(self.theta_Dn[:, :]))**2
        PatternDnP = (np.abs(EPDn[:, :, 0])**2 + np.abs(EPDn[:, :, 1])**2 + np.abs(EPDn[:, :, 2])**2) * np.abs(np.cos(self.theta_Dn[:, :]))**2
        PatternUp = PatternUpS + PatternUpP
        PatternDn = PatternDnS + PatternDnP
        nPatternUp = PatternUp 
        nPatternDn = PatternDn
        return nPatternUp,nPatternDn

# Part III Data Process Part
# **********************************************************************************************
# ______________________________________________________________________________________________

    def Cal_RhoPhi_Dis(self, Pattern_RhoPhi):
        Pattern_Rho = np.sum(Pattern_RhoPhi, axis=0)
        Pattern_Phi = np.sum(Pattern_RhoPhi, axis=1)
        return Pattern_Rho, Pattern_Phi

    def Trans_XY_to_RhoPhi(self, kx_grid, ky_grid, Pattern_XY, kx_grid_in, ky_grid_in):
        Pattern_RhoPhi = Fun_BFP_Image.Grid_Data_TZH(
            kx_grid, ky_grid, Pattern_XY, kx_grid_in, ky_grid_in)
        Pattern_Rho, Pattern_Phi = self.Cal_RhoPhi_Dis(Pattern_RhoPhi)

        return Pattern_Rho, Pattern_Phi


# Part V Structure visualization part
# **********************************************************************************************
# ______________________________________________________________________________________________
    def Show_Structure(self):
        Fun_BFP_Image.Show_Structure(self.Eplist, self.dl, self.WL0, self.nUp, self.nDn, self.POSD)


        
# Add in 2021 07 27
# Part VI The definition of Dipole3D and 3D pattern
# **********************************************************************************************
# ______________________________________________________________________________________________
    def QDDark(self,para):
        alpha=para[0]
        phi=para[1]
        ratio=para[2]

        # For Dipole 1
        d1x = -np.cos(alpha) * np.cos(phi)
        d1y = -np.cos(alpha) * np.sin(phi)
        d1z =  np.sin(alpha)

        # For dipole 2
        d2x =  np.sin(phi)
        d2y = -np.cos(phi)
        d2z =  0

        # For dark axis
        d3x=np.sin(alpha)*np.cos(phi)
        d3y=np.sin(alpha)*np.sin(phi)
        d3z=np.cos(alpha)

        nor=np.sqrt(1**2+1**2+np.abs(ratio)**2)
        p1 =np.array([d1x, d1y, d1z])/nor
        p2 = np.array([d2x, d2y, d2z])/nor
        p3 =np.array([d3x, d3y, d3z])/nor*ratio
        
        return p1,p2,p3

    def Dipole3D(self,para):
        # The 3D dipole model consists of 3 orthogonal dipoles
        alpha=para[0]
        beta=para[1]
        phix=para[2]
        phiy=para[3]
        phiz=para[4]
        dx=np.transpose([1,0,0])
        dy=np.transpose([0,alpha,0])
        dz=np.transpose([0,0,beta])

        norm=np.sqrt(1**2+alpha**2+beta**2)

        rotation=np.zeros((3,3))
        rotation[:,0]=np.transpose(np.array([np.cos(phiz)*np.cos(phiy),np.sin(phiz)*np.cos(phiy),-np.sin(phiy)]))
        rotation[:,1]=np.transpose(np.array([np.cos(phiz)*np.sin(phiy)*np.sin(phix)-np.sin(phiz)*np.cos(phix),np.sin(phiz)*np.sin(phiy)*np.sin(phix)+np.cos(phiz)*np.cos(phix),np.cos(phiy)*np.sin(phix)]))
        rotation[:,2]=np.transpose(np.array([np.cos(phiz)*np.sin(phiy)*np.cos(phix)+np.sin(phiz)*np.sin(phix),np.sin(phiz)*np.sin(phiy)*np.cos(phix)-np.cos(phiz)*np.sin(phix),np.cos(phiy)*np.cos(phix)]))

        d1=np.dot(rotation,dx)/norm
        d2=np.dot(rotation,dy)/norm
        d3=np.dot(rotation,dz)/norm

        return d1,d2,d3

    def Pattern3DPara(self,para):
        d1,d2,d3=self.Dipole3D(para)
        PatternUpd1,PatternDnd1 = self.Cal_Pattern_List_QD_p1(d1)
        PatternUpd2,PatternDnd2 = self.Cal_Pattern_List_QD_p1(d2)
        PatternUpd3,PatternDnd3 = self.Cal_Pattern_List_QD_p1(d3)
        PatternUpd1[np.isnan(PatternUpd1)]=0
        PatternUpd2[np.isnan(PatternUpd2)]=0
        PatternUpd3[np.isnan(PatternUpd3)]=0
        PatternDnd1[np.isnan(PatternDnd1)]=0
        PatternDnd2[np.isnan(PatternDnd2)]=0
        PatternDnd3[np.isnan(PatternDnd3)]=0
        PatternUp=PatternUpd1+PatternUpd2+PatternUpd3
        PatternDn=PatternDnd1+PatternDnd2+PatternDnd3
        return PatternUp,PatternDn

    def Pattern2DDarkPara(self,para):
        d1,d2,d3=self.QDDark(para)
        PatternUpd1,PatternDnd1 = self.Cal_Pattern_List_QD_p1(d1)
        PatternUpd2,PatternDnd2 = self.Cal_Pattern_List_QD_p1(d2)
        PatternUpd3,PatternDnd3 = self.Cal_Pattern_List_QD_p1(d3)
        PatternUpd1[np.isnan(PatternUpd1)]=0
        PatternUpd2[np.isnan(PatternUpd2)]=0
        PatternUpd3[np.isnan(PatternUpd3)]=0
        PatternDnd1[np.isnan(PatternDnd1)]=0
        PatternDnd2[np.isnan(PatternDnd2)]=0
        PatternDnd3[np.isnan(PatternDnd3)]=0
        PatternUp=PatternUpd1+PatternUpd2+PatternUpd3
        PatternDn=PatternDnd1+PatternDnd2+PatternDnd3
        return PatternUp,PatternDn

    def Pattern3D(self,p1,p2,p3):
        PatternUpd1,PatternDnd1 = self.Cal_Pattern_List_QD_p1(d1)
        PatternUpd2,PatternDnd2 = self.Cal_Pattern_List_QD_p1(d2)
        PatternUpd3,PatternDnd3 = self.Cal_Pattern_List_QD_p1(d3)
        PatternUpd1[np.isnan(PatternUpd1)]=0
        PatternUpd2[np.isnan(PatternUpd2)]=0
        PatternUpd3[np.isnan(PatternUpd3)]=0
        PatternDnd1[np.isnan(PatternDnd1)]=0
        PatternDnd2[np.isnan(PatternDnd2)]=0
        PatternDnd3[np.isnan(PatternDnd3)]=0
        PatternUp=PatternUpd1+PatternUpd2+PatternUpd3
        PatternDn=PatternDnd1+PatternDnd2+PatternDnd3
        return PatternUp,PatternDn
