# Generating Rayleigh scattering cross cestions
import numpy as np

def Ray_xsec(n_ref, wn, nd_stp, King):
  Ray_xsec = ((24.0 * np.pi**3 * wn**4)/(nd_stp**2)) \
     * ((n_ref**2 - 1.0)/(n_ref**2 + 2.0))**2  * King
  return Ray_xsec

# Constants for H Rayleigh opacity
c_s = 2.99792458e10
hp = 6.62607015e-27
m_el = 9.1093837015e-28
wl_ly = 121.567 * 1.0e-7
f_ly = c_s/wl_ly
w_l = (2.0 * np.pi * f_ly) / 0.75
a_fine = 7.2973525693e-3 # fine structure constant
Comp_e = hp/(m_el*c_s) # Compton wavelength for electrons
sigT = ((8.0*np.pi)/3.0) * ((a_fine * Comp_e)/(2.0*np.pi))**2 # Thomson cross section [cm2]

cp = [1.26537,3.73766,8.8127,19.1515,39.919,81.1018,161.896,319.001,622.229,1203.82]

# Wavelength grid
wl_nm = np.arange(0.1,800.1, 0.1)
nwl = len(wl_nm)
wl_um = wl_nm / 1e3
wl_cm = wl_um / 1e4
wn = 1.0/wl_cm
freq = c_s/wl_cm


# Choose the species
sp = 'N2'

match sp:
  case 'H2':
    # Use Irwin (2009)+ parameters (same as Cox 2000)
    A = 13.58e-5 ; B = 7.52e-3
    nd_stp = 2.65163e19
    King = 1.0
  case 'NH3':
    # Use Irwin (2009)+ parameters
    A = 37.0e-5 ; B = 12.0e-3 ; DPol = 0.0922
    nd_stp = 2.65163e19
    King = (6.0  + 3.0 * DPol) / (6.0 - 7.0 * DPol)
  case 'He':
    # Use Thalman et al. (2014) parameters
    #A = 2283.0 ; B = 1.8102e13 ; C = 1.5342e10
    #nd_stp = 2.546899e19
    #King = 1.0
    # Use Irwin (2009)+ parameters
    A = 3.48e-5 ; B = 2.3e-3
    nd_stp = 2.65163e19
    King = 1.0
  case 'CO':
    # Use Sneeps & Ubachs (2005) expression
    # Typo error in Sneeps & Ubachs corrected (Kitzmann per.com.)
    #A = 22851.0; B = 0.456e14 ; C = 71427.0**2
    #nd_stp = 2.546899e19
    #King = 1.0
    # Use Irwin (2009)+ parameters
    A = 32.7e-5 ; B = 8.1e-3
    nd_stp = 2.65163e19
    King = 1.0
  case 'CO2':
    # Use Sneeps & Ubachs (2005) expression
    # Error in Sneeps & Ubachs last term is 0.1218145e−6 not 0.1218145e−4 (See Kitzmann 2017 p 4 foootnote)
    # Error also in the multiplication factor (Kitzmann per.com.)
    #nd_stp = 2.546899e19
    # Use Irwin (2009)+ parameters
    A = 43.9e-5 ; B = 6.4e-3
    nd_stp = 2.65163e19
  case 'CH4':
    # Use Sneeps & Ubachs (2005) expression
    King = 1.0
    nd_stp = 2.546899e19
  case 'O2':
    # Use Thalman et al. (2014) parameters
    #A = 20564.8 ; B = 2.480899e13 ; C = 4.09e9
    #nd_stp = 2.68678e19
    # Use Irwin (2009)+ parameters
    A = 26.63e-5 ; B = 5.07e-3 ; DPol = 0.054
    nd_stp = 2.65163e19
    King = (6.0  + 3.0 * DPol) / (6.0 - 7.0 * DPol)
  case 'N2':
    # Use Thalman et al. (2014) parameters -
    # Note typo in Thalman et al. (2014) e13 error - used Sneep & Ubachs (2005) expression  
    #nd_stp = 2.546899e19
    # Use Irwin (2009)+ parameters
    A = 29.06e-5 ; B = 7.7e-3; DPol = 0.030
    nd_stp = 2.65163e19
    King = (6.0  + 3.0 * DPol) / (6.0 - 7.0 * DPol)
  case 'Ar':
    # Use Thalman et al. (2014) parameters
    #A = 6432.135 ; B = 286.06021e12 ; C = 14.4e9
    #King = 1.0
    #nd_stp = 2.546899e19
    # Use Irwin (2009)+ parameters
    A = 27.92e-5 ; B = 5.6e-3
    nd_stp = 2.65163e19
  case 'N2O':
    # Use Sneeps & Ubachs (2005) expression  
    nd_stp = 2.546899e19
  case 'NO':
    # Use Irwin (2009)+ parameters
    A = 28.9e-5 ; B = 7.4e-3
    nd_stp = 2.65163e19
    King = 1.0
  case 'H':
    # Lee & Kim (2004) expression
    King = 1
  case 'el' | 'e-':
    # Electrons
    King = 1
  case _:
    print('Species not availible: ', sp, 'stopping')
    quit()

# Calculate cross sections for each wavelength
cross = np.zeros(nwl)
for l in range(nwl):
  match sp:
    case 'H2' | 'He' | 'NH3' | 'CO' | 'O2' | 'N2' | 'NO' | 'CO2' | 'Ar':
      n_ref = A * (1.0 + B/wl_um[l]**2) + 1.0
      cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    #case  'Ar' | 'CO':
    #  if (wl_nm[l] < 200.0):
    #    cross[l]  = 0.0
    #    continue
    #  n_ref = (A + (B / (C - wn[l]**2)))/1e8 + 1.0
    #  cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    #case 'CO2':
    #  if (wl_nm[l] < 200.0):
    #    cross[l]  = 0.0
    #    continue
    #  n_ref = (1.1427e3 * ((5799.25 / (128908.9**2 - wn[l]**2)) + (120.05 / (89223.8**2 - wn[l]**2)) \
    #    + (5.3334 / (75037.5**2 - wn[l]**2)) + (4.3244 / (67837.7**2 - wn[l]**2)) \
    #    + (0.1218145e-6 / (2418.136**2 - wn[l]**2)))) + 1.0
    #  King = 1.1364 + 25.3e-12*wn[l]**2
    #  cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    case 'CH4':
      n_ref = (46662.0 + 4.02e-6*wn[l]**2)/1e8 + 1.0
      cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    #case 'O2':
    #  if (wl_nm[l] < 200.0):
    #    cross[l]  = 0.0
    #    continue
    #  n_ref = (A + (B / (C - wn[l]**2)))/1e8 + 1.0
    #  King = 1.09 + 1.385e-11*wn[l]**2 + 1.448e-20*wn[l]**4
    #  cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    #case 'N2':
    #  if (wl_nm[l] < 200.0):
    #    cross[l]  = 0.0
    #    continue
    #  if (wn[l] > 21360.0):
    #    A = 5677.465 ; B = 318.81874e12 ; C = 14.4e9
    #  else:
    #    A = 6498.2 ; B = 307.4335e12 ; C = 14.4e9
    #  n_ref = (A + (B / (C - wn[l]**2)))/1e8 + 1.0
    #  King = 1.034 + 3.17e-12*wn[l]
    #  cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    case 'N2O':
      if (wl_nm[l] < 200.0):
        cross[l]  = 0.0
        continue
      n_ref = (46890.0 + 4.12e-6*wn[l]**2)/1e8 + 1.0
      DPol = 0.0577 + 11.8e-12*wn[l]**2
      King = (3.0  + 6.0 * DPol) / (3.0 - 4.0 * DPol)
      cross[l] = Ray_xsec(n_ref,wn[l],nd_stp,King)
    case 'H':
      w = 2.0 * np.pi * freq[l]
      wwl = w/w_l
      # Lee and Kim (2004)
      if (wwl <= 0.6):
        # Low energy limit
        xsec = 0.0
        for p in range(10):
          xsec += (cp[p] * wwl**(2 * p))
        xsec *= wwl**4
      else:
        #  High energy limit (approaching Lyman alpha wavelengths)
        wb = (w - 0.75*w_l)/(0.75*w_l)
        xsec = (0.0433056/wb**2)*(1.0 - 1.792*wb - 23.637*wb**2 - 83.1393*wb**3 \
          - 244.1453*wb**4 - 699.473*wb**5)
      # Multiply by Thomson x-section
      cross[l] = np.maximum(xsec * sigT, 0.0)
    case 'el' | 'e-':
      cross[l] = sigT  
    case _:
      print('Invalid species: ', sp, 'stopping')
      quit()

# Output file
fname = sp + '_scat.txt'
f = open(fname,'w')
f.write('#lambda (nm)  cross(cm^2 molecule^-1)' + '\n')
for l in range(nwl):
  f.write("{:.2f}".format(wl_nm[l]) + ' ' + "{:.6e}".format(cross[l]) + '\n')
f.close()
