'''

'''

from random import seed
from random import random
import numpy as np
from numba import jit, config, int32, float64, prange
from numba.core import types
from numba.experimental import jitclass
from numba.typed import Dict


## Flag to disable JIT
config.DISABLE_JIT = False

## Definitions of packet properties required by numba
pac_def = [
    ('flag', int32),      
    ('id', int32),
    ('cost', float64),
    ('nzp', float64),
    ('zc', int32),
    ('zp', float64),
    ('tau_p', float64),
    ('tau', float64),
    ('iscat',int32),
]

## Class that defines packet properties
@jitclass(pac_def)
class pac:
  def __init__(self, flag, id, cost, nzp, zc, zp, tau_p, tau, iscat):
    self.flag = flag
    self.id = id
    self.cost = cost
    self.nzp = nzp
    self.zc = zc
    self.zp = zp
    self.tau_p = tau_p
    self.tau = tau
    self.iscat = iscat

## Function that integrates a packet through the 1D plane-parallel grid
@jit(nopython=True, cache=True)
def tauint_1D_pp(ph, nlay, z, sig_ext, l, Jdot):

  # Initial tau is at zero
  ph.tau = 0.0

  # Do loop until tau is equal or more than sampled tau
  while (ph.tau < ph.tau_p):

    # Calculate dsz, the distance to the next vertical level
    if (ph.nzp > 0.0):
      # Packet travelling upward, find distance to upper level
      dsz = (z[ph.zc+1]-ph.zp)/ph.nzp
      zoffset = 1
    elif (ph.nzp < 0.0):
      # Packet traveling downward, find distance to lower level
      dsz = (z[ph.zc]-ph.zp)/ph.nzp
      zoffset = -1
    else:
      # Packet traveling directly in z plane
      # Return, packet does not move in z direction
      break
    
    # Calculate optical depth to level edge
    taucell = dsz * sig_ext[ph.zc]

    # Check if packet ends path in this layer
    if ((ph.tau + taucell) >= ph.tau_p):

      # Packet stops in this cell - move distance then exit loop
      d1 = (ph.tau_p-ph.tau)/sig_ext[ph.zc]
      ph.zp += d1 * ph.nzp

      # Update estimator
      Jdot[ph.zc,l] += d1 # Mean intensity estimator
      #Jdot[ph.zc,l] += d1*ph.nzp # Flux estimator

      # tau of packet is now the sampled tau
      ph.tau = ph.tau_p
    else:

      #Packet continues to level edge - update position, cell index and tau
      ph.zp += (dsz + 1.0e-12) * ph.nzp

      # Update estimator
      Jdot[ph.zc,l] += dsz # Mean intensity estimator
      #Jdot[ph.zc,l] += dsz*ph.nzp  # Flux estimator

      # Apply integer offset to cell number
      ph.zc += zoffset

      # Add tau of cell to tau counter of packet
      ph.tau += taucell

      # Check is packet has exited the domain
      if ((ph.zc > nlay-1) or (ph.zp >= z[-1])):
        # Packet has exited the domain at the top of atmosphere
        ph.flag = 1
        break
      elif((ph.zc < 0) or (ph.zp <= z[0])):
        # Packet has hit the lower part od the domain (surface)
        ph.flag = -2
        break

  return

## Function for incident stellar radiation properties
@jit(nopython=True, cache=True)
def inc_stellar(ph, nlay, z, mu_z):

  ph.cost = -mu_z   # Initial direction is zenith angle
  ph.nzp = ph.cost  # give nzp cost
  ph.zc = nlay-1    # Initial layer number is top of atmosphere
  ph.zp = z[-1] - 1.0e-12 # Initial position is topic atmosphere minus a little bit

  return

## Function to scatter a packet into a new direction
@jit(nopython=True, cache=True)
def scatter(ph, g):

  # Find the type of scattering the packet undergoes 
  # (1 = isotropic, 2 = Rayleigh, 3 = Henyey-Greenstein)
  match ph.iscat:

    case 1:
      # Isotropic scattering
      ph.cost = 2.0 * random() - 1.0
      ph.nzp = ph.cost
      return
    
    case 2:
      # Rayleigh scattering via direct spherical coordinate sampling
      # Assumes non-polarised incident packet
      q = 4.0*random() - 2.0
      u = (-q + np.sqrt(1.0 + q**2))**(1.0/3.0)
      bmu = u - 1.0/u

    case 3:
      # Sample from single HG function
      if (g != 0.0):
        hgg = g
        g2 = hgg**2

        bmu = ((1.0 + g2) - \
          ((1.0 - g2) / (1.0 - hgg + 2.0 * hgg * random()))**2) \
          / (2.0*hgg)
      else:
        # g = 0, Isotropic scattering
        ph.cost = 2.0 * random() - 1.0
        ph.nzp = ph.cost
        return
      
    case _ :
      # Invalid iscat value
      print('Invalid iscat chosen: ' + str(ph.iscat) + ' doing isotropic')
      ph.cost = 2.0 * random() - 1.0
      ph.nzp = ph.cost
      return
    
  # Now apply rotation if non-isotropic scattering
  # Change direction of packet given by sampled direction
 
  # Calculate change in direction in grid reference frame
  if (bmu >= 1.0):
    # Packet directly forward scatters - no change in direction
    ph.cost = ph.cost
  elif (bmu <= -1.0):
    # Packet directly backward scatters - negative sign for cost
    ph.cost = -ph.cost
  else:
    # Packet scatters according to sampled cosine and current direction
    # Save current cosine direction of packet and calculate sine
    costp = ph.cost
    sintp = 1.0 - costp**2
    if (sintp <= 0.0):
      sintp = 0.0
    else:
      sintp = np.sqrt(sintp)

    # Randomly decide if scattered in +/- quadrant direction and find new cosine direction
    sinbt = np.sqrt(1.0 - bmu**2)
    ri1 = 2.0 * np.pi * random()
    if (ri1 > np.pi):
      cosi3 = np.cos(2.0*np.pi - ri1)
      # Calculate new cosine direction
      ph.cost = costp * bmu + sintp * sinbt * cosi3
    else: #(ri1 <= pi)
      cosi1 = np.cos(ri1)
      ph.cost = costp * bmu + sintp * sinbt * cosi1

    #Give nzp the cost value
    ph.nzp = ph.cost

  return

## Function to scatter from a surface (Assumed Lambertian)
@jit(nopython=True, cache=True)
def scatter_surf(ph, z):
    
    ph.cost = np.sqrt(random()) # Sample cosine from Lambertian cosine law
    ph.nzp = ph.cost            # Give nzp cost value
    ph.zc = 0                   # Initial layer is lowest (0)
    ph.zp = z[0] + 1.0e-12      # Initial position is lowest altitude plus a little bit

    return

## Main gCMCRT function for wavelength and packet loop
#def gCMCRT_main(Nph, nlay, nwl, cross, VMR_cross, n_ray, ray, VMR_ray, g, nd, Iinc, surf_alb, mu_z, z):
@jit(nopython=True, cache=True, parallel=True)
def gCMCRT_main(Nph, nlay, nwl, all_sp, ph_sp, ray_sp, nd_sp, cross, ray, Iinc, mu_z, dze):

  # print('in gCMCRT : stopping')
  # print('Nph', str(Nph))
  # print('nlay', nlay)
  # print('nwl', nwl)
  # print('all_sp',all_sp)
  # print('ph_sp', ph_sp)
  # print('ray_sp', ray_sp)
  # sp = 'O2'
  # print('nd_cross', nd_sp[:,all_sp.index(sp)])
  # print('cross', cross['O2'])
  # print('ray', ray['H2'])
  # print('Iinc', Iinc)
  # print('mu_z', mu_z)
  # print('dze',dze)
  # quit()

  # Number of levels
  nlev = nlay + 1

  # Initalise Jdot
  Jdot = np.zeros((nlay, nwl))

  # Initialise packet energy array
  e0dt = np.zeros(nwl)

  # Initialise surface albedo
  surf_alb = np.zeros(nwl)

  # Initialise assymetrey factor
  g = np.zeros((nlay, nwl))

  # Reconstruct altitude grid from dze
  z = np.zeros(nlev)
  for i in range(nlay):
    z[i+1] = z[i] + dze[i]

  # Set number of cross section species and rayleigh
  n_cross = len(ph_sp)
  n_ray = len(ray_sp)

  #for l in range(nwl):
  for l in prange(nwl):

    # Working arrays
    sig_ext = np.zeros(nlay)
    sig_sca = np.zeros(nlay)

    # Find the total absorption opacity from photocross sections
    for i in range(n_cross):

      #print(all_sp.index())
      sig_ext[:] += nd_sp[:,all_sp.index(ph_sp[i])] * cross[i,l] # Absorption opacity [cm-1]

    # Find the total rayleigh opacity from Rayleigh species cross sections
    for i in range(n_ray):
      sig_sca[:] += nd_sp[:,all_sp.index(ray_sp[i])] * ray[i,l]   # Scattering opacity [cm-1]

    # Extinction = photocross + Rayleigh
    sig_ext[:] += sig_sca[:]

    # Find the tau grid starting from upper boundary - only needed for testing
    # tau = np.zeros(nlev)
    # tau[-1] = 0.0
    # for k in range(nlay-1,-1,-1):
    #   tau[k] = tau[k+1] + sig_ext[k] * dze[k]

    # Find scattering albedo - MCRT only works well up to around 0.98 ssa, so limit to that
    alb = np.zeros(nlay)
    alb[:] = np.minimum(sig_sca[:]/sig_ext[:],0.95)

    # Energy carried by each packet for this wavelength
    e0dt[l] = (mu_z * Iinc[l])/float(Nph)

    # Initialse random seed for this wavelength 
    seed(int(1 + (l+1)**2 + np.sqrt(l+1)))

    # Packet number loop
    for n in range(Nph):

      # Initialise packet variables (janky python way)
      flag = 0
      id = l*Nph + n
      iscat = 2

      # Initialise photon packet with initial values
      ph = pac(flag, id, 0.0, 0.0, 0, 0.0, 0.0, 0.0, iscat)

      # Place photon at top of atmosphere with negative zenith angle
      inc_stellar(ph, nlay, z, mu_z)

      # Scattering loop
      while(ph.flag == 0):

        # Sample a random optical depth of the photon
        ph.tau_p = -np.log(random())

        # Move the photon through the grid given by the sampled tau
        tauint_1D_pp(ph, nlay, z, sig_ext, l, Jdot)

        # Check status of the photon after moving
        match ph.flag:
          case 1:
            # Photon exited top of atmosphere, exit loop
            break
          case -2: 
            # Photon hit surface surface, test surface albedo
            if (random() < surf_alb[l]):
              # Packet scattered off surface (Lambertian assumed)
              scatter_surf(ph, z)
            else:
              # Packet was absorbed by the surface
              break
          case 0:
            # Photon still in grid, test atmospheric albedo
            if (random() < alb[ph.zc]):
              # Packet get scattered into new direction
              scatter(ph, g[ph.zc,l])
            else:
              # Packet was absorbed, exit loop
              break
          case _:
            # Photon has an invalid flag, raise error
            print(str(ph.id) + ' has invalid flag: ' + str(ph.flag))

    ## Scale estimator to dze
    Jdot[:,l] = e0dt[l]*Jdot[:,l]/dze[:]

  return Jdot