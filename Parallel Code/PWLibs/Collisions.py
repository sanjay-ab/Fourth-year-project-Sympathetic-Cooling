import numpy as np
import scipy.constants as sc
from math import isnan
import scipy.stats
from sys import platform
from scipy.interpolate import interp1d

amu = sc.physical_constants["atomic mass constant"][0] #atomic mass unit
mLi = 6.941*amu #mass of lithium in kg
mH = 1.008*amu #mass of hydrogen in kg
M = mLi + mH #total mass of both atoms
reduced_mass = (mLi*mH)/M

if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"
number_density_profile_data = np.genfromtxt(f"Input{slash}Li_init_Pos140uK.csv", delimiter=',') #get lithium number density divided by N (N = no. lithium atoms) at 140uK for each cartesian direction (Density is symmetrical in x, y, z)
cross_section_data = np.genfromtxt(f"Input{slash}mom_transfer_cross_section_SI.csv",delimiter = ",") #momentum transfer cross section for different collision energy
number_density_profile = interp1d(number_density_profile_data[:,0],number_density_profile_data[:,1],kind='linear',assume_sorted=True, bounds_error=False) #interpolator for the number density profile
cross_section = interp1d(cross_section_data[:,0],cross_section_data[:,1], kind='linear',assume_sorted=True, bounds_error=False) #interpolator for cross section
dist = scipy.stats.johnsonsu(-0.01636474,  1.635485,   -0.01688527,  1.14395079) #velocity distn of lithiums for 140uK

def check_collisions(HRange, N, dt, gamma): 
    """Function calculates the probability of a collision for each hydrogen atom and computes and elastic collision if one occurs"""
    indicies = [] #array holding the indices of hydrogen atoms that have escaped the trap - i.e. rows to be deleted.
    #gamma is the inelastic to elastic collision ratio
    for n, atom in enumerate(HRange):
        pos = np.absolute(atom[:3]) #cartesian position of the hydrogen particle in meters

        if isnan(pos[0]): #if particle is not in the trap
            indicies.append(n) #append to array to remove the row at the end of function
            continue

        tot_number_density = 0 #variable to hold the number density of the lithium

        for coord in pos:
            if coord >= 3e-3: #if coordinate is greater than 3e-3 then there is no number density since the particle is out of the lithium atom cloud
                tot_number_density = 0
                break
            else:
                tot_number_density += number_density_profile(coord) #for each coordinate x, y, z find number density in that axis for the particle and add them together for an average.

        if tot_number_density == 0:
            continue #if no number density then move onto next particle in loop
        else:
            tot_number_density = tot_number_density*(N/3) #calculate average number density at the hydrogen atom's postion

            v_h = atom[3:] #velocity of hydrogen atom
            v_li = np.array(dist.rvs(3)) #velocity of a randomly chosen lithium atom.
            v_r = np.linalg.norm(v_h - v_li) #relative velocity between the two atoms
            Energy = 0.5 * reduced_mass * v_r**2 #calculate collision energy - the energy available for the collision in the centre of mass frame (excludes energy due to movement of CoM)

            P = cross_section(Energy) * v_r * dt * tot_number_density #probability of a collision
        if np.random.rand()<(P*(1+gamma)): #check against prob of (elastic + inelastic) collision.
            #collision occurs
            if np.random.rand()<gamma:
                #inelastic collision occurs -> both particles are ejected from the trap.
                indicies.append(n) #mark row to remove hydrogen from trap.
                continue 

            vc = (mH*v_h + mLi * v_li)/M #centre of mass velocity
            p = (v_h - vc)*mH #transform momentum into centre of mass frame
            
            #now choose random b vector orthogonal to p
            c = [p[1],-p[0],0] #find two vectors that are mutually orthogonal and orthogonal to p
            d = np.cross(c,p)
            d = d/np.linalg.norm(d) #ensure both vectors are unit vectors
            c = c/np.linalg.norm(c)

            theta = np.random.rand() * 2*np.pi #choose random angle between 0 and 2pi

            b = np.cos(theta) * c + np.sin(theta) * d #choose b such that it is orthogonal to p and has unit norm as a random linear combination of c and d
            b = b * np.random.rand() # give b random magnitude between 0 and 1

            e = (p/np.linalg.norm(p)) * np.sqrt(1-np.dot(b,b)) + b #calculate unit vector that connects the two centres of the particles

            p = p - 2*(np.dot(p,e))*e #calculate centre of mass momentum after the collision

            v_h = p/mH + vc #convert velocities back into lab frame
            
            HRange[n,3:] = v_h #new velocity of hydrogen atom

    HRange = np.delete(HRange, indicies, axis=0) #deletes rows which are NaN, i.e. which hydrogen atoms have left the trap.

    return HRange