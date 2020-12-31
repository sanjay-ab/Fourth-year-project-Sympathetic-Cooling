import numpy as np
import scipy.constants as sc
from math import isnan
import scipy.stats
from sys import platform

amu = sc.physical_constants["atomic mass constant"][0] #atomic mass unit
mLi = 6.941*amu #mass of lithium and hydrogen in kg
mH = 1.008*amu
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"
number_density = np.genfromtxt(f"Input{slash}Li_init_Pos140uK.csv", delimiter=',') #get lithium number density divided by N (N = no. lithium atoms) at 140uK for each cartesian direction (Density is symmetrical in x, y, z)
cross_section = np.genfromtxt(f"Input{slash}mom_transfer_cross_section_SI.csv",delimiter = ",") #collision energies and corresponding momentum transfer cross section
inelastic_collision_prob = 0.001 #probability that a collision is inelastic - default = 1/1000
dist = scipy.stats.nct(4.817373592924506, 0.00748904541816595, -0.001638093164728389, 0.6623003499450637) #velocity distn of lithiums for 140uK

def checkProbLi(HRange, N, dt, T): #calculates the probability of a collision and determines whether one occurs.
    indicies = []
    for n, atom in enumerate(HRange):
        pos = np.absolute(atom[:3]) #cartesian position of the hydrogen particle in meters

        if isnan(pos[0]): #if particle is not in the trap
            indicies.append(n) #append to array to remove the row at the end of function
            continue

        tot_number_density = 0 #variable to hold the number density of the lithium

        for coord in pos:
            if coord >= 2.7e-3: #if coordinate is greater than 2.7e-3 then there is no number density since the particle is out of the lithium atom cloud
                tot_number_density = 0
                break
            else:
                loc = np.searchsorted(number_density[:,0],coord,side='right') #find index of number density
                tot_number_density += number_density[loc,1] #for each coordinate x, y, z find number density in that axis for the particle and add them together for an average.

        if tot_number_density == 0:
            continue #if no number density then move onto next particle in loop
        else:
            tot_number_density = tot_number_density*(N/3) #calculate average number density at the hydrogen atom's postion

            v_h = atom[3:] #velocity of hydrogen atom
            v_li = np.array(dist.rvs(3)) #velocity of a randomly chosen lithium atom.
            v_r = np.linalg.norm(v_h - v_li) #relative velocity between the two atoms
            Energy = 0.5 * (mH * np.dot(v_h,v_h) + mLi * np.dot(v_li,v_li)) #calculate collision energy

            loc = np.searchsorted(cross_section[:,0], Energy, side='right') #find index of cross section
            CS = 0.5*(cross_section[loc,1] + cross_section[loc-1,1]) #calculate average cross section from the cross sections corresponding to energies adjacent to E.

            P = CS * v_r * dt * tot_number_density #probability of a collision

        if np.random.rand()<P:
            #collision occurs
            if np.random.rand()<inelastic_collision_prob:
                #inelastic collision occurs -> both particles are ejected from the trap.
                N = N - 1 #reduce number of lithium atoms by 1
                T = ( np.sqrt(T) + np.sqrt((np.pi*mLi)/(8*sc.k)) * ( - np.linalg.norm(v_li)/N) )**2 #temperature change if atom is ejected from trap
                indicies.append(n) #mark row in HRange for removal -> hydrogen removed from trap.
                continue #continue to next loop

            vc = (mH*v_h + mLi * v_li)/(mH + mLi) #centre of mass velocity
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

            newv_li = -p/mLi + vc

            if np.linalg.norm(newv_li) > 16.378: #checks if atom has energy greater than the trap depth
            #calculates the new temperature of the lithium distribution by changing the mean speed of the distn
                T = ( np.sqrt(T) + np.sqrt((np.pi*mLi)/(8*sc.k)) * ( - np.linalg.norm(v_li)/N) )**2 #atom is ejected from the trap
                N = N - 1 #reduce number of lithium atoms by 1
            else:
                T = ( np.sqrt(T) + np.sqrt((np.pi*mLi)/(8*sc.k)) * ((np.linalg.norm(newv_li) - np.linalg.norm(v_li))/N) )**2 #atom remains in the trap

    HRange = np.delete(HRange, indicies, axis=0) #deletes rows which are NaN, i.e. which hydrogen atoms have left the trap.

    return HRange, N, T