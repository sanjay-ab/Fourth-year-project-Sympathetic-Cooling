import numpy as np
import time
from sys import platform
from PWLibs.H_GS_E import Hzeeman	# Import the Hydrogen Zeeman energy calculator module
from PWLibs.ARBInterp import tricubic	# Tricubic interpolator module
from PWLibs.Trap_Dist import makeH	# Module for making random atom distributions
from PWLibs.TrapVV import rVV	# Velocity Verlet numerical integrator module 
from PWLibs.collisions import checkProbLi #Module for calculating collisions
from mpl_toolkits.mplot3d import Axes3D #allows graphing in 3D
import matplotlib.pyplot as plt 
import scipy.constants as sc 

mH = 1.008*sc.physical_constants["atomic mass constant"][0] # mass H
N = 1e4	# no. of hydrogen atoms to create
Nli = 1e7 # Number of lithium atoms.
T = 0.1 # temperature of sample, K
Tli = 140e-6 #temperature of lithium cloud.
rsd = 1e-3 # standard deviation of simulated atom initial positions

t = 0	# time elapsed
dt = 1e-6	# timestep of simulation
t_end = 1e-5	# when do we want to stop the simulation

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

print("Loading field:")
trapfield=np.genfromtxt(f"Input{slash}Trap.csv", delimiter=',') # Load MT-MOT magnetic field
trapfield[:,:3]*=1e-3 # Modelled the field in mm for ease, make ;it m
# This is a vector field but the next piece of code automatically takes the magnitude of the field
print("Done")

print("Calculating Zeeman energies across grid:")
HEnergies = Hzeeman(trapfield)	# this instantiates the H Zeeman energy class with the field
# This class instance now contains 4 Zeeman energy fields for the different substates of H
# For now we are only interested in the F=1, mF=1 state, HEnergies.U11
# Others are U00, U1_1, U10
print("Done")

print("Loading interpolator:")
HU11Interp = tricubic(HEnergies.U11,'quiet')	# This creates an instance of the tricubic interpolator for this energy field
# We can now query arbitrary coordinates within the trap volume and get the interpolated Zeeman energy and its gradient
print("Done")

print("Making atom sample:")
# make a sample of Li atoms using the parameters from the top of the script

HRange = makeH(N, T, rsd)	# array of Nx6, N atoms, each row is x, y, z, vx, vy, vz

#let hydrogens move for 100us to get a more accurate initial distn
while t < 1e-4:
	HRange = rVV(HU11Interp, HRange, dt, mH)
	t += dt

#HRange = np.genfromtxt(f"Input{slash}H_end, t=1, dt=1e-5, Nli = 1e7.csv", delimiter=',') #take initial hydrogen distn from end distn of previous simulation
HRange_init = HRange.copy() # make a copy of the initial distribution to save out, so we can compare them if we want
print("Done")

print("Solving particle motion:")
#moves all particles velocities and postions through a time dt. Uses the rk4 integration since we need to integrate over the potential to find the path of the particles.
#we also need the interpolator since the field is not known continuously we need to need to interpolate between known points to get the whole field

loop=time.perf_counter()	# use this counter variable to display a progress report in the loop below
timer = loop #use this to display time elapsed
t = 0 #reset time for simulation

while t < t_end:

	HRange = rVV(HU11Interp, HRange, dt, mH)	# this uses the imported Runge-Kutta integrator to iterate the particle positions and velocities
	HRange, Nli, Tli = checkProbLi(HRange, Nli, dt, Tli) #checks whether a collision occurs and carries out the calculation if one does.

	t += dt	# increment time elapsed

	if time.perf_counter()- loop > 30:	# every 30 seconds this will meet this criterion and run the status update code below
		print ('Loop ' + '{:.1f}'.format(100 * (t / t_end)) + ' % complete. Time elapsed: {}s'.format(int(time.perf_counter()-timer)))	# percentage complete
		loop = time.perf_counter()	# reset status counter

print(f"Done \n Time Elapsed: {int(time.perf_counter()-timer)}")

print("Saving output:")
np.savetxt(r"C:\Users\Sanjay\Documents\Uni stuff\Physics\Project\Code\Sequential Code\Output\H_init.csv", HRange_init, delimiter=',')
np.savetxt(r"C:\Users\Sanjay\Documents\Uni stuff\Physics\Project\Code\Sequential Code\Output\H_end.csv", HRange, delimiter=',')
print("Done")

def plotting():
	"""Function takes care of graphing the result of the simulation"""

	print("Graphing Output")
	init_speeds = [np.linalg.norm(HRange_init[i,3:]) for i in range(len(HRange_init))]
	speeds = [np.linalg.norm(HRange[i,3:]) for i in range(len(HRange))]
	kinetic_energy = [0.5* mH* s**2 for s in speeds]
	init_kinetic_energy = [0.5* mH* s**2 for s in init_speeds]

	colour = colours(speeds,max(speeds)) #speeds is array of speeds of each hydrogen atom. colours is an array of colour values to show those speeds on a graph.
	init_colour = colours(init_speeds,max(init_speeds))

	print(f"Number of hydrogen left in trap: {len(HRange)}")
	print(f"Number of lithium left in trap: {Nli}")
	print(f"Temperature of lithium in trap: {Tli}")

	plot2 = plt.figure('3D Final Positions of Particles')
	ax2 = plot2.add_subplot(111, projection='3d') #plot final positions of particles in 3d while colour coding speeds 
	ax2.set_xlabel("x(mm)", fontsize = 10)
	ax2.set_ylabel("y(mm)", fontsize = 10)
	ax2.set_zlabel("z(mm)", fontsize = 10)
	ax2.set_xlim3d([-10,10])
	ax2.set_ylim3d([-10,10])
	ax2.set_zlim3d([-10,10])
	ax2.scatter(HRange[:,0]*1000,HRange[:,1]*1000,HRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Final Speeds of hydrogen particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(speeds, bins= "auto", zorder = 1,density = True) #histogram of final speeds of hydrogen atoms.

	_ = plt.figure('Initial Speeds of hydrogen particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(init_speeds, bins= "auto", zorder = 1,density = True) #histogram of initial speeds of hydrogen atoms.

	_ = plt.figure('Final Kinetic Energy of hydrogen particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of final kinetic energy of hydrogen atoms.

	_ = plt.figure('Initial Kinetic Energy of hydrogen particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(init_kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of initial kinetic energy of hydrogen atoms.

	_ = plt.figure('Final Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(HRange[:,0]*1000,HRange[:,1]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Initial Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(HRange_init[:,0]*1000,HRange_init[:,1]*1000, c=init_colour, s=0.5) #plots positions of initial particles and colour codes them according to their speeds.

	_ = plt.figure('Final Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(HRange[:,2]*1000,HRange[:,5],s=1)

	_ = plt.figure('Initial Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(HRange_init[:,2]*1000,HRange_init[:,5],s=1)

	plt.show()

def colours(speeds,maxSpeed):
	"""function that creates a colour key based on speeds for showing speed in plotting"""
	colour = []

	for  item in speeds: #create an array for colour coding graphs.
		colour.append([float(item/maxSpeed), 0 ,float(1-item/maxSpeed)]) #colour moves from blue (cold) to red (hot)

	return colour #colour not turned into an np array as it will not be modified again

plotting() #call plotting function to show graphs.