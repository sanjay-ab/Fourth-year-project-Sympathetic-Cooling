# Masters-project (Sympathetic cooling)
Some code I've written for my masters project on "Simulating the sympathetic cooling of atomic hydrogen with ultra-cold atomic lithium". This repo holds the code for the sympathetic cooling side of the simulation. The code for the laser cooling side is held in another repo. 

This was a collaborative project and only the files that I produced are held here. The additional libraries and data required for the program to run have not been included.

The program runs step-wise with a defineable time-step. Every time-step, the program solves the equation of motion of the hydogen atoms in the magnetic trap by numerically integrating using a velocity verlet algorithm. After the integration, it uses a probabalistic method to check whether each hydrogen atom collides with an atom from lithium atom cloud. If a collision occurs then a random velocity is chosen for the lithium atom and an elastic collision is computed. 

There are two sets of code, for parallel and sequential execution. The parallel code runs significantly faster than the sequential code. Although, it lacks some functionality, such as the ability to track the temperature and particle number of the lithium atom cloud, however, this is fairly insignificant anyway.
