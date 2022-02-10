import numpy as np
import random
import math

def rmsd(x,y):
    Nx = len(x)
    Ny = len(y)
    if Nx!=Ny:
        print('WARNING! Nx!=Ny. Could not calculate rmsd value')
        return 'WARNING! Nx!=Ny. Could not calculate rmsd value'
    delta = 0
    for i in range(Nx):
        delta += (x[i]-y[i])*(x[i]-y[i])
    delta = np.sqrt(delta/(Nx-1))
    return delta

def partition_holders_all(Nthr,Nsteps,minlength):   
    interval     = minlength                           # Spacing between start points
    end_interval = Nsteps-interval+1             # +1 because of the indexing and how arange works 
    startpoints  = np.arange(0,end_interval, interval) # Points to start the calculation of the rmsd.
    numberofsamples = len(startpoints)                 # Number of such walks. Will be the number of lists
    len_all = 0
    partitioned_walks_holder = []
    time_holder              = []
    steps_holder             = []
    lengths                  = []
    for i in range(numberofsamples):
        length = Nsteps-i*interval
        len_all += length
        lengths.append(length)
        partitioned_walks_holder.append(np.zeros((Nthr,length))) 
        time_holder.append(np.zeros(length))
        steps_holder.append(np.arange(0,length))
    return time_holder, steps_holder, partitioned_walks_holder, numberofsamples, len_all, lengths


def partition_dist_all(thesepositions,partition_walks, Nsteps, minlength, thisthr, Nthr): # 
    interval     = minlength                           # Spacing between start points
    end_interval = Nsteps-interval+1                   # +1 because of the indexing and how arange works 
    startpoints  = np.arange(0,end_interval, interval) # Points to start the calculation of the rmsd.
    numberofsamples = len(startpoints)                 # Number of such walks. Will be the number of lists
    for i in range(numberofsamples): # Looping over the number of partitioned walks
        startindex = startpoints[i]
        startpoint = thesepositions[startindex]
        counter    = 1
        length     = Nsteps-i*interval
        these_rmsd = np.zeros((Nthr,length))
        this_index = startindex+counter
        while this_index<Nsteps:
            this_point = thesepositions[this_index]
            distvec    = this_point-startpoint
            rmsd       = np.dot(distvec,distvec)
            these_rmsd[thisthr,counter] += rmsd                 # Only setting some elements to be non-zero: Those that belong to the current threshold
            this_index+=1
            counter   +=1 
        partition_walks[i] += these_rmsd
    return partition_walks

def partition_holders_averaged(Nsteps,minlength): 
    interval     = minlength                           # Spacing between start points
    end_interval = Nsteps-interval+1             # +1 because of the indexing and how arange works 
    startpoints  = np.arange(0,end_interval, interval) # Points to start the calculation of the rmsd.
    numberofsamples = len(startpoints)                 # Number of such walks. Will be the number of lists
    len_all = 0
    partitioned_walks_holder = []
    time_holder              = []
    steps_holder             = []
    lengths                  = []
    for i in range(numberofsamples):
        print('start of walk', i, ':',startpoints[i])
        length = Nsteps-i*interval
        len_all += length
        lengths.append(length)
        partitioned_walks_holder.append(np.zeros(length)) # Smaller than it's original counterpart
        time_holder.append(np.zeros(length))
        steps_holder.append(np.arange(0,length))
    return time_holder, steps_holder, partitioned_walks_holder, numberofsamples, len_all, lengths, startpoints

def partition_dist_averaged(thesepositions,partition_walks, Nsteps, minlength): 
    interval     = minlength                           # Spacing between start points
    end_interval = Nsteps-interval+1             # +1 because of the indexing and how arange works 
    startpoints  = np.arange(0,end_interval, interval) # Points to start the calculation of the rmsd.
    numberofsamples = len(startpoints)                 # Number of such walks. Will be the number of lists
    for i in range(numberofsamples): # Looping over the number of partitioned walks
        startindex = startpoints[i]
        startpoint = thesepositions[startindex]
        counter    = 1
        length     = Nsteps-i*interval
        these_rmsd = np.zeros(length)
        this_index = startindex+counter
        while this_index<Nsteps:
            this_point = thesepositions[this_index]
            distvec    = this_point-startpoint
            rmsd       = np.dot(distvec,distvec)
            these_rmsd[counter] += rmsd 
            this_index+=1
            counter   +=1 
        partition_walks[i] += these_rmsd
    return partition_walks
