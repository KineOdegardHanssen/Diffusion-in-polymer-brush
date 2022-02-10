import numpy as np
import random
import sys
import math

## Placing the bead in the brush

# System info
radius  = 1
spacing = 4.5
charge  = -1
T       = 3

# Number of files out, etc.
Nconfigs    = 100
Nplacements = 10
Nfiles = Nplacements
seed   = 23

# Technicalities: Placement of bead
tol        = 0.2*radius # The closest we'll allow the bead to be to a brush 'atom'
substr_tol = 1.1*radius # the closest we'll allow the bead to be to the substrate # Previously: 0.7
distrange  = 0.5*radius  # The spread of z-values of the bead. E.g. from 0.5 to 1.5. Will be from substr_tol to substr_tol+distrange. # Previously: 0.5

startlines = 23 # The number of lines we are going to read from the data.-infile and write to the data.-outfile 

vx = np.sqrt(T)
vy = np.sqrt(T)
vz = np.sqrt(T)

xran = np.zeros(Nfiles)
yran = np.zeros(Nfiles)
zran = np.zeros(Nfiles)

brush_beads = [] # We know where the substrate atoms are (at z=0), but we should know where the beads are so that we can avoid overlap

infoldername  = '/Diffusion_staticbrush/Spacing'+ str(spacing) +'/Initial_configs/Before_bead/'
outfoldername = '/Diffusion_staticbrush/Spacing' + str(spacing) + '/Radius' +str(radius)+'/Initial_configs/'


for i in range(Nconfigs):
    outfilename = 'data.config%i_beadplacement' % (i+1)
    infilename  = infoldername+'data.config%i' % (i+1)
    infile      = open(infilename, 'r')
    lines       = infile.readlines()
    Nlines      = len(lines)
    infile.close()
    
    N_atoms      = int(lines[2].split()[0]) # Line 2
    N_atomtypes  = int(lines[3].split()[0]) # Line 3
    N_bonds      = int(lines[4].split()[0]) # Line 4
    N_bondtypes  = int(lines[5].split()[0]) # Line 5
    N_angles     = int(lines[6].split()[0]) # Line 6
    N_angletypes = int(lines[7].split()[0]) # Line 7
    
    
    xmin = float(lines[9].split()[0]) 
    xmax = float(lines[9].split()[1])
    ymin = float(lines[10].split()[0])
    ymax = float(lines[10].split()[1])
    zmin = float(lines[9].split()[0])
    zmax = float(lines[9].split()[1])
    
    molID     = np.zeros(N_atoms)
    atom_type = np.zeros(N_atoms)
    qs        = np.zeros(N_atoms)
    xpos      = np.zeros(N_atoms)
    ypos      = np.zeros(N_atoms)
    zpos      = np.zeros(N_atoms)
    bond_atom1 = [] 
    bond_atom2 = []
    
    keyword = 'None'
    for line in lines:
        words = line.split()
        if len(words)>0 and words[0]=='Velocities':
            break 
        
        if len(words)==10:
            j = int(words[0])
            molID[j-1]     = int(words[1])
            this_atom_type = int(words[2])
            xposition      = float(words[4])
            yposition      = float(words[5])
            zposition      = float(words[6])
            atom_type[j-1] = this_atom_type
            qs[j-1]        = float(words[3])
            xpos[j-1]      = xposition
            ypos[j-1]      = yposition
            zpos[j-1]      = zposition
            # Check if this is a brush atom, then append it to the brush list if it is.
            if this_atom_type==1 or this_atom_type==2:
                brush_beads.append(np.array([xposition,yposition,zposition]))
    # ten elements in the lines that we want. Not that number of elements in any other line.
    # atom style full: atom-ID molecule-ID atom-type q x y z
    # We want to keep the information about the velocities and the bonds and angles too
    
    ### Part where we randomly place the diffusing bead.
    
    beadpos = []
    
    j = 0
    while j<Nfiles: 
        breakit = False
        xran[j] = random.uniform(xmin,xmax)
        yran[j] = random.uniform(ymin,ymax)
        zran[j] = random.uniform(substr_tol,substr_tol+distrange) # Random placement not too far from the substrate.
        rran = np.array([xran[j],yran[j],zran[j]])
        for rbr in brush_beads:
            distvec = rbr-rran
            dist2   = np.dot(distvec,distvec)
            if dist2<tol:
                breakit = True
                break
        if breakit==True:
            continue
        else:
            beadpos.append(rran)
            j+=1
    
    freebead_number = N_atoms+1
    ############# Write to file. Multiple times. #############
    for j in range(Nfiles):
        filename = outfoldername + outfilename + '%i' % (j+1)
        outfile = open(filename, 'w')
        # Write lines from infile to the outfile.
        for k in range(startlines-1):
            if len(lines[k].split())==2 and lines[k].split()[1]=='atoms':
                outfile.write('%i atoms\n' % (N_atoms+1))
            else:
                outfile.write(lines[k])
        for k in range(startlines-1,startlines+N_atoms-1):
            words = lines[k].split()
            if len(words)>0:
                outfile.write('%i %i %i %.16e %.16e %.16e %.16e\n' % (int(words[0]), int(words[1]), int(words[2]), float(words[3]), float(words[4]), float(words[5]), float(words[6])))
            else:
                break
        outfile.write('%i %i 3 0 %.16e %.16e %.16e\n' % (freebead_number, max(molID)+1,xran[j],yran[j],zran[j])) # atom-ID molecule-ID atom-type q x y z 
        for k in range(N_atoms+3):                        # Writing velocities to file.
            outfile.write(lines[k+startlines+N_atoms-1])
        outfile.write('%i %.16e %.16e %.16e\n' % (freebead_number,vx,vy,vz))
        for k in range(2*N_atoms+4+startlines-2,Nlines):      # Write the rest of the lines to file. Everything from here on can be copy-pasted
            outfile.write(lines[k])
        outfile.close()                                   # This should be important since we write to multiple files
