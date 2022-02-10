import numpy as np
import random
import sys
import math

## Placing the bead in the brush

# System info
spacing = 15
charge  = -1
T       = 3

# Number of files out, etc.
Nfiles = 1000
seed   = 23 

# Technicalities: Placement of bead
tol        = 0.2 # The closest we'll allow the bead to be to a brush 'atom'
substr_tol = 1.1 # the closest we'll allow the bead to be to the substrate # Previously: 0.7
distrange  = 0.5  # The spread of z-values of the bead. E.g. from 0.5 to 1.5. Will be from substr_tol to substr_tol+distrange. # Previously: 0.5

startlines = 23 # The number of lines we are going to read from the data.-infile and write to the data.-outfile

vx = np.sqrt(T)
vy = np.sqrt(T)
vz = np.sqrt(T)

xran = np.zeros(Nfiles)
yran = np.zeros(Nfiles)
zran = np.zeros(Nfiles)

brush_beads = [] # We know where the substrate atoms are (at z=0), but we should know where the beads are so that we can avoid overlap

foldername  = '/Diffusion_nostiffness/Spacing%i/Initial_configurations/' % spacing
outfilename = 'data.bead_wsubstr_eq_N909_d%i_charge%i_mass1_nostiffness_file' % (spacing,charge)
infilename = 'data.chaingrids_substrate_N909_Nchains9_Ly3_gridspacing%i_twofixed_charge%i_mass1_nostiffness_equilibrated' % (spacing, charge)
infile     = open(infilename, 'r')
lines      = infile.readlines()
Nlines     = len(lines)
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
        i = int(words[0])
        molID[i-1]     = int(words[1])
        this_atom_type = int(words[2])
        xposition      = float(words[4])
        yposition      = float(words[5])
        zposition      = float(words[6])
        atom_type[i-1] = this_atom_type
        qs[i-1]        = float(words[3])
        xpos[i-1]      = xposition
        ypos[i-1]      = yposition
        zpos[i-1]      = zposition
        # Check if this is a brush atom, then append it to the brush list if it is.
        if this_atom_type==1 or this_atom_type==2:
            brush_beads.append(np.array([xposition,yposition,zposition]))
# ten elements in the lines that we want. Not that number of elements in any other line.
# atom style full: atom-ID molecule-ID atom-type q x y z
# We want to keep the information about the velocities and the bonds and angles too

### Part where I randomly place the diffusing bead.

beadpos = []

i = 0
while i<Nfiles:
    breakit = False
    xran[i] = random.uniform(xmin,xmax)
    yran[i] = random.uniform(ymin,ymax)
    zran[i] = random.uniform(substr_tol,substr_tol+distrange) # Random placement not too far from the substrate.
    rran = np.array([xran[i],yran[i],zran[i]])
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
        i+=1

print('beadpos:',beadpos)

freebead_number = N_atoms+1
############# Write to file. Multiple times. #############
for i in range(Nfiles):
    filename = foldername + outfilename + '%i' % (i+1)
    outfile = open(filename, 'w')
    # Write lines from infile to the outfile.
    for j in range(startlines-1):
        if len(lines[j].split())==2 and lines[j].split()[1]=='atoms':
            outfile.write('%i atoms\n' % (N_atoms+1))
        else:
            outfile.write(lines[j])
    for j in range(startlines-1,startlines+N_atoms-1):
        words = lines[j].split()
        if len(words)>0:
            outfile.write('%i %i %i %.16e %.16e %.16e %.16e\n' % (int(words[0]), int(words[1]), int(words[2]), float(words[3]), float(words[4]), float(words[5]), float(words[6])))
        else:
            break
    outfile.write('%i %i 3 0 %.16e %.16e %.16e\n' % (freebead_number, max(molID)+1,xran[i],yran[i],zran[i])) # atom-ID molecule-ID atom-type q x y z
    for j in range(N_atoms+3):                        # Writing velocities to file.
        outfile.write(lines[j+startlines+N_atoms-1])
    outfile.write('%i %.16e %.16e %.16e\n' % (freebead_number,vx,vy,vz))
    for j in range(2*N_atoms+4+startlines-2,Nlines):      # Write the rest of the lines to file. Everything from here on can be copy-pasted
        outfile.write(lines[j])
    outfile.close()                                   # This should be important since we write to multiple files
