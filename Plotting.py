import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Ellipse

import AstrometricData as AD
import CoordinateTransformations as CT

def plot(
		ADf: AD.AstrometricDataframe,
		output: str
	):

	plot_positions(
		positions = ADf.positions,
		file_name = os.path.join(output, "positions.pdf")
	)

	plot_proper_motions(
		positions = ADf.positions,
		proper_motions = ADf.proper_motions,
		file_name = os.path.join(output, "proper_motions.pdf")
	)

	plot_overlap_matrix(
		matrix = ADf.overlap_matrix,
		names = list(ADf.YlmQ_names.values()),
		file_name = os.path.join(output, "overlap_matrix.pdf")
	)

	plot_corner(
		np.loadtxt(os.path.join(output, 'posterior_samples.dat'))[:, 0:-2],
		names = list(ADf.almQ_names.values()),
		file_name = os.path.join(output, "corner.pdf")
	)

def plot_positions(
		positions: np.ndarray,
		file_name: str
	):
    """
    Plot the QSO positions in Mollweide projection
    """

    positions_Mollweide = CT.point_transform(
    							input_points = positions,
    						    input_coordinates = "geographic",
    						    output_coordinates = "Mollweide"
    						)

    N_obj = positions.shape[0]

    if N_obj <= 10000:
    	marker_size = 10.
    elif N_obj <= 100000:
    	marker_size = 1.
    elif N_obj <= 1000000:
    	marker_size = 0.05
    else:
    	marker_size = 0.01

    plt.figure(figsize = (20,10))

    ax = plt.subplot(111)

    ax.scatter(positions_Mollweide[:,0],
    		   positions_Mollweide[:,1],
    		   marker = 'o',
    		   c = 'black',
    		   s = marker_size
    		  )

    border_ellipse = Ellipse(
	    					xy = (0., 0.),
	    				  	width = 4.*np.sqrt(2),
	    				  	height = 2.*np.sqrt(2), 
	                      	edgecolor = 'black',
	                      	fc = 'None',
	                      	lw = 1.
                    	)

    ax.add_patch(border_ellipse)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(file_name, dpi=1200)
    plt.clf()

def plot_proper_motions(
		positions: np.ndarray,
		proper_motions: np.ndarray,
		file_name: str
	):
    max_N_pm = 1000

    if positions.shape[0] > max_N_pm:
        pm_to_show_ind = np.random.choice(positions.shape[0], size=max_N_pm)

        positions = positions[pm_to_show_ind]
        proper_motions = proper_motions[pm_to_show_ind]

    positions_Cartesian = CT.point_transform(
    							input_points = positions,
    						    input_coordinates = "geographic",
    						    output_coordinates = "Cartesian"
    						)

    proper_motions_Cartesian = CT.geographic_to_Cartesian_vector(positions, proper_motions)

    zeros_array = np.zeros(shape = proper_motions_Cartesian.shape)
    
    proper_motions_Cartesian_linspace = np.linspace(
    										start = zeros_array,
    										stop = proper_motions_Cartesian,
    										num = 10,
    										endpoint = True
    									)

    proper_motions_geographic_linspace = CT.point_transform(
    											input_points = positions_Cartesian + proper_motions_Cartesian_linspace,
    						    			 	input_coordinates = "Cartesian",
    						    			 	output_coordinates = "geographic"
    						    			)

    proper_motions_Mollweide_linspace = CT.point_transform(
    										input_points = proper_motions_geographic_linspace,
    						    			input_coordinates = "geographic",
    						    			output_coordinates = "Mollweide"
    						    		)

    plt.figure(figsize = (20,10))

    ax = plt.subplot(111)

    for line_Mollweide, line_geographic in zip(
    		np.swapaxes(proper_motions_Mollweide_linspace, 0, 1),
    		np.swapaxes(proper_motions_geographic_linspace, 0, 1)
    	):
    	if np.abs(line_geographic[0,0] - line_geographic[-1,0]) > np.pi: continue

    	part_line = line_Mollweide[0:-1,:]

    	start_arrow = line_Mollweide[-2,:]
    	delta_arrow = line_Mollweide[-1,:] - line_Mollweide[-2,:]

    	ax.plot(
    		part_line[:,0],
    		part_line[:,1],
    		c = 'b',
    		linewidth = 1.
    	)

    	ax.arrow(
    		x = start_arrow[0],
    		y = start_arrow[1],
    		dx = delta_arrow[0],
    		dy = delta_arrow[1],
    		linewidth = 0.01,
    		width = 0.005,
    		color = 'b',
    		length_includes_head = True
    	)

    border_ellipse = Ellipse(
		xy = (0., 0.),
		width = 4.*np.sqrt(2),
		height = 2.*np.sqrt(2), 
		edgecolor = 'black',
		fc = 'None',
		lw = 1.
	)

    ax.add_patch(border_ellipse)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(file_name, dpi=300)
    plt.clf()

def plot_overlap_matrix(
		matrix: np.ndarray,
		names: list,
		file_name: str
	):
    """
    Plot an overlap matrix
    """

    N_labels = len(names)

    plt.figure(figsize=(12,10))

    plt.imshow(matrix)

    plt.xticks(np.arange(N_labels), names, rotation=90)
    plt.yticks(np.arange(N_labels), names)

    plt.colorbar()

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.clf()

def plot_corner(
	samples: np.array,
	names: list,
	file_name: str,
):
	if len(names) + 2 == samples.shape[1]:
		names.extend(['beta', 'gamma'])
		
	corner.corner(
        samples,
        labels = names,
    )
	
	plt.savefig(file_name)
	plt.clf()
    
 #    def pm_hist(self, outfile):
 #        """
 #        Plot a histogram of the proper motions of the quasars 
 #        """
 #        proper_motions_Cartesian = np.linalg.norm(CT.geographic_to_Cartesian_vector(self.positions, self.proper_motions), axis = 1)
 #        plt.hist(proper_motions_Cartesian)
            
 #        plt.xlabel('Proper motion [mas/yr]')
 #        plt.ylabel('Number of quasars')
 #        plt.title('Histogram of quasar proper motions')
 #        plt.yscale('log')

 #        plt.tight_layout()
 #        plt.savefig(outfile)
 #        plt.clf()

       