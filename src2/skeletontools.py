# Import
import numpy as np

import kimimaro
import cloudvolume

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def draw_skeleton(skel, nodes_selected):
    # Custom function to draw skeleton. Adapted from original Skeleton.viewer() in cloudvolume package.

    if skel is None:
        print("No skeleton found.")
        return()
    
    # Set skeleton drawing parameters
    draw_vertices=True
    draw_edges=True
    units='nm'
    color_by='radius'
    #color_by='radius', # aka 'r' or 'components' aka 'c'
    #color_by='components', # aka 'r' or 'components' aka 'c'

    # Draw skeleton
    ################################################################################################

    RADII_KEYWORDS = ('radius', 'radii', 'r')
    COMPONENT_KEYWORDS = ('component', 'components', 'c')

    #newTk = Tk()
    fig = plt.figure(figsize=(6,6))
    #canvas = FigureCanvasTkAgg(fig, newTk)
    ax = Axes3D(fig)
    ax.view_init(elev=0, azim=180) # Set initial viewing orientation
    ax.set_xlabel(units)
    ax.set_ylabel(units)
    ax.set_zlabel(units)

    # Set plot axes equal. Matplotlib doesn't have an easier way to
    # do this for 3d plots.
    X = skel.vertices[:,0]
    Y = skel.vertices[:,1]
    Z = skel.vertices[:,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ### END EQUALIZATION CODE ###

    component_colors = ['k', 'deeppink', 'dodgerblue', 'mediumaquamarine', 'gold' ]

    def draw_component(i, skel):
        component_color = component_colors[ i % len(component_colors) ]

        if draw_vertices:
            xs = skel.vertices[:,0]
            ys = skel.vertices[:,1]
            zs = skel.vertices[:,2]

            if color_by in RADII_KEYWORDS:
                colmap = cm.ScalarMappable(cmap=cm.get_cmap('rainbow'))
                colmap.set_array(skel.radii)

                normed_radii = skel.radii / np.max(skel.radii)
                yg = ax.scatter(xs, ys, zs, c=cm.rainbow(normed_radii), marker='o', picker=True)
                cbar = fig.colorbar(colmap)
                cbar.set_label('radius (' + units + ')', rotation=270)

                # Draw selected nodes in a different color and size
                for n in nodes_selected.keys():
                    x,y,z = skel.vertices[n]
                    ax.scatter(x, y, z, color='k', marker='o', s=70)
                    #print(n,':',x,y,z)

            elif color_by in COMPONENT_KEYWORDS:
                yg = ax.scatter(xs, ys, zs, color=component_color, marker='.', picker=True)
            else:
                yg = ax.scatter(xs, ys, zs, color='k', marker='.', picker=True)

        if draw_edges:
            for e1, e2 in skel.edges:
                pt1, pt2 = skel.vertices[e1], skel.vertices[e2]
                ax.plot(	
                    [ pt1[0], pt2[0] ],
                    [ pt1[1], pt2[1] ],
                    zs=[ pt1[2], pt2[2] ],
                    color=(component_color if not draw_vertices else 'silver'),
                    linewidth=1,
                )

    if color_by in COMPONENT_KEYWORDS:
        for i, skel in enumerate(skel.components()):
            draw_component(i, skel)
    else:
        draw_component(0, skel)

    #END Draw skeleton
    ################################################################################################
    
    
def skeletonize(img, scale, const, anisotropy):
    skels = kimimaro.skeletonize(img, teasar_params={
          'scale': scale,
          'const': const,  # physical units
          'pdrf_exponent': 4,
          'pdrf_scale': 10000,
          'soma_detection_threshold': 1100,  # physical units
          'soma_acceptance_threshold': 3500,  # physical units
          'soma_invalidation_scale': 1.0,
          'soma_invalidation_const': 300,  # physical units
          'max_paths': 300, }, dust_threshold=100,
          anisotropy=anisotropy,
          fix_branching=True,
          fix_borders=True,
          fill_holes=False,
          fix_avocados=False,
          progress=True,
          parallel=1,
          parallel_chunk_size=100)
    #print(skels)
    return(skels)
