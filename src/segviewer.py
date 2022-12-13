# Import modules
import os
import re
import cv2
import random
import copy
import numpy as np
import stackview
from skimage import measure

from tkinter import Tk, filedialog
from matplotlib import pyplot as plt

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display, clear_output

import skeletontools

class ImageLoader():
    def __init__(self):
        self.stack = []
        self.stack_resized = []
        self.stack_rgb_resized = []
        self.filenames = []
        #print("ImageLoader created")
        
    # General utility functions
    def sorted_nicely(self, l): 
        """ Sort the given iterable in the way that humans expect."""
        """Taken from: http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python"""
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)
    ###END FUNCTION sorted_nicely(l)

    def load_image_sequence(self, dirname):
        self.dirname = dirname
        if not os.path.isdir(self.dirname):
        #if self.dirname == "":
            print("ERROR: No valid directory.")
            return(None, None, None)

        #print ("Loading data...")
        self.filenames = []
        for fn in self.sorted_nicely(os.listdir(self.dirname)):
            if fn.endswith(".png"):
                self.filenames.append(os.path.join(self.dirname, fn))
            #files=sorted(filenames)

        self.stack = []
        self.stack_resized = []
        self.stack_rgb_resized = []

        for k, fn in enumerate(self.filenames):
            #print(k,fn)
            image = cv2.imread(fn)
            #print("###",image.shape)
            #print(np.unique(image))
            #print(np.sum(image[0]-image[1]), np.sum(image[1]-image[2]))
            if len(image.shape) == 3:
                image = image[:,:,0] #Assuming that images are gray scale, i.e. all channels are identical.
            #print(">>>",image.shape, np.array([image]*3).transpose((1,-1,0)).shape)

            h = image.shape[0]
            w = image.shape[1]
            image_dim_threshold = 500
            if (h >= image_dim_threshold or w >= image_dim_threshold) or (h < image_dim_threshold and w < image_dim_threshold):
                if h > w:
                    image_height = image_dim_threshold
                    image_width = int((w*image_dim_threshold)/h)
                else:
                    image_width = image_dim_threshold
                    image_height = int((h*image_dim_threshold)/w)
                #print("Image",k,"is too large, (w,h) = ("+str(w)+","+str(h)+"). Resizing to "+str((image_width,image_height))+".")
                image_resized = cv2.resize(image,(image_width,image_height), interpolation = cv2.INTER_NEAREST) # resize without generating intermediate colors/values during interpolation
            else:
                image_width = w
                image_height = h

            ###???CAUTION: Use image_resized in this step for testing skeletonization only. DO NOT use image_resized for final version.
            #stack.append(image[:,:,0]) # Assuming that images are gray scale. The variable threeDmatrix will be used for skeletonization. Note that images with original resolution are stored here. Useful for quantitatie analysis.
            self.stack.append(image) # Assuming that images are gray scale. The variable threeDmatrix will be used for skeletonization. Note that images with original resolution are stored here. Useful for quantitatie analysis.
            self.stack_resized.append(image_resized) # Comment out this line in final version. Use the above line instead.

            #stack_rgb.append(image_resized) # The variable 'stack_rgb' stores images in RGB format. Used in visualization of images. Note that resized images are stored here. Quantitative analysis with these images might not give correct results.
            self.stack_rgb_resized.append(np.array([image_resized]*3).transpose((1,-1,0))) # The variable 'stack_rgb' stores images in RGB format. Used in visualization of images. Note that resized images are stored here. Quantitative analysis with these images might not give correct results.

        #print("Original image shape:", image.shape)
        #print("Resized image shape:", image_resized.shape)
        #print(">>>",image.shape,self.image_width,self.image_height)

        #Create a original copy of self.matrices. Useful for referring to original segments easily.
        #matrices_original = np.copy(matrices)

        self.stack = np.dstack(self.stack).transpose(2,0,1)
        self.stack_resized = np.dstack(self.stack_resized).transpose(2,0,1)
        self.stack_rgb_resized = np.array(self.stack_rgb_resized)

        #print("Done")
        
        #print(">>>", np.unique(self.stack), np.unique(self.stack_resized), np.unique(self.stack_rgb_resized))

        return(self.stack, self.stack_resized, self.stack_rgb_resized)



class Seg_viewer():
    def __init__(self):#, em_image=None, label_image=None, rgb_image=None, colormapHex=None):

        """
        Segmentation Viewer: View segmented 3D stack and, optionally, corresponding raw image stack
        """

        # Image loader
        self.stack_em, self.stack_em_resized, self.stack_em_rgb_resized = None, None, None
        self.stack_seg, self.stack_seg_resized, self.stack_seg_rgb_resized = None, None, None

        #self.seg_viewer = None
        
        self.skel = None
    
        # Create framework
        self.create_data_loader()
        self.create_seg_viewer()
        self.create_viewer()
        
    def create_viewer(self):
        self.viewer = widgets.VBox([self.data_loader, self.seg_viewer])

    def create_data_loader(self):
        # Display widgets for loading image data 
        #global dirname_em_widget, dirname_seg_widget
        #global em_label_widget, seg_label_widget

        # sample_data/C14_mip3_crop_image_exports/EM
        # sample_data/C14_mip3_crop_image_exports/segmentation
        style = {'description_width': '250px'}#'initial'
        layout = {'width': '750px'}
        self.dirname_em_widget = widgets.Text(value='', description="Non-segmented images folder:", disabled=False, style=style, layout=layout)
        self.dirname_seg_widget = widgets.Text(value='', description="Segmented images folder:", disabled=False, style=style, layout=layout)
        self.load_button_widget = widgets.Button(description='Click to load images', disabled=False, button_style='', tooltip='Load images')

        self.em_label_widget = widgets.Label(value="EM images: Not loaded")
        self.seg_label_widget = widgets.Label(value="Segmentation images: Not loaded")

        self.dirname_em_widget.value = "sample_data/C14_mip3_crop_image_exports/EM"
        self.dirname_seg_widget.value = "sample_data/C14_mip3_crop_image_exports/segmentation"

        self.dirname_em_browse_widget = widgets.Button(layout={'width':'35px'}, style={'button_color':'white'}, icon="fa-folder-open", tooltip='Choose folder')
        self.dirname_seg_browse_widget = widgets.Button(layout={'width':'35px'}, style={'button_color':'white'}, icon="fa-folder-open", tooltip='Choose folder')
        
        #display(widgets.VBox([widgets.HBox([widgets.Label(value="EM folder path:"), dirname_em_widget]),
        #                      widgets.HBox([widgets.Label(value="Segmentation folder path:"), dirname_seg_widget]),
        #                      load_button_widget]))

        self.data_loader = widgets.VBox([widgets.HBox([self.dirname_em_widget,self.dirname_em_browse_widget]),
                                         widgets.HBox([self.dirname_seg_widget,self.dirname_seg_browse_widget]),
                                         self.load_button_widget,
                                         self.em_label_widget,
                                         self.seg_label_widget
                                        ])

        self.dirname_em_browse_widget.on_click(self.set_em_dir)
        self.dirname_seg_browse_widget.on_click(self.set_seg_dir)
        self.load_button_widget.on_click(self.image_loader)

        # TO DO:
        # Check the sizes and show warning in case of mismatch between EM and seg sizes.

    def set_em_dir(self, _):
        self.dirname_em_widget.value = self.browse_dir(title="Choose folder containing original image sequence")

    def set_seg_dir(self, _):
        self.dirname_seg_widget.value = self.browse_dir(title="Choose folder containing segmented image sequence")

    def browse_dir(self, title="Choose Directory"):
        #clear_output()                                         # Button is deleted after it is clicked.
        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        #files = filedialog.askopenfilename(multiple=True)    # List of selected files will be set button's file attribute.
        dirname = filedialog.askdirectory(title=title)
        return dirname

    def browse_file(self, title="Choose a file name"):
        #clear_output()                                         # Button is deleted after it is clicked.
        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        files = filedialog.askopenfilename(multiple=True)    # List of selected files will be set button's file attribute.
        #dirname = filedialog.askdirectory(title=title)
        return dirname

    
    def create_seg_viewer(self, as_placeholder=True):
        if as_placeholder:
            self.seg_viewer = widgets.HBox([widgets.Label(value="Viewer: No data to show.")])
            return()
        
        self.slicers = []
        if self.stack_em_resized is not None:
            self.slicers.append(stackview.slice(self.stack_em_resized, axis=0, continuous_update=True))
        if self.stack_seg_resized is not None and self.stack_seg_resized is not None:
            self.slicers.append(stackview.slice(self.stack_seg_rgb_resized, axis=0, continuous_update=True))

        
        style = {'description_width': '40px'}#'initial'
        layout = {'width': '150px'}
        #self.seg_list = widgets.VBox([widgets.ColorPicker(concise=False,
        #                                                  description='Seg '+str(k),
        #                                                  value=v,
        #                                                  style=style,
        #                                                  layout=layout,
        #                                                  disabled=False) for k,v in sorted(self.colormapHex.items())])

        self.seg_list = widgets.VBox([widgets.HBox([widgets.ColorPicker(concise=False,
                                                            description=str(k),
                                                            value=v,
                                                            style=style,
                                                            layout=layout,
                                                            disabled=False),
                                                    widgets.Checkbox(value=False,
                                                            description="",
                                                            style={'description_width': '0px'},
                                                            layout={'width': '30px'},
                                                            indent=False)
                                                   ]) for k,v in sorted(self.colormapHex.items())
                                     ])

        
        
        self.recolor_button = widgets.Button(description="Recolor segments", layout={'width': '150px'}, tooltip="Recolor segments")
        self.skel_button = widgets.Button(description="Skeletonize", layout={'width': '150px'}, tooltip="Skeletonize selected segments")
        self.exportobj_button = widgets.Button(description="Export OBJ", layout={'width': '150px'}, tooltip="Export OBJ file for selected segments")
        #Function for "Find protrusions" is not yet implemented here.
        #self.protr_button = widgets.Button(description="Find protrusions", layout={'width': '150px'}, tooltip="Identify protrusions of selected segments")
        
        #self.labels_picker = widgets.VBox([self.seg_list] + [self.recolor_button])

        self.labels_picker = widgets.VBox([self.seg_list] +
                                          [widgets.VBox([self.recolor_button,
                                                         self.skel_button,
                                                         self.exportobj_button
                                                         #self.protr_button
                                                        ])]
                                         )

        self.seg_viewer = widgets.HBox([x for x in self.slicers] + [self.labels_picker])

        self.recolor_button.on_click(self.recolor_segments())
        self.skel_button.on_click(self.skeletonize_selected_labels())
        self.exportobj_button.on_click(self.exportobj_selected_labels())
        #self.protr_button.on_click(self.identify_protrusions()) # Not yet defined.

    def rgb_hex_to_dec(self, hexcolor):
        hexcolor = hexcolor.lstrip("#")
        return([int(hexcolor[i:i+2], 16) for i in [0,2,4]])

    def recolor_segments(self):
        def recolor_segments(b):
            
            for hbox in self.seg_viewer.children[2-self.em_image_absent].children[0].children:
                cp, lp = hbox.children
                hexcolor = cp.value
                deccolor = self.rgb_hex_to_dec(hexcolor)
                label = int(cp.description.replace("Seg ",""))
                print("Recoloring", cp.description, "to", hexcolor, deccolor, label)
                
                self.colormapDec[label] = deccolor
                self.colormapHex[label] = hexcolor

            self.stack_seg_rgb_resized = colorize_seg(colormap=self.colormapDec,
                                                      label_image=self.stack_seg_resized,
                                                      rgb_image=self.stack_seg_rgb_resized)
            
            clear_output()
            self.create_seg_viewer(as_placeholder=False)
            self.create_viewer()
            display(self.viewer)

        return(recolor_segments)

    def get_obj(self, obj_name='default_obj', verts=None, edges=None, faces=None, v_num_offset=0, material_class='default'):
        # Function to create obj format string which can be written to an obj file.
        obj = "o " + str(obj_name) + "\n"

        for item in verts:
            obj += "v {0} {1} {2}\n".format('{:.6f}'.format(item[0]),
                                            '{:.6f}'.format(item[1]),
                                            '{:.6f}'.format(item[2]))

        if material_class.strip() == '':
            material_class = 'default'
        obj += "usemtl " + material_class + "\n"
        
        for item in faces:
            obj += "f " + " ".join([str(x+v_num_offset) for x in item]) + "\n"
        v_num_offset += len(verts)

        return(obj, v_num_offset)

    def exportobj_selected_labels(self):
        def exportobj_selected_labels(b):
                        
            self.dirname_output = self.browse_dir(title="Choose output folder")
            label_list = []
            img = copy.copy(self.stack_seg)
            selected_label_count = 0
            for container in self.seg_viewer.children[2-self.em_image_absent].children[0].children:
                cp, lp = container.children # Get colorpicker (cp) and labelpicker (lp)
                label = int(cp.description.replace("Seg ",""))
                
                if lp.value:
                    #print("*", label, lp.value) # Uncomment to see selected segment label.
                    img[img == label] = 255
                    label_list.append(label)
                    selected_label_count += 1
                    #print("*", np.unique(img)) # Uncomment to see unique segment labels in the image.
            if selected_label_count == 0:
                print("Please select at least one segment label before performing segmentation.")
                return(None)

            voxel_size_nm = (32,32,30) # This is same as the voxel size in nanometers (give in x,y,z or h,w,z order)
            scale_nm_to_um = 0.001
            voxel_size_um = tuple(np.array(voxel_size_nm)*scale_nm_to_um)
            
            # Create surface using marching cubes algorithm. Default method is Lewiner.
            verts, faces, normals, values = measure.marching_cubes(img, step_size=1, spacing=voxel_size_um)
            
            obj_name = "_".join([str(x) for x in label_list])
            # Get obj formatted string
            obj, v_num_offset = self.get_obj(obj_name="Seg_"+obj_name, verts=verts, faces=faces, v_num_offset=1, material_class='default')
            
            # Write obj file
            objfilename = obj_name + ".obj"
            with open(self.dirname_output + "/" + objfilename,'w') as o:
                o.write(obj)
        
        return(exportobj_selected_labels)
            
    def skeletonize_selected_labels(self):
        def skeletonize_selected_labels(b):
            
            img = copy.copy(self.stack_seg)
            selected_label_count = 0
            for container in self.seg_viewer.children[2-self.em_image_absent].children[0].children:
                cp, lp = container.children # Get colorpicker (cp) and labelpicker (lp)
                label = int(cp.description.replace("Seg ",""))
                if lp.value:
                    #print("*", label, lp.value) # Uncomment to see selected segment label.
                    img[img == label] = 255
                    selected_label_count += 1
                    #print("*", np.unique(img)) # Uncomment to see unique segment labels in the image.
            if selected_label_count == 0:
                print("Please select at least one segment label before performing segmentation.")
                return(None)
            #print(np.unique(img)) # Uncomment this line to see the unique segment labels in the image.
            # Kimimaro skeletonization
            voxel_size_nm = (32,32,30) # This is same as the voxel size in nanometers (give in x,y,z or h,w,z order)
            print("Assuming voxel size in nanometers:", voxel_size_nm)
            print("Please see function skeletonize_selected_labels() in the segviewer.py file to define custom voxel size.") 
            self.skel = skeletontools.skeletonize(img, 1, 1000, voxel_size_nm)
        return(skeletonize_selected_labels)

    def get_skel(self, label=255, downsample=0):
        if self.skel is not None:
            return(self.skel[label].downsample(downsample))
        return(None)
    
    #def identify_protrusions(self):
    #    def identify_protrusions(b):
    #        print("Function- identify_protrusions: Not yet defined.", b)
    #    return(identify_protrusions)
        
    def recolor_segments_old(self):
        def recolor_segments(b):
            
            for cp in self.seg_viewer.children[2-self.em_image_absent].children[0].children:
                hexcolor = cp.value
                deccolor = self.rgb_hex_to_dec(hexcolor)
                label = int(cp.description.replace("Seg ",""))
                print("Recoloring", cp.description, "to", hexcolor, deccolor, label)
                
                self.colormapDec[label] = deccolor
                self.colormapHex[label] = hexcolor

            self.stack_seg_rgb_resized = colorize_seg(colormap=self.colormapDec,
                                                      label_image=self.stack_seg_resized,
                                                      rgb_image=self.stack_seg_rgb_resized)
            
            clear_output()
            self.create_seg_viewer(as_placeholder=False)
            self.create_viewer()
            display(self.viewer)

        return(recolor_segments)

    def image_loader(self,_):

        self.reset()

        loader = ImageLoader()

        if self.dirname_em_widget.value.strip() != '':
            self.stack_em, self.stack_em_resized, self.stack_em_rgb_resized = \
            loader.load_image_sequence(self.dirname_em_widget.value)
            #print("Loaded EM image stack of shape", stack_em.shape)

        if self.dirname_seg_widget.value.strip() != '':
            self.stack_seg, self.stack_seg_resized, self.stack_seg_rgb_resized = \
            loader.load_image_sequence(self.dirname_seg_widget.value)
            #print("Loaded segmentation image stack of shape", stack_seg.shape)

            # If there is a label with value 255, then change it to some other unused value between 1 and 254
            for mat in [self.stack_seg, self.stack_seg_resized, self.stack_seg_rgb_resized]:
                unique_labels = np.unique(mat)
                if 255 in unique_labels:
                    while True:
                        l = random.randint(1,254)
                        if l not in unique_labels:
                            print("New value for label 255:", l)
                            break
                    mat[mat == 255] = l
        else:
            print("Please provide at least one segmented image.")
            return()

        if self.stack_em is not None:
            #print("EM stack shapes:", stack_em.shape, stack_em_resized.shape, stack_em_rgb_resized.shape)
            self.em_label_widget.value = "EM images: Shape = " + str(self.stack_em.shape) + \
                                         " resized to " + str(self.stack_em_resized.shape)
        else:
            #print("No EM data available.")
            self.em_label_widget.value = "EM images: Not loaded"
            pass

        #print("Segmentation stack shapes:", stack_seg.shape, stack_seg_resized.shape, stack_seg_rgb_resized.shape)
        self.seg_label_widget.value = "Segmentation images: Shape = " + str(self.stack_seg.shape) + \
                                      " resized to " + str(self.stack_seg_resized.shape)


        self.em_image_absent = self.stack_em_resized is None
        
        # Assign random colors
        self.colormapDec, self.colormapHex = colormap_random(label_image=self.stack_seg_resized)

        self.stack_seg_rgb_resized = colorize_seg(colormap=self.colormapDec,
                                                  label_image=self.stack_seg_resized,
                                                  rgb_image=self.stack_seg_rgb_resized)

        # Create segmentation viewer using loaded images
        clear_output()
        self.create_seg_viewer(as_placeholder=False)
        self.create_viewer()
        display(self.viewer)


    def reset(self):
        self.stack_em, self.stack_em_resized, self.stack_em_rgb_resized = None, None, None
        self.stack_seg, self.stack_seg_resized, self.stack_seg_rgb_resized = None, None, None
        self.em_image_absent = self.stack_em_resized is None
        self.colormapDec, self.colormapHex = None, None
        
######################################################################################

# Common function useful for colorizing segmented image stacks

def colormap_random(label_image=None):
    uniqe_labels = np.unique(label_image)#_rgb_resized)
    #print(uniqe_labels)
    
    ### Assign random colors to the segments
    colormap = {l:[random.randint(16,255),random.randint(16,255),random.randint(16,255)] for l in uniqe_labels[1:]}
    colormapHex = {}
    for label in colormap:
        #print(label, colormap[label], )
        colormapHex[label] = "#"+''.join([hex(x)[2:] for x in colormap[label]])
    #print(colormap,colormapHex)
    return(colormap, colormapHex)

def colorize_seg(colormap=None, label_image=None, rgb_image=None):
    uniqe_labels = np.unique(label_image)
    rgb_image = copy.copy(rgb_image)
    for i in range(0,label_image.shape[0]):
        #print(im.shape)
        for ch in [0,1,2]:
            #print(im[:,:,ch].shape)
            for l in uniqe_labels[1:]:
                rgb_image[i][:,:,ch][label_image[i] == l] = colormap[l][ch]
    return(rgb_image)

# Testing
#print("Recoloring segments...")
#colormapDec, colormapHex = colormap_random(label_image=stack_seg_resized)

#stack_seg_rgb_resized = colorize_seg(colormap=colormap,
#                                     label_image=stack_seg_resized,
#                                     rgb_image=stack_seg_rgb_resized)

######################################################################################

