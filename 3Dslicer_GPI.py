# Author: Gabe Shepherd
# Date: 2020-06

import gpi
from gpi import QtGui, QtWidgets
import numpy as np

# WIDGET
class WindowLevel(gpi.GenericWidgetGroup):
    """Provides an interface to the BasicCWFCSliders."""
    valueChanged = gpi.Signal()

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.sl = gpi.BasicCWFCSliders()
        self.sl.valueChanged.connect(self.valueChanged)
        self.pb = gpi.BasicPushButton()
        self.pb.set_button_title('reset')
        # layout
        wdgLayout = QtWidgets.QVBoxLayout()
        wdgLayout.addWidget(self.sl)
        wdgLayout.addWidget(self.pb)
        self.setLayout(wdgLayout)
        # default
        self.set_min(0)
        self.set_max(100)
        self.sl.set_allvisible(True)
        self.reset_sliders()
        self.pb.valueChanged.connect(self.reset_sliders)

    # setters
    def set_val(self, val):
        """Set multiple values with a python-dict with keys:
        level, window, floor and ceiling. -Requires integer
        values for each key.
        """
        self.sl.set_center(val['level'])
        self.sl.set_width(val['window'])
        self.sl.set_floor(val['floor'])
        self.sl.set_ceiling(val['ceiling'])

    def set_min(self, val):
        """Set min for level, window, floor and ceiling (int)."""
        self.sl.set_min(val)

    def set_max(self, val):
        """Set max for level, window, floor and ceiling (int)."""
        self.sl.set_max(val)

    # getters
    def get_val(self):
        val = {}
        val['level'] = self.sl.get_center()
        val['window'] = self.sl.get_width()
        val['floor'] = self.sl.get_floor()
        val['ceiling'] = self.sl.get_ceiling()
        return val

    def get_min(self):
        return self.sl.get_min()

    def get_max(self):
        return self.sl.get_max()

    def reset_sliders(self):
        val = {}
        val['window'] = 100
        val['level'] = 50
        val['floor'] = 0
        val['ceiling'] = 100
        self.set_val(val)

class ExternalNode(gpi.NodeAPI):
    """ Display Sagittal, Coronal, and Transverse Slices of 3D data.
    INPUT:
        in - 3D image data (RGB is 4 --> currently unsupported)
    OUTPUT:
        out - combined image of all three slices
        sagittal - only this slice
        coronal - only this slice
        transverse - only this slice
    """

    def execType(self):
        return gpi.GPI_APPLOOP

    def initUI(self):
        # Widgets
        self.addWidget('ExclusivePushButtons', 'Complex Display',
                       buttons=['R', 'I', 'M', 'P', 'C'], val=4)
        self.real_cmaps = ['Gray', 'IceFire', 'Fire', 'Hot', 'HOT2', 'BGR']
        self.complex_cmaps = ['HSV', 'HSL', 'HUSL', 'CoolWarm']
        self.addWidget('ExclusivePushButtons', 'Color Map',
                       buttons=self.real_cmaps, val=0, collapsed=True)
        self.addWidget('SpinBox', 'Edge Pixels', min=0)
        self.addWidget('SpinBox', 'Black Pixels', min=0)
        self.addWidget('DisplayBox', 'Viewport:')
        self.addWidget('Slider', 'Axial Slice (Blue)', val=20, min=0, max=40)  # only initial values
        self.addWidget('Slider', 'Coronal Slice (Green)', val=20, min=0, max=40)
        self.addWidget('Slider', 'Sagittal Slice (Red)', val=20, min=0, max=40)
        self.addWidget('ExclusivePushButtons', 'Extra Dimension', buttons=['Slice', 'Tile', 'RGB(A)'], val=0, collapsed=True)
        self.addWidget('SpinBox', '# Columns', val=1, collapsed=True)
        self.addWidget('SpinBox', '# Rows', val=1, collapsed=True)
        self.addWidget('WindowLevel', 'L W F C:', collapsed=True)
        self.addWidget('ExclusivePushButtons', 'Scalar Display',
                       buttons=['Pass', 'Mag', 'Sign'], val=0, collapsed=True)
        self.addWidget('DoubleSpinBox', 'Gamma', min=0.1, max=10, val=1, singlestep=0.05, decimals=3, collapsed=True)
        self.addWidget('ExclusivePushButtons', 'Zero Ref',
                       buttons=['---', '0->', '-0-', '<-0'], val=0, collapsed=True)
        self.addWidget('PushButton', 'Fix Range', button_title='Auto-Range On', toggle=True, collapsed=True)
        self.addWidget('DoubleSpinBox', 'Range Min', collapsed=True)
        self.addWidget('DoubleSpinBox', 'Range Max', collapsed=True)
        self.addWidget('Slider', 'Slice', min=1, val=1, collapsed=True)
        self.addWidget('ExclusivePushButtons', 'Slice/Tile Dimension', buttons=['0', '1', '2'], val=0, collapsed=True)

        # IO Ports
        self.addInPort('in', 'NPYarray', drange=(2, 3))
        self.addOutPort('out data', 'NPYarray')
        self.addOutPort('sagittal slice', 'NPYarray')
        self.addOutPort('coronal slice', 'NPYarray')
        self.addOutPort('transverse slice', 'NPYarray')

    def validate(self):

        # Complex or Scalar?
        data = self.getData('in')
        # dimfunc = self.getVal('Extra Dimension')
        dimfunc = 2
        self.setAttr('Extra Dimension', buttons=['Slice', 'Tile', 'RGB(A)'], val=dimfunc)
        self.setAttr('Slice/Tile Dimension', visible=False)
        self.setAttr('Slice', visible=False)
        self.setAttr('# Rows', visible=False)
        self.setAttr('# Columns', visible=False)

        if dimfunc == 2: # RGBA
          self.setAttr('Complex Display',visible=False)
          self.setAttr('Color Map',visible=False)
          self.setAttr('Scalar Display',visible=False)
          self.setAttr('Edge Pixels',visible=False)
          self.setAttr('Black Pixels',visible=False)
          self.setAttr('Zero Ref',visible=False)
          self.setAttr('Range Min',visible=False)
          self.setAttr('Range Max',visible=False)

        else:

          if np.iscomplexobj(data):
            self.setAttr('Complex Display',visible=True)
            scalarvis = self.getVal('Complex Display') != 4
          else:
            self.setAttr('Complex Display',visible=False)
            scalarvis = True

          if scalarvis:
            self.setAttr('Color Map',buttons=self.real_cmaps,
                         collapsed=self.getAttr('Color Map', 'collapsed'))
          else:
            self.setAttr('Color Map',buttons=self.complex_cmaps,
                         collapsed=self.getAttr('Color Map', 'collapsed'))

          self.setAttr('Scalar Display',visible=scalarvis)
          self.setAttr('Edge Pixels',visible=not scalarvis)
          self.setAttr('Black Pixels',visible=not scalarvis)

          if self.getVal('Scalar Display') == 2:
            self.setAttr('Zero Ref',visible=False)
          else:
            self.setAttr('Zero Ref',visible=scalarvis)

          self.setAttr('Range Min',visible=scalarvis)
          self.setAttr('Range Max',visible=scalarvis)

          zval = self.getVal('Zero Ref')
          if zval == 1:
            self.setAttr('Range Min',val=0)
          elif zval == 3:
            self.setAttr('Range Max',val=0)

          if self.getVal('Fix Range'):
            self.setAttr('Fix Range',button_title="Fixed Range On")
          else:
            self.setAttr('Fix Range',button_title="Auto-Range On")

        return 0

    def compute(self):
        data3d = self.getData('in').astype(np.float64)
        dim = list(np.shape(data3d))
        # reset default values with the values from the dimensions of the input data
        self.setAttr('Axial Slice (Blue)', max=dim[0])
        self.setAttr('Coronal Slice (Green)', max=dim[1])
        self.setAttr('Sagittal Slice (Red)', max=dim[2])
        xslice = self.getVal('Axial Slice (Blue)')
        yslice = self.getVal('Coronal Slice (Green)')
        zslice = self.getVal('Sagittal Slice (Red)')
        sagittalSlice = data3d[xslice, :, :] 
        coronalSlice = data3d[:, yslice, :]      
        transverseSlice = data3d[:, :, zslice]     

        # find and define max value for the visible data
        maximum = np.max([np.max(sagittalSlice), np.max(coronalSlice), np.max(transverseSlice)])
        divisor = maximum/255.0

        # normalize data (to display floats, the data ranges from 0 to 255)
        sagittalSlice = sagittalSlice/divisor
        coronalSlice = coronalSlice/divisor
        transverseSlice = transverseSlice/divisor

        # add borders and crosshairs
        # make rgb data
        sagittalSlice = np.dstack([sagittalSlice, sagittalSlice, sagittalSlice, sagittalSlice])         
        coronalSlice = np.dstack([coronalSlice, coronalSlice, coronalSlice, coronalSlice])              
        transverseSlice = np.dstack([transverseSlice, transverseSlice, transverseSlice, transverseSlice])   

        # create R border for sagittal data
        dim_r = list(np.shape(sagittalSlice))
        r_buffer = np.zeros([dim_r[0]+4, dim_r[1]+4, 4], dtype=np.float64)           
        new_dim = list(np.shape(r_buffer))
        red_dim = new_dim
        for x in range(new_dim[0]):
            for y in range(new_dim[1]):
                if x > 1 and y > 1 and x < (new_dim[0]-2) and y < (new_dim[1]-2):
                    r_buffer[x][y] = sagittalSlice[x-2, y-2, :]
                else: r_buffer[x][y][0] = 255
                if x == yslice and ((y > 1 and y < (zslice-5)) or (y > (zslice+5) and y < (new_dim[1]-2))):
                    r_buffer[x][y][1] = 255
                if y == zslice and ((x > 1 and x < (yslice-5)) or (x > (yslice+5) and x < (new_dim[0]-2))):
                    r_buffer[x][y][2] = 255
        
        # create G border for coronal data
        dim_g = list(np.shape(coronalSlice))
        g_buffer = np.zeros([dim_g[0]+4, dim_g[1]+4, 4], dtype=np.float64)               
        new_dim = list(np.shape(g_buffer))
        green_dim = new_dim
        for x in range(new_dim[0]):
            for y in range(new_dim[1]):
                if x > 1 and y > 1 and x < (new_dim[0]-2) and y < (new_dim[1]-2):
                    g_buffer[x][y] = coronalSlice[x-2, y-2, :]
                else: g_buffer[x][y][1] = 255
                if x == xslice and ((y > 1 and y < (zslice-5)) or (y > (zslice+5) and y < (new_dim[1]-2))):
                    g_buffer[x][y][0] = 255
                if y == zslice and ((x > 1 and x < (xslice-5)) or (x > (xslice+5) and x < (new_dim[0]-2))):
                    g_buffer[x][y][2] = 255
        
        # create B border for axial data
        dim_b = list(np.shape(transverseSlice))
        b_buffer = np.zeros([dim_b[0]+4, dim_b[1]+4, 4], dtype=np.float64)             
        new_dim = list(np.shape(b_buffer))
        blue_dim = new_dim
        for x in range(new_dim[0]):
            for y in range(new_dim[1]):
                if x > 1 and y > 1 and x < (new_dim[0]-2) and y < (new_dim[1]-2):
                    b_buffer[x][y] = transverseSlice[x-2, y-2, :]
                else: b_buffer[x][y][2] = 255
                if x == xslice and ((y > 1 and y < (yslice-5)) or (y > (yslice+5) and y < (new_dim[1]-2))):
                    b_buffer[x][y][0] = 255
                if y == yslice and ((x > 1 and x < (xslice-5)) or (x > (xslice+5) and x < (new_dim[0]-2))):
                    b_buffer[x][y][1] = 255
        
        # combine into one array
        combine1 = np.append(g_buffer, b_buffer, axis=1)     # combine coronal and transverse slices
        pad = np.zeros([dim[1]+4, dim[1]+4, 4], dtype=np.float64)              # pad sagittal slice with zeros
        combine2 = np.append(pad, r_buffer, axis=1)
        combinedData = np.append(combine1, combine2, axis=0)

        # from ImageDisplay (RGBA case only)
        from matplotlib import cm

        # make a copy for changes
        data = combinedData.astype(np.uint8)

        # get extra dimension parameters and modify data
        dimfunc = self.getVal('Extra Dimension')
        dimval = self.getVal('Slice/Tile Dimension')
        if data.ndim == 3 and dimfunc < 2:
            if dimfunc == 0: # slice data
                slval = self.getVal('Slice')-1
                if dimval == 0:
                    data = data[slval,...]
                elif dimval == 1:
                    data = data[:,slval,:]
                else:
                    data = data[...,slval]
            else: # tile data
                ncol = self.getVal('# Columns')
                nrow = self.getVal('# Rows')

                # add some blank tiles
                data = np.rollaxis(data, dimval)
                N, xres, yres = data.shape
                N_new = ncol * nrow
                pad_vals = ((0, N_new - N), (0, 0), (0, 0))
                data = np.pad(data, pad_vals, mode='constant')

                # from http://stackoverflow.com/a/13990648/333308
                data = np.reshape(data, (nrow, ncol, xres, yres))
                data = np.swapaxes(data, 1, 2)
                data = np.reshape(data, (nrow*xres, ncol*yres))


        # Read in parameters, make a little floor:ceiling adjustment
        gamma = self.getVal('Gamma')
        lval = self.getAttr('L W F C:', 'val')
        cval = self.getVal('Complex Display')

        if 'Complex Display' in self.widgetEvents():
          if cval == 4:
            self.setAttr('Color Map', buttons=self.complex_cmaps,
                         collapsed=self.getAttr('Color Map', 'collapsed'),
                         val=0)
          else:
            self.setAttr('Color Map', buttons=self.real_cmaps,
                         collapsed=self.getAttr('Color Map', 'collapsed'),
                         val=0)

        cmap = self.getVal('Color Map')
        sval = self.getVal('Scalar Display')
        zval = self.getVal('Zero Ref')
        fval = self.getVal('Fix Range')
        rmin = self.getVal('Range Min')
        rmax = self.getVal('Range Max')

        flor = 0.01*lval['floor']
        ceil = 0.01*lval['ceiling']
        if ceil == flor:
          if ceil == 1.:
            flor = 0.999
          else:
            ceil += 0.001

        # DISPLAY RGB image
        if data.shape[-1] > 3:
          red   = data[:,:,0].astype(np.uint8)
          green = data[:,:,1].astype(np.uint8)
          blue  = data[:,:,2].astype(np.uint8)
          if(data.ndim == 3 and data.shape[-1] == 4) :
            alpha = data[:,:,3].astype(np.uint8)
          else:
            alpha = 255.*np.ones(blue.shape)
        else:
          self.log.warn("input veclen of "+str(data.shape[-1])+" is incompatible")
          return 1

        h, w = red.shape[:2]
        image1 = np.zeros((h, w, 4), dtype=np.uint8)
        image1[:, :, 0] = red
        image1[:, :, 1] = green
        image1[:, :, 2] = blue
        image1[:, :, 3] = alpha
        format_ = QtGui.QImage.Format_RGB32

        image = QtGui.QImage(image1.data, w, h, format_)

        #send the RGB values to the output port
        imageTru = np.zeros((h, w, 4), dtype=np.uint8)
        imageTru[:, :, 0] = red
        imageTru[:, :, 1] = green
        imageTru[:, :, 2] = blue
        imageTru[:, :, 3] = alpha
        image.ndarray = imageTru
        if image.isNull():
            self.log.warn("Image Viewer: cannot load image")
        self.setAttr('Viewport:', val=image)

        # attempt to get line end coordinates
        line = self.getAttr('Viewport:', 'line')
        if line:
          i0, j0 = line[0]
          i1, j1 = line[1]
          # slicer and volumetric mode
          if ((j0 > (green_dim[0]-1)) and (j1 > (green_dim[0]-1))) \
            and ((i0 < (green_dim[1]-1)) and (i1 < (green_dim[1]-1))):
            print("ERROR: both endpoints cannot be in the control box")
          else:
            if ((j0 > (green_dim[0]-1)) or (j1 > (green_dim[0]-1))) \
              and ((i0 < (green_dim[1]-1)) or (i1 < (green_dim[1]-1))): # slicer mode
              if (j0 > (green_dim[0]-1)) and (i0 < (green_dim[1]-1)):    # first point is in control box
                ii = i1
                jj = j1
              else:                                                     # second point is in control box
                ii = i0
                jj = j0
            else:                                                                     # volume mode
              ii = (i0 + i1)//2
              jj = (j0 + j1)//2
              i_width = abs(i0 - i1) + 1
              j_width = abs(j0 - j1) + 1
            # determine which box we are playing in
            if (ii > (green_dim[1]-1)) and  (jj > (green_dim[0]-1)):                # blue region
              # adjust coordinates to match blue region only (top corner is (0,0))
              ii -= (green_dim[1]-1)
              jj -= (green_dim[0]-1)
              self.setAttr('Coronal Slice (Green)', val=jj)
              self.setAttr('Sagittal Slice (Red)', val=ii)
            elif ii > (green_dim[1]-1):                                             # red region
              # adjust coordinates to match red region only (top corner is (0,0))
              ii -= (green_dim[1]-1)
              self.setAttr('Axial Slice (Blue)', val=jj)
              self.setAttr('Coronal Slice (Green)', val=ii)
            else:                                                                   # green region (cannot be control region)
              self.setAttr('Axial Slice (Blue)', val=jj)
              self.setAttr('Sagittal Slice (Red)', val=ii)

        self.setData('out data', combinedData)
        self.setData('sagittal slice', sagittalSlice)
        self.setData('coronal slice', coronalSlice)
        self.setData('transverse slice', transverseSlice)
        return 0