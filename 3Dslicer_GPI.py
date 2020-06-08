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
        self.addWidget('Slider', 'Dim 1 val', val=20, min=0, max=40)  # only initial values
        self.addWidget('Slider', 'Dim 2 val', val=20, min=0, max=40)
        self.addWidget('Slider', 'Dim 3 val', val=20, min=0, max=40)
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

        # IO Ports
        self.addInPort('in', 'NPYarray', drange=(2, 3))
        self.addOutPort('out data', 'NPYarray')
        self.addOutPort('sagittal slice', 'NPYarray')
        self.addOutPort('coronal slice', 'NPYarray')
        self.addOutPort('transverse slice', 'NPYarray')

    def validate(self):
        # Complex or Scalar?
        data = self.getData('in')
        dimfunc = self.getVal('Extra Dimension')
        if dimfunc > 1:
            dimfunc = 0
            self.setAttr('Extra Dimension', buttons=['Slice', 'Tile'], val=dimfunc)
        self.setAttr('Extra Dimension', visible=False)
        self.setAttr('Slice/Tile Dimension', visible=False)
        self.setAttr('Slice', visible=False)
        self.setAttr('# Rows', visible=False)
        self.setAttr('# Columns', visible=False)
        self.setAttr('L W F C:',visible=(dimfunc != 2))
        self.setAttr('Gamma',visible=(dimfunc != 2))
        self.setAttr('Fix Range',visible=(dimfunc != 2))
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
        self.setAttr('Dim 1 val', max=dim[0])
        self.setAttr('Dim 2 val', max=dim[1])
        self.setAttr('Dim 3 val', max=dim[2])
        sagittalSlice = data3d[self.getVal('Dim 1 val'), :, :]          # 150x100
        coronalSlice = data3d[:, self.getVal('Dim 2 val'), :]           # 80x100
        transverseSlice = data3d[:, :, self.getVal('Dim 3 val')]        # 80x150
        combine1 = np.append(transverseSlice, coronalSlice, axis=1)     # combine coronal and transverse slices
        pad = np.zeros([dim[1], dim[1]], dtype=np.float64)              # pad sagittal slice with zeros
        combine2 = np.append(pad, sagittalSlice, axis=1)
        combinedData = np.append(combine1, combine2, axis=0)

        # from ImageDisplay import compute (at least this is essentially what I am doing here)
        from matplotlib import cm

        # make a copy for changes
        data = combinedData

        # get extra dimension parameters and modify data
        dimfunc = self.getVal('Extra Dimension')

        # Read in parameters, make a little floor:ceiling adjustment
        gamma = self.getVal('Gamma')
        lval = self.getAttr('L W F C:', 'val')
        cval = self.getVal('Complex Display')
        cmap = self.getVal('Color Map')
        sval = self.getVal('Scalar Display')
        zval = self.getVal('Zero Ref')
        fval = self.getVal('Fix Range')
        rmin = self.getVal('Range Min')
        rmax = self.getVal('Range Max')

        flor = 0.01 * lval['floor']
        ceil = 0.01 * lval['ceiling']
        if ceil == flor:
            if ceil == 1.:
                flor = 0.999
            else:
                ceil += 0.001

        # SHOW COMPLEX DATA
        if np.iscomplexobj(data) and cval == 4:
            mag = np.abs(data)
            phase = np.angle(data, deg=True)

            # normalize the mag
            data_min = 0.
            if fval:
                data_max = rmax
            else:
                data_max = mag.max()
                self.setAttr('Range Max', val=data_max)
            data_range = data_max - data_min
            dmask = np.ones(data.shape)
            new_min = data_range * flor + data_min
            new_max = data_range * ceil + data_min
            mag = np.clip(mag, new_min, new_max)

            if new_max > new_min:
                if (gamma == 1):  # Put in check for gamma=1, the common use case, just to save time
                    mag = (mag - new_min) / (new_max - new_min)
                else:
                    mag = pow((mag - new_min) / (new_max - new_min), gamma)
            else:
                mag = np.ones(mag.shape)

            # ADD BORDERS
            edgpix = self.getVal('Edge Pixels')
            blkpix = self.getVal('Black Pixels')
            if (edgpix + blkpix) > 0:
                # new image will be h2 x w2
                # frame defines edge pixels to paint with phase table
                h, w = mag.shape
                h2 = h + 2 * (edgpix + blkpix)
                w2 = w + 2 * (edgpix + blkpix)
                mag2 = np.zeros((h2, w2))
                phase2 = np.zeros((h2, w2))
                frame = np.zeros((h2, w2)) == 1
                frame[0:edgpix, :] = frame[h2 - edgpix:h2, :] = True
                frame[:, 0:edgpix] = frame[:, w2 - edgpix:w2] = True

                mag2[edgpix + blkpix:edgpix + blkpix + h, edgpix + blkpix:edgpix + blkpix + w] = mag
                mag2[frame] = 1

                phase2[edgpix + blkpix:edgpix + blkpix + h, edgpix + blkpix:edgpix + blkpix + w] = phase
                xloc = np.tile(np.linspace(-1., 1., w2), (h2, 1))
                yloc = np.transpose(np.tile(np.linspace(1., -1., h2), (w2, 1)))
                phase2[frame] = np.degrees(np.arctan2(yloc[frame], xloc[frame]))

                mag = mag2
                phase = phase2

            # now colorize!
            if cmap == 0:  # HSV
                phase_cmap = cm.hsv
            elif cmap == 1:  # HSL
                try:
                    import seaborn as sns
                except:
                    self.log.warn("Seaborn (required for HSL map) not available! Falling back on HSV.")
                    phase_cmap = cm.hsv
                else:  # from http://stackoverflow.com/a/34557535/333308
                    import matplotlib.colors as col
                    hlsmap = col.ListedColormap(sns.color_palette("hls", 256))
                    phase_cmap = hlsmap
            elif cmap == 2:  # HUSL
                try:
                    import seaborn as sns
                except:
                    self.log.warn("Seaborn (required for HUSL map) not available! Falling back on HSV.")
                    phase_cmap = cm.hsv
                else:  # from http://stackoverflow.com/a/34557535/333308
                    import matplotlib.colors as col
                    huslmap = col.ListedColormap(sns.color_palette("husl", 256))
                    phase_cmap = huslmap
            elif cmap == 3:  # coolwarm
                phase_cmap = cm.coolwarm

            mag_norm = mag
            phase_norm = (phase + 180) / 360
            # phase shift to match old look better
            if cmap != 3:
                phase_norm = (phase_norm - 1 / 3) % 1
            colorized = 255 * cm.gray(mag_norm) * phase_cmap(phase_norm)
            red = colorized[..., 0]
            green = colorized[..., 1]
            blue = colorized[..., 2]
            alpha = colorized[..., 3]

        # DISPLAY SCALAR DATA
        elif dimfunc != 2:

            if np.iscomplexobj(data):
                if cval == 0:  # Real
                    data = np.real(data)
                elif cval == 1:  # Imag
                    data = np.imag(data)
                elif cval == 2:  # Mag
                    data = np.abs(data)
                elif cval == 3:  # Phase
                    data = np.angle(data, deg=True)

            if sval == 1:  # Mag
                data = np.abs(data)
            elif sval == 2:  # Sign
                sign = np.sign(data)
                data = np.abs(data)

            # normalize the data
            if fval:
                data_min = rmin
                data_max = rmax
            else:
                data_min = data.min()
                data_max = data.max()

            if sval != 2:
                if zval == 1:
                    data_min = 0.
                elif zval == 2:
                    data_max = max(abs(data_min), abs(data_max))
                    data_min = -data_max
                elif zval == 3:
                    data_max = 0.
                data_range = data_max - data_min
                self.setAttr('Range Min', val=data_min)
                self.setAttr('Range Max', val=data_max)
            else:
                data_min = 0.
                data_max = max(abs(data_min), abs(data_max))
                data_range = data_max
                self.setAttr('Range Min', val=-data_range)
                self.setAttr('Range Max', val=data_range)

            dmask = np.ones(data.shape)
            new_min = data_range * flor + data_min
            new_max = data_range * ceil + data_min
            data = np.minimum(np.maximum(data, new_min * dmask), new_max * dmask)

            if new_max > new_min:
                if (gamma == 1):  # Put in check for gamma=1, the common use case, just to save time
                    data = 255. * (data - new_min) / (new_max - new_min)
                else:
                    data = 255. * pow((data - new_min) / (new_max - new_min), gamma)
            else:
                data = 255. * np.ones(data.shape)

            if sval != 2:  # Not Signed Data (Pass or Mag)
                # Show based on a color map
                if cmap == 0:  # Grayscale
                    red = green = blue = np.uint8(data)
                    alpha = 255. * np.ones(blue.shape)
                else:
                    rd = np.zeros(data.shape)
                    gn = np.zeros(data.shape)
                    be = np.zeros(data.shape)
                    zmask = np.zeros(data.shape)
                    fmask = np.ones(data.shape)

                    if cmap == 1:  # IceFire
                        hue = 4. * (data / 256.)
                        hindex0 = hue < 1.
                        hindex1 = np.logical_and(hue >= 1., hue < 2.)
                        hindex2 = np.logical_and(hue >= 2., hue < 3.)
                        hindex3 = np.logical_and(hue >= 3., hue < 4.)

                        be[hindex0] = hue[hindex0]
                        gn[hindex0] = zmask[hindex0]
                        rd[hindex0] = zmask[hindex0]

                        gn[hindex1] = (hue - 1.)[hindex1]
                        rd[hindex1] = (hue - 1.)[hindex1]
                        be[hindex1] = fmask[hindex1]

                        gn[hindex2] = fmask[hindex2]
                        rd[hindex2] = fmask[hindex2]
                        be[hindex2] = (3. - hue)[hindex2]

                        rd[hindex3] = fmask[hindex3]
                        gn[hindex3] = (4. - hue)[hindex3]
                        be[hindex3] = zmask[hindex3]

                    elif cmap == 2:  # Fire
                        hue = 4. * (data / 256.)
                        hindex0 = hue < 1.
                        hindex1 = np.logical_and(hue >= 1., hue < 2.)
                        hindex2 = np.logical_and(hue >= 2., hue < 3.)
                        hindex3 = np.logical_and(hue >= 3., hue < 4.)

                        be[hindex0] = hue[hindex0]
                        rd[hindex0] = zmask[hindex0]
                        gn[hindex0] = zmask[hindex0]

                        be[hindex1] = (2. - hue)[hindex1]
                        rd[hindex1] = (hue - 1.)[hindex1]
                        gn[hindex1] = zmask[hindex1]

                        rd[hindex2] = fmask[hindex2]
                        gn[hindex2] = (hue - 2.)[hindex2]
                        be[hindex2] = zmask[hindex2]

                        rd[hindex3] = fmask[hindex3]
                        gn[hindex3] = fmask[hindex3]
                        be[hindex3] = (hue - 3.)[hindex3]

                    elif cmap == 3:  # Hot
                        hue = 3. * (data / 256.)
                        hindex0 = hue < 1.
                        hindex1 = np.logical_and(hue >= 1., hue < 2.)
                        hindex2 = np.logical_and(hue >= 2., hue < 3.)

                        rd[hindex0] = hue[hindex0]
                        be[hindex0] = zmask[hindex0]
                        gn[hindex0] = zmask[hindex0]

                        gn[hindex1] = (hue - 1.)[hindex1]
                        rd[hindex1] = fmask[hindex1]
                        be[hindex1] = zmask[hindex1]

                        rd[hindex2] = fmask[hindex2]
                        gn[hindex2] = fmask[hindex2]
                        be[hindex2] = (hue - 2.)[hindex2]

                    if cmap == 4:  # Hot2, from ASIST (http://asist.umin.jp/index-e.htm)
                        rindex0 = data < 20.0
                        rindex1 = np.logical_and(data >= 20.0, data <= 100.0)
                        rindex2 = np.logical_and(data > 100.0, data < 128.0)
                        rindex3 = np.logical_and(data >= 128.0, data <= 191.0)
                        rindex4 = data > 191.0
                        rd[rindex0] = data[rindex0] * 4.0
                        rd[rindex1] = 80.0 - (data[rindex1] - 20.0)
                        rd[rindex3] = (data[rindex3] - 128.0) * 4.0
                        rd[rindex4] = data[rindex4] * 0.0 + 255.0
                        rd = rd / 255.0;

                        gindex0 = data < 45.0
                        gindex1 = np.logical_and(data >= 45.0, data <= 130.0)
                        gindex2 = np.logical_and(data > 130.0, data < 192.0)
                        gindex3 = data >= 192.0
                        gn[gindex1] = (data[gindex1] - 45.0) * 3.0
                        gn[gindex2] = data[gindex2] * 0.0 + 255.0
                        gn[gindex3] = 252.0 - (data[gindex3] - 192.0) * 4.0
                        gn = gn / 255.0

                        bindex0 = (data < 1.0)
                        bindex1 = np.logical_and(data >= 1.0, data < 86.0)
                        bindex2 = np.logical_and(data >= 86.0, data <= 137.0)
                        bindex3 = data > 137.0
                        be[bindex1] = (data[bindex1] - 1.0) * 3.0
                        be[bindex2] = 255.0 - (data[bindex2] - 86.0) * 5.0
                        be = be / 255.0

                    elif cmap == 5:  # BGR
                        hue = 4. * (data / 256.)
                        hindex0 = hue < 1.
                        hindex1 = np.logical_and(hue >= 1., hue < 2.)
                        hindex2 = np.logical_and(hue >= 2., hue < 3.)
                        hindex3 = np.logical_and(hue >= 3., hue < 4.)

                        be[hindex0] = hue[hindex0]
                        gn[hindex0] = zmask[hindex0]
                        rd[hindex0] = zmask[hindex0]

                        gn[hindex1] = (hue - 1.)[hindex1]
                        rd[hindex1] = zmask[hindex1]
                        be[hindex1] = fmask[hindex1]

                        gn[hindex2] = fmask[hindex2]
                        rd[hindex2] = (hue - 2.)[hindex2]
                        be[hindex2] = (3. - hue)[hindex2]

                        rd[hindex3] = fmask[hindex3]
                        gn[hindex3] = (4. - hue)[hindex3]
                        be[hindex3] = zmask[hindex3]

                    blue = np.uint8(255. * rd)
                    red = np.uint8(255. * be)
                    green = np.uint8(255. * gn)
                    alpha = np.uint8(255. * np.ones(blue.shape))

            else:  # Signed data, positive numbers green, negative numbers magenta
                red = np.zeros(data.shape)
                green = np.zeros(data.shape)
                blue = np.zeros(data.shape)
                red[sign <= 0] = data[sign <= 0]
                blue[sign <= 0] = data[sign <= 0]
                green[sign >= 0] = data[sign >= 0]

                red = red.astype(np.uint8)
                green = green.astype(np.uint8)
                blue = blue.astype(np.uint8)
                alpha = np.uint8(data)

        # DISPLAY RGB image
        else:

            if data.shape[-1] > 3:
                red = data[:, :, 0].astype(np.uint8)
                green = data[:, :, 1].astype(np.uint8)
                blue = data[:, :, 2].astype(np.uint8)
                if (data.ndim == 3 and data.shape[-1] == 4):
                    alpha = data[:, :, 3].astype(np.uint8)
                else:
                    alpha = 255. * np.ones(blue.shape)
            else:
                self.log.warn("input veclen of " + str(data.shape[-1]) + " is incompatible")
                return 1

        h, w = red.shape[:2]
        image1 = np.zeros((h, w, 4), dtype=np.uint8)
        image1[:, :, 0] = red
        image1[:, :, 1] = green
        image1[:, :, 2] = blue
        image1[:, :, 3] = alpha
        format_ = QtGui.QImage.Format_RGB32

        image = QtGui.QImage(image1.data, w, h, format_)

        # send the RGB values to the output port
        imageTru = np.zeros((h, w, 4), dtype=np.uint8)
        imageTru[:, :, 0] = red
        imageTru[:, :, 1] = green
        imageTru[:, :, 2] = blue
        imageTru[:, :, 3] = alpha
        image.ndarray = imageTru
        if image.isNull():
            self.log.warn("Image Viewer: cannot load image")

        self.setAttr('Viewport:', val=image)


        # how do I actually output this data here into the viewports (I essentially need to include ImageDisplay)
        self.setData('out data', combinedData)
        self.setData('sagittal slice', sagittalSlice)
        self.setData('coronal slice', coronalSlice)
        self.setData('transverse slice', transverseSlice)
        return 0