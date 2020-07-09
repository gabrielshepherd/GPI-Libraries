# Copyright (c) 2014, Dignity Health
#
#     The GPI core node library is licensed under
# either the BSD 3-clause or the LGPL v. 3.
#
#     Under either license, the following additional term applies:
#
#         NO CLINICAL USE.  THE SOFTWARE IS NOT INTENDED FOR COMMERCIAL
# PURPOSES AND SHOULD BE USED ONLY FOR NON-COMMERCIAL RESEARCH PURPOSES.  THE
# SOFTWARE MAY NOT IN ANY EVENT BE USED FOR ANY CLINICAL OR DIAGNOSTIC
# PURPOSES.  YOU ACKNOWLEDGE AND AGREE THAT THE SOFTWARE IS NOT INTENDED FOR
# USE IN ANY HIGH RISK OR STRICT LIABILITY ACTIVITY, INCLUDING BUT NOT LIMITED
# TO LIFE SUPPORT OR EMERGENCY MEDICAL OPERATIONS OR USES.  LICENSOR MAKES NO
# WARRANTY AND HAS NOR LIABILITY ARISING FROM ANY USE OF THE SOFTWARE IN ANY
# HIGH RISK OR STRICT LIABILITY ACTIVITIES.
#
#     If you elect to license the GPI core node library under the LGPL the
# following applies:
#
#         This file is part of the GPI core node library.
#
#         The GPI core node library is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version. GPI core node library is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
#         You should have received a copy of the GNU Lesser General Public
# License along with the GPI core node library. If not, see
# <http://www.gnu.org/licenses/>.

# Author: Jim Pipe / Nick Zwart
# Date: 2013 Sep 01

import gpi
from gpi import QtGui, QtWidgets
import numpy as np

def colorthings(obj, data, dimfunc):# Read in parameters, make a little floor:ceiling adjustment
  gamma = obj.getVal('Gamma')
  lval = obj.getAttr('L W F C:', 'val')
  cval = obj.getVal('Complex Display')

  if 'Complex Display' in obj.widgetEvents():
    if cval == 4:
      obj.setAttr('Color Map', buttons=obj.complex_cmaps,
                    collapsed=obj.getAttr('Color Map', 'collapsed'),
                    val=0)
    else:
      obj.setAttr('Color Map', buttons=obj.real_cmaps,
                    collapsed=obj.getAttr('Color Map', 'collapsed'),
                    val=0)

  cmap = obj.getVal('Color Map')
  sval = obj.getVal('Scalar Display')
  zval = obj.getVal('Zero Ref')
  fval = obj.getVal('Fix Range')
  rmin = obj.getVal('Range Min')
  rmax = obj.getVal('Range Max')

  flor = 0.01*lval['floor']
  ceil = 0.01*lval['ceiling']
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
      obj.setAttr('Range Max',val=data_max)
    data_range = data_max-data_min
    dmask = np.ones(data.shape)
    new_min = data_range*flor + data_min
    new_max = data_range*ceil + data_min
    mag = np.clip(mag, new_min, new_max)

    if new_max > new_min:
      if (gamma == 1): # Put in check for gamma=1, the common use case, just to save time
        mag = (mag - new_min)/(new_max-new_min)
      else:
        mag = pow((mag - new_min)/(new_max-new_min),gamma)
    else:
      mag = np.ones(mag.shape)

    # ADD BORDERS
    edgpix = obj.getVal('Edge Pixels')
    blkpix = obj.getVal('Black Pixels')
    if (edgpix + blkpix) > 0:
      # new image will be h2 x w2
      # frame defines edge pixels to paint with phase table
      h, w = mag.shape
      h2 = h + 2*(edgpix+blkpix)
      w2 = w + 2*(edgpix+blkpix)
      mag2 = np.zeros((h2,w2))
      phase2 = np.zeros((h2,w2))
      frame = np.zeros((h2,w2)) == 1
      frame[0:edgpix,:] = frame[h2-edgpix:h2,:] = True
      frame[:,0:edgpix] = frame[:,w2-edgpix:w2] = True

      mag2[edgpix+blkpix:edgpix+blkpix+h,edgpix+blkpix:edgpix+blkpix+w] = mag
      mag2[frame] = 1

      phase2[edgpix+blkpix:edgpix+blkpix+h,edgpix+blkpix:edgpix+blkpix+w] = phase
      xloc = np.tile(np.linspace(-1.,1.,w2),(h2,1))
      yloc = np.transpose(np.tile(np.linspace(1.,-1.,h2),(w2,1)))
      phase2[frame] = np.degrees(np.arctan2(yloc[frame],xloc[frame]))

      mag = mag2
      phase = phase2

    # now colorize!
    if cmap == 0: # HSV
      phase_cmap = cm.hsv
    elif cmap == 1: # HSL
      try:
        import seaborn as sns
      except:
        # self.log.warn("Seaborn (required for HSL map) not available! Falling back on HSV.")
        phase_cmap = cm.hsv
      else: # from http://stackoverflow.com/a/34557535/333308
        import matplotlib.colors as col
        hlsmap = col.ListedColormap(sns.color_palette("hls", 256))
        phase_cmap = hlsmap
    elif cmap == 2: #HUSL
      try:
        import seaborn as sns
      except:
        # self.log.warn("Seaborn (required for HUSL map) not available! Falling back on HSV.")
        phase_cmap = cm.hsv
      else: # from http://stackoverflow.com/a/34557535/333308
        import matplotlib.colors as col
        huslmap = col.ListedColormap(sns.color_palette("husl", 256))
        phase_cmap = huslmap
    elif cmap == 3: # coolwarm
      phase_cmap = cm.coolwarm

    mag_norm = mag
    phase_norm = (phase + 180) / 360
    # phase shift to match old look better
    if cmap != 3:
      phase_norm = (phase_norm - 1/3) % 1
    colorized = 255 * cm.gray(mag_norm) * phase_cmap(phase_norm)
    red = colorized[...,0]
    green = colorized[...,1]
    blue = colorized[...,2]
    alpha = colorized[...,3]

  # DISPLAY SCALAR DATA
  elif dimfunc != 2:

    if np.iscomplexobj(data):
      if cval == 0: # Real
        data = np.real(data)
      elif cval == 1: # Imag
        data = np.imag(data)
      elif cval == 2: # Mag
        data = np.abs(data)
      elif cval == 3: # Phase
        data = np.angle(data, deg=True)

    if sval == 1: # Mag
      data = np.abs(data)
    elif sval == 2: # Sign
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
        data_max = max(abs(data_min),abs(data_max))
        data_min = -data_max
      elif zval == 3:
        data_max = 0.
      data_range = data_max-data_min
      obj.setAttr('Range Min',val=data_min)
      obj.setAttr('Range Max',val=data_max)
    else:
      data_min = 0.
      data_max = max(abs(data_min),abs(data_max))
      data_range = data_max
      obj.setAttr('Range Min',val=-data_range)
      obj.setAttr('Range Max',val=data_range)

    dmask = np.ones(data.shape)
    new_min = data_range*flor + data_min
    new_max = data_range*ceil + data_min
    data = np.minimum(np.maximum(data,new_min*dmask),new_max*dmask)

    if new_max > new_min:
      if (gamma == 1): # Put in check for gamma=1, the common use case, just to save time
        data = 255.*(data - new_min)/(new_max-new_min)
      else:
        data = 255.*pow((data - new_min)/(new_max-new_min),gamma)
    else:
      data = 255.*np.ones(data.shape)

    if sval != 2: #Not Signed Data (Pass or Mag)
      # Show based on a color map
      if cmap == 0: # Grayscale
        red = green = blue = np.uint8(data)
        alpha = 255. * np.ones(blue.shape)
      else:
        rd = np.zeros(data.shape)
        gn = np.zeros(data.shape)
        be = np.zeros(data.shape)
        zmask = np.zeros(data.shape)
        fmask = np.ones(data.shape)

        if cmap == 1: # IceFire
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          gn[hindex0] = zmask[hindex0]
          rd[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = (hue-1.)[hindex1]
          be[hindex1] = fmask[hindex1]

          gn[hindex2] = fmask[hindex2]
          rd[hindex2] = fmask[hindex2]
          be[hindex2] = (3.-hue)[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = (4.-hue)[hindex3]
          be[hindex3] = zmask[hindex3]

        elif cmap == 2: # Fire
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          rd[hindex0] = zmask[hindex0]
          gn[hindex0] = zmask[hindex0]

          be[hindex1] = (2.-hue)[hindex1]
          rd[hindex1] = (hue-1.)[hindex1]
          gn[hindex1] = zmask[hindex1]

          rd[hindex2] = fmask[hindex2]
          gn[hindex2] = (hue-2.)[hindex2]
          be[hindex2] = zmask[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = fmask[hindex3]
          be[hindex3] = (hue-3.)[hindex3]

        elif cmap == 3: # Hot
          hue = 3.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)

          rd[hindex0] = hue[hindex0]
          be[hindex0] = zmask[hindex0]
          gn[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = fmask[hindex1]
          be[hindex1] = zmask[hindex1]

          rd[hindex2] = fmask[hindex2]
          gn[hindex2] = fmask[hindex2]
          be[hindex2] = (hue-2.)[hindex2]

        if cmap == 4: # Hot2, from ASIST (http://asist.umin.jp/index-e.htm)
          rindex0 = data < 20.0
          rindex1 = np.logical_and(data >=  20.0, data <= 100.0)
          rindex2 = np.logical_and(data > 100.0, data < 128.0)
          rindex3 = np.logical_and(data >= 128.0, data <= 191.0)
          rindex4 = data > 191.0
          rd[rindex0] = data[rindex0] * 4.0
          rd[rindex1] = 80.0 - (data[rindex1] - 20.0)
          rd[rindex3] = (data[rindex3] - 128.0) * 4.0
          rd[rindex4] = data[rindex4] * 0.0 + 255.0
          rd = rd/255.0

          gindex0 = data < 45.0
          gindex1 = np.logical_and(data >= 45.0, data <= 130.0)
          gindex2 = np.logical_and(data > 130.0, data < 192.0)
          gindex3 = data >= 192.0
          gn[gindex1] = (data[gindex1] - 45.0)*3.0
          gn[gindex2] = data[gindex2] * 0.0 + 255.0
          gn[gindex3] = 252.0 - (data[gindex3] - 192.0)*4.0
          gn = gn/255.0

          bindex0 = (data < 1.0)
          bindex1 = np.logical_and(data >= 1.0, data < 86.0)
          bindex2 = np.logical_and(data >= 86.0, data <= 137.0)
          bindex3 = data > 137.0
          be[bindex1] = (data[bindex1] - 1.0)*3.0
          be[bindex2] = 255.0 - (data[bindex2] - 86.0)*5.0
          be = be/255.0

        elif cmap == 5: # BGR
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          gn[hindex0] = zmask[hindex0]
          rd[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = zmask[hindex1]
          be[hindex1] = fmask[hindex1]

          gn[hindex2] = fmask[hindex2]
          rd[hindex2] = (hue-2.)[hindex2]
          be[hindex2] = (3.-hue)[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = (4.-hue)[hindex3]
          be[hindex3] = zmask[hindex3]

        blue = np.uint8(255.*rd)
        red = np.uint8(255.*be)
        green = np.uint8(255.*gn)
        alpha = np.uint8(255.*np.ones(blue.shape))

    else: #Signed data, positive numbers green, negative numbers magenta
      red = np.zeros(data.shape)
      green = np.zeros(data.shape)
      blue = np.zeros(data.shape)
      red[sign<=0] = data[sign<=0]
      blue[sign<=0] = data[sign<=0]
      green[sign>=0] = data[sign>=0]

      red = red.astype(np.uint8)
      green = green.astype(np.uint8)
      blue = blue.astype(np.uint8)
      alpha = np.uint8(data)

  # DISPLAY RGB image
  else:

    if data.shape[-1] > 3:
      red   = data[:,:,0].astype(np.uint8)
      green = data[:,:,1].astype(np.uint8)
      blue  = data[:,:,2].astype(np.uint8)
      if(data.ndim == 3 and data.shape[-1] == 4):
          alpha = data[:,:,3].astype(np.uint8)
      else:
          alpha = 255.*np.ones(blue.shape)
    else:
        obj.log.warn("input veclen of "+str(data.shape[-1])+" is incompatible")
        return 1

  # combine data into 4D array and output from widget (not to viewport)
  h, w = red.shape[:2]
  imageTru = np.zeros((h, w, 4), dtype=np.uint8)
  imageTru[:, :, 0] = red
  imageTru[:, :, 1] = green
  imageTru[:, :, 2] = blue
  imageTru[:, :, 3] = alpha
  return imageTru

def process_slice(obj, data, dimfunc, outport, islice, jslice, kslice, i_width, j_width, k_width):

  imageTru = colorthings(obj, data, dimfunc)
  dim = list(np.shape(imageTru))

  # define colors --> I have no idea why the colors work this way but this is the correct formatting
  r = [0, 0, 255, 0]
  b = [255, 0, 0, 0]
  g = [0, 255, 0, 0]

  # define border and crosshair colors by plane; also define widths for that dimension
  if outport == 'sagittal slice':
    border = g
    hori_crosshair = b
    vert_crosshair = r
    jwidth = i_width
    iwidth = j_width
    icenter = islice
    jcenter = jslice
  elif outport == 'coronal slice':
    border = r
    hori_crosshair = b
    vert_crosshair = g
    jwidth = i_width
    iwidth = k_width
    icenter = islice
    jcenter = kslice
  elif outport == 'transverse slice':
    border = b
    hori_crosshair = r
    vert_crosshair = g
    iwidth = j_width
    jwidth = k_width
    icenter = jslice
    jcenter = kslice
    
  # add border
  img = np.stack([np.pad(imageTru[:,:,c], ([2, 2]), mode='constant', constant_values=border[c]) for c in range(4)], axis=2)

  # add crosshairs
  vert1 = jcenter - ((iwidth - 1)//2)
  vert2 = jcenter + (iwidth//2)
  hori1 = icenter - ((jwidth - 1)//2)
  hori2 = icenter + (jwidth//2)
  print("iwidth: " + str(iwidth))
  print("jwidth: " + str(jwidth))
  if iwidth == 1 and jwidth == 1:
    for i in range(2, dim[0]-1):
      if i < (icenter-5) or i > (icenter+5):
        img[i, vert1] = vert_crosshair
    for j in range(2, dim[1]-1):
      if j <(jcenter-5) or j > (jcenter+5):
        img[hori1, j] = hori_crosshair
  else:
    for i in range(2, dim[0]-1):
      img[i, vert1] = vert_crosshair
      img[i, vert2] = vert_crosshair
    for j in range(2, dim[1]-1):
      img[hori1, j] = hori_crosshair
      img[hori2, j] = hori_crosshair
  return img, imageTru

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
  """2D image viewer for real or complex NPYarrays.

  INPUT:
  2D data, real or complex
  3D uint8 ARGB data (e.g. output of another ImageDisplay node)

  OUTPUT:
  3D data of displayed image, last dimension has length 4 for ARGB byte (uint8) data

  WIDGETS:
  Complex Display - If data are complex, allows you to show Real, Imaginary, Magnitude, Phase, or "Complex" data.
      If C (Complex) is chosen, then pixel brightness reflects value magnitude, while pixel color reflects value phase.
      If input data are real-valued, this widget is hidden

  Color Map - Chooses from a number of colormaps for real-valued data.  Not available if Scalar Display is "Sign"

  Edge Pixels - only visible for complex input data with Complex Display set to "C"
      Setting this to N creates an N-pixel color ring around the image border illustrating the phase-to-color mapping

  Black Pixels - only visible for complex input data with Complex Display set to "C"
      Setting this to N creates an N-pixel black ring around the image border (but inside the Edge Pixels ring) to
      separate the Edge pixel ring from the actual data image

  Viewport - displays the image
    Double clicking on Viewport area brings up a scaling widget to change the image size, and change graphic overlay

  L W F C - (hidden by default - double click on widget area to show sliders)
            Adjust value-to-pixel brightness mapping using Level/Window or Floor/Ceiling

  Scalar Display - visible for real data, or complex data with "Complex Display" set to R, I, M, or P
    Pass uses the real data in the data-to-pixel mapping
    Mag uses the magnitude data in the data-to-pixel mapping (i.e. affects negative values only)
    Sign will display positive values in green, and the absolutele value of negative values in magenta

  Gamma - changes gamma of display function.  Default value of 1 gives linear mapping of data to pixel value
    pixel values refect value of data^gamma

  Zero Ref - visible for real data, or complex data with "Complex Display" set to R, I, M, or P
              also invisible if "Scalar Display" set to Sign
    This is used for the data-to-pixel value mapping
    --- maps the smallest value to black, and the largest value to white
    0-> maps zero to black, and the largest value to white.  All negative numbers are black
    -0- maps zero to middle gray, with the largest magnitude set to black (if negative) or white (if positive)
    <-0 maps zero to white, and the most negative value to black.  All positive numbers are white

  Fix Range
    If Auto-Range On, the data range for pixel value mapping is rescaled whenever new data appears at input
    If Fixed-Range On, the data range is fixed (and can be changed using Range Min and Range Max)

  Range Min - shows minimum data value used for mapping to pixel values
    This value can be changed if (and only if) "Fix Range" is set to "Fixed-Ranged On"

  Range Max - shows maximum data value used for mapping to pixel values
    This value can be changed if (and only if) "Fix Range" is set to "Fixed-Ranged On"
  """

  def execType(self):
    return gpi.GPI_PROCESS
  
  def initUI(self):

    # Widgets
    self.addWidget('ExclusivePushButtons','Complex Display',
                    buttons=['R','I','M','P','C'], val=4)
    self.real_cmaps = ['Gray','IceFire','Fire','Hot','HOT2','BGR']
    self.complex_cmaps = ['HSV','HSL','HUSL','CoolWarm']
    self.addWidget('ExclusivePushButtons','Color Map',
                    buttons=self.real_cmaps, val=0, collapsed=True)
    self.addWidget('SpinBox', 'Edge Pixels', min=0)
    self.addWidget('SpinBox', 'Black Pixels', min=0)
    self.addWidget('DisplayBox', 'Viewport:')
    self.addWidget('Slider', 'Axial Slice (Blue)', val=1, min=0, max=40)  # only initial values
    self.addWidget('Slider', 'Coronal Slice (Red)', val=1, min=0, max=40)
    self.addWidget('Slider', 'Sagittal Slice (Green)', val=1, min=0, max=40)
    self.addWidget('SpinBox', '# Columns', val=1)
    self.addWidget('SpinBox', '# Rows', val=1)
    self.addWidget('WindowLevel', 'L W F C:', collapsed=True)
    self.addWidget('ExclusivePushButtons','Scalar Display',
                    buttons=['Pass','Mag','Sign'], val=0)
    self.addWidget('DoubleSpinBox', 'Gamma',min=0.1,max=10,val=1,singlestep=0.05,decimals=3)
    self.addWidget('ExclusivePushButtons','Zero Ref',
                    buttons=['---','0->','-0-','<-0'], val=0)
    self.addWidget('PushButton', 'Fix Range', button_title='Auto-Range On', toggle=True)
    self.addWidget('DoubleSpinBox', 'Range Min')
    self.addWidget('DoubleSpinBox', 'Range Max')
    self.addWidget('SpinBox', 'i width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'j width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'k width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'volume', min=0, max=1, val=0)

    # IO Ports
    self.addInPort('in', 'NPYarray', drange=(2,3))
    self.addOutPort('out', 'NPYarray')
    self.addOutPort('sagittal slice', 'NPYarray')
    self.addOutPort('coronal slice', 'NPYarray')
    self.addOutPort('transverse slice', 'NPYarray')

  def validate(self):
    data = self.getData('in')
    if np.iscomplexobj(data):
      self.setAttr('Complex Display',visible=True)
      scalarvis = self.getVal('Complex Display') != 4
    else:
      self.setAttr('Complex Display',visible=False)
      scalarvis = True        
    self.setAttr('Scalar Display',visible=scalarvis)
    self.setAttr('Edge Pixels',visible=not scalarvis)
    self.setAttr('Black Pixels',visible=not scalarvis)
    self.setAttr('Zero Ref',visible=False)
    self.setAttr('# Rows', visible=False)
    self.setAttr('# Columns', visible=False)
    self.setAttr('i width', visible=False)
    self.setAttr('j width', visible=False)
    self.setAttr('k width', visible=False)
    if self.getVal('Fix Range'):
      self.setAttr('Fix Range',button_title="Fixed Range On")
    else:
      self.setAttr('Fix Range',button_title="Auto-Range On")
    return 0

  def compute(self):

    # events = self.getEvents()
    # print(str(events['_WDG_EVENT_']))

    from matplotlib import cm

    # get and format data
    data3d = self.getData('in').astype(np.float64)
    dim = list(np.shape(data3d))
    save = False

    # slicer and volume mode here
    line = self.getAttr('Viewport:', 'line')
    if line:
      red_dim = [dim[0], dim[2]]
      i0, j0 = line[0]
      i1, j1 = line[1]
      # is there an endpoint in the control box?
      if ((j0 > (red_dim[0]-1)) or (j1 > (red_dim[0]-1))) and ((i0 < (red_dim[1]-1)) or (i1 < (red_dim[1]-1))):
        if (j0 > (red_dim[0]-1)) and (j1 > (red_dim[0]-1)) and (i0 < (red_dim[1]-1)) and (i1 < (red_dim[1]-1)):
          save = True
        else:
          self.setAttr('volume', val=0)
          self.setAttr('i width', val=1)
          self.setAttr('j width', val=1)
          self.setAttr('k width', val=1)
          if (j0 > (red_dim[0]-1)) and (i0 < (red_dim[1]-1)):                 # first point is in control box
            ii = i1
            jj = j1
          else:                                                               # second point is in control box
            ii = i0
            jj = j0
      else:                       # volume mode
        self.setAttr('volume', val=1)
        # calculate centers
        ii = (i0 + i1)//2
        jj = (j0 + j1)//2
        # calculate widths
        i_width = abs(i0 - i1) + 1
        j_width = abs(j0 - j1) + 1
        # check to make sure that both endpoints are in the same zone
        if i0 > (red_dim[1]+3) and  j0 > (red_dim[0]+3) and i1 > (red_dim[1]+3) and  j1 > (red_dim[0]+3):
          self.setAttr('j width', val=i_width)
          self.setAttr('k width', val=j_width)
        elif i0 > (red_dim[1]+3) and  i1 > (red_dim[1]+3) and j0 < (red_dim[0]+3) and j1 < (red_dim[1]+3):
          self.setAttr('i width', val=j_width)
          self.setAttr('j width', val=i_width)
        elif i0 < (red_dim[1]+3) and  i1 < (red_dim[1]+3) and j0 < (red_dim[0]+3) and j1 < (red_dim[1]+3):
          self.setAttr('i width', val=j_width)
          self.setAttr('k width', val=i_width)
        else:
          self.log.error("You cannot draw a box in two different slices!")
      # update centers based on which zone we are in
      if save is not True:
        if (ii > (red_dim[1]+3)) and  (jj > (red_dim[0]+3)):
          ii -= (red_dim[1]+4)
          jj -= (red_dim[0]+4)
          self.setAttr('Coronal Slice (Red)', val=jj)
          self.setAttr('Sagittal Slice (Green)', val=ii)
        elif ii > (red_dim[1]+3):
          ii -= (red_dim[1]+4)
          self.setAttr('Axial Slice (Blue)', val=jj)
          self.setAttr('Coronal Slice (Red)', val=ii)
        else:
          self.setAttr('Axial Slice (Blue)', val=jj)
          self.setAttr('Sagittal Slice (Green)', val=ii)

    # reset default values with the values from the dimensions of the input data
    self.setAttr('Axial Slice (Blue)', max=dim[0]-1)
    self.setAttr('Coronal Slice (Red)', max=dim[1]-1)
    self.setAttr('Sagittal Slice (Green)', max=dim[2]-1)
    self.setAttr('i width', max=dim[0]-1)
    self.setAttr('j width', max=dim[1]-1)
    self.setAttr('k width', max=dim[2]-1)
    islice = self.getVal('Axial Slice (Blue)')
    jslice = self.getVal('Coronal Slice (Red)')
    kslice = self.getVal('Sagittal Slice (Green)')
    iwidth = self.getVal('i width')
    jwidth = self.getVal('j width')
    kwidth = self.getVal('k width')
    axialSlice = data3d[islice, :, :] 
    coronalSlice = data3d[:, jslice, :]
    sagittalSlice = data3d[:, :, kslice]
    maximum = np.max([np.max(sagittalSlice), np.max(coronalSlice), np.max(axialSlice)])
    minimum = np.min([np.min(sagittalSlice), np.min(coronalSlice), np.min(axialSlice)])
    self.setAttr('Range Min', val=minimum)
    self.setAttr('Range Max', val=maximum)

    # get and process slices
    greenRegion, sagittalSliceImg = process_slice(self, sagittalSlice, 0, 'sagittal slice', islice, jslice, kslice, iwidth, jwidth, kwidth)
    redRegion, coronalSliceImg = process_slice(self, coronalSlice, 0, 'coronal slice', islice, jslice, kslice, iwidth, jwidth, kwidth)
    blueRegion, axialSliceImg = process_slice(self, axialSlice, 0, 'transverse slice', islice, jslice, kslice, iwidth, jwidth, kwidth)

    # combine all slices into one image
    combine1 = np.append(redRegion, greenRegion, axis=1)
    pad = np.zeros([dim[1]+4, dim[1]+4, 4], dtype=greenRegion.dtype)
    combine2 = np.append(pad, blueRegion, axis=1)
    img = np.append(combine1, combine2, axis=0)

    # format and display image in Viewport
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    alpha = img[:, :, 3]
    h, w = red.shape[:2]
    image1 = np.zeros((h, w, 4), dtype=np.uint8)
    image1[:, :, 0] = red
    image1[:, :, 1] = green
    image1[:, :, 2] = blue
    image1[:, :, 3] = alpha
    format_ = QtGui.QImage.Format_RGB32
    image = QtGui.QImage(image1.data, w, h, format_)
    if image.isNull():
      self.log.warn("Image Viewer: cannot load image")
    self.setAttr('Viewport:', val=image)

    # output volume and slices
    if save is True or True:                            # override to always save for now
      zz = int(self.getVal('Sagittal Slice (Green)'))
      yy = int(self.getVal('Coronal Slice (Red)'))
      xx = int(self.getVal('Axial Slice (Blue)'))
      print("xx: " + str(xx))
      print("yy: " + str(yy))
      print("zz: " + str(zz))
      # retrieve volume
      if int(self.getVal('volume')) == 1:
        # get widths
        x_width = int(self.getVal('i width'))
        y_width = int(self.getVal('j width'))
        z_width = int(self.getVal('k width'))
        print("x width: " + str(x_width))
        print("y width: " + str(y_width))
        print("z width: " + str(z_width))
        # calculate ranges for x, y & z
        x0 = xx - ((x_width-1)//2)
        x1 = xx + (x_width//2)
        y0 = yy - ((y_width-1)//2)
        y1 = yy + (y_width//2)
        z0 = zz - ((z_width-1)//2)
        z1 = zz + (z_width//2)
        print("x0: " + str(x0))
        print("x1: " + str(x1))
        print("y0: " + str(y0))
        print("y1: " + str(y1))
        print("z0: " + str(z0))
        print("z1: " + str(z1))
        # output array
        if x0 == x1: x1 += 1
        elif y0 == y1: y1 += 1
        elif z0 == z1: z1 += 1
        if x_width != 1: x_width = x1 - x0
        if y_width != 1: y_width = y1 - y0
        if z_wid# Copyright (c) 2014, Dignity Health
#
#     The GPI core node library is licensed under
# either the BSD 3-clause or the LGPL v. 3.
#
#     Under either license, the following additional term applies:
#
#         NO CLINICAL USE.  THE SOFTWARE IS NOT INTENDED FOR COMMERCIAL
# PURPOSES AND SHOULD BE USED ONLY FOR NON-COMMERCIAL RESEARCH PURPOSES.  THE
# SOFTWARE MAY NOT IN ANY EVENT BE USED FOR ANY CLINICAL OR DIAGNOSTIC
# PURPOSES.  YOU ACKNOWLEDGE AND AGREE THAT THE SOFTWARE IS NOT INTENDED FOR
# USE IN ANY HIGH RISK OR STRICT LIABILITY ACTIVITY, INCLUDING BUT NOT LIMITED
# TO LIFE SUPPORT OR EMERGENCY MEDICAL OPERATIONS OR USES.  LICENSOR MAKES NO
# WARRANTY AND HAS NOR LIABILITY ARISING FROM ANY USE OF THE SOFTWARE IN ANY
# HIGH RISK OR STRICT LIABILITY ACTIVITIES.
#
#     If you elect to license the GPI core node library under the LGPL the
# following applies:
#
#         This file is part of the GPI core node library.
#
#         The GPI core node library is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version. GPI core node library is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
#         You should have received a copy of the GNU Lesser General Public
# License along with the GPI core node library. If not, see
# <http://www.gnu.org/licenses/>.

# Author: Jim Pipe / Nick Zwart
# Date: 2013 Sep 01

import gpi
from gpi import QtGui, QtWidgets
import numpy as np

def colorthings(obj, data, dimfunc):# Read in parameters, make a little floor:ceiling adjustment
  gamma = obj.getVal('Gamma')
  lval = obj.getAttr('L W F C:', 'val')
  cval = obj.getVal('Complex Display')

  if 'Complex Display' in obj.widgetEvents():
    if cval == 4:
      obj.setAttr('Color Map', buttons=obj.complex_cmaps,
                    collapsed=obj.getAttr('Color Map', 'collapsed'),
                    val=0)
    else:
      obj.setAttr('Color Map', buttons=obj.real_cmaps,
                    collapsed=obj.getAttr('Color Map', 'collapsed'),
                    val=0)

  cmap = obj.getVal('Color Map')
  sval = obj.getVal('Scalar Display')
  zval = obj.getVal('Zero Ref')
  fval = obj.getVal('Fix Range')
  rmin = obj.getVal('Range Min')
  rmax = obj.getVal('Range Max')

  flor = 0.01*lval['floor']
  ceil = 0.01*lval['ceiling']
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
      obj.setAttr('Range Max',val=data_max)
    data_range = data_max-data_min
    dmask = np.ones(data.shape)
    new_min = data_range*flor + data_min
    new_max = data_range*ceil + data_min
    mag = np.clip(mag, new_min, new_max)

    if new_max > new_min:
      if (gamma == 1): # Put in check for gamma=1, the common use case, just to save time
        mag = (mag - new_min)/(new_max-new_min)
      else:
        mag = pow((mag - new_min)/(new_max-new_min),gamma)
    else:
      mag = np.ones(mag.shape)

    # ADD BORDERS
    edgpix = obj.getVal('Edge Pixels')
    blkpix = obj.getVal('Black Pixels')
    if (edgpix + blkpix) > 0:
      # new image will be h2 x w2
      # frame defines edge pixels to paint with phase table
      h, w = mag.shape
      h2 = h + 2*(edgpix+blkpix)
      w2 = w + 2*(edgpix+blkpix)
      mag2 = np.zeros((h2,w2))
      phase2 = np.zeros((h2,w2))
      frame = np.zeros((h2,w2)) == 1
      frame[0:edgpix,:] = frame[h2-edgpix:h2,:] = True
      frame[:,0:edgpix] = frame[:,w2-edgpix:w2] = True

      mag2[edgpix+blkpix:edgpix+blkpix+h,edgpix+blkpix:edgpix+blkpix+w] = mag
      mag2[frame] = 1

      phase2[edgpix+blkpix:edgpix+blkpix+h,edgpix+blkpix:edgpix+blkpix+w] = phase
      xloc = np.tile(np.linspace(-1.,1.,w2),(h2,1))
      yloc = np.transpose(np.tile(np.linspace(1.,-1.,h2),(w2,1)))
      phase2[frame] = np.degrees(np.arctan2(yloc[frame],xloc[frame]))

      mag = mag2
      phase = phase2

    # now colorize!
    if cmap == 0: # HSV
      phase_cmap = cm.hsv
    elif cmap == 1: # HSL
      try:
        import seaborn as sns
      except:
        # self.log.warn("Seaborn (required for HSL map) not available! Falling back on HSV.")
        phase_cmap = cm.hsv
      else: # from http://stackoverflow.com/a/34557535/333308
        import matplotlib.colors as col
        hlsmap = col.ListedColormap(sns.color_palette("hls", 256))
        phase_cmap = hlsmap
    elif cmap == 2: #HUSL
      try:
        import seaborn as sns
      except:
        # self.log.warn("Seaborn (required for HUSL map) not available! Falling back on HSV.")
        phase_cmap = cm.hsv
      else: # from http://stackoverflow.com/a/34557535/333308
        import matplotlib.colors as col
        huslmap = col.ListedColormap(sns.color_palette("husl", 256))
        phase_cmap = huslmap
    elif cmap == 3: # coolwarm
      phase_cmap = cm.coolwarm

    mag_norm = mag
    phase_norm = (phase + 180) / 360
    # phase shift to match old look better
    if cmap != 3:
      phase_norm = (phase_norm - 1/3) % 1
    colorized = 255 * cm.gray(mag_norm) * phase_cmap(phase_norm)
    red = colorized[...,0]
    green = colorized[...,1]
    blue = colorized[...,2]
    alpha = colorized[...,3]

  # DISPLAY SCALAR DATA
  elif dimfunc != 2:

    if np.iscomplexobj(data):
      if cval == 0: # Real
        data = np.real(data)
      elif cval == 1: # Imag
        data = np.imag(data)
      elif cval == 2: # Mag
        data = np.abs(data)
      elif cval == 3: # Phase
        data = np.angle(data, deg=True)

    if sval == 1: # Mag
      data = np.abs(data)
    elif sval == 2: # Sign
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
        data_max = max(abs(data_min),abs(data_max))
        data_min = -data_max
      elif zval == 3:
        data_max = 0.
      data_range = data_max-data_min
      obj.setAttr('Range Min',val=data_min)
      obj.setAttr('Range Max',val=data_max)
    else:
      data_min = 0.
      data_max = max(abs(data_min),abs(data_max))
      data_range = data_max
      obj.setAttr('Range Min',val=-data_range)
      obj.setAttr('Range Max',val=data_range)

    dmask = np.ones(data.shape)
    new_min = data_range*flor + data_min
    new_max = data_range*ceil + data_min
    data = np.minimum(np.maximum(data,new_min*dmask),new_max*dmask)

    if new_max > new_min:
      if (gamma == 1): # Put in check for gamma=1, the common use case, just to save time
        data = 255.*(data - new_min)/(new_max-new_min)
      else:
        data = 255.*pow((data - new_min)/(new_max-new_min),gamma)
    else:
      data = 255.*np.ones(data.shape)

    if sval != 2: #Not Signed Data (Pass or Mag)
      # Show based on a color map
      if cmap == 0: # Grayscale
        red = green = blue = np.uint8(data)
        alpha = 255. * np.ones(blue.shape)
      else:
        rd = np.zeros(data.shape)
        gn = np.zeros(data.shape)
        be = np.zeros(data.shape)
        zmask = np.zeros(data.shape)
        fmask = np.ones(data.shape)

        if cmap == 1: # IceFire
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          gn[hindex0] = zmask[hindex0]
          rd[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = (hue-1.)[hindex1]
          be[hindex1] = fmask[hindex1]

          gn[hindex2] = fmask[hindex2]
          rd[hindex2] = fmask[hindex2]
          be[hindex2] = (3.-hue)[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = (4.-hue)[hindex3]
          be[hindex3] = zmask[hindex3]

        elif cmap == 2: # Fire
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          rd[hindex0] = zmask[hindex0]
          gn[hindex0] = zmask[hindex0]

          be[hindex1] = (2.-hue)[hindex1]
          rd[hindex1] = (hue-1.)[hindex1]
          gn[hindex1] = zmask[hindex1]

          rd[hindex2] = fmask[hindex2]
          gn[hindex2] = (hue-2.)[hindex2]
          be[hindex2] = zmask[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = fmask[hindex3]
          be[hindex3] = (hue-3.)[hindex3]

        elif cmap == 3: # Hot
          hue = 3.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)

          rd[hindex0] = hue[hindex0]
          be[hindex0] = zmask[hindex0]
          gn[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = fmask[hindex1]
          be[hindex1] = zmask[hindex1]

          rd[hindex2] = fmask[hindex2]
          gn[hindex2] = fmask[hindex2]
          be[hindex2] = (hue-2.)[hindex2]

        if cmap == 4: # Hot2, from ASIST (http://asist.umin.jp/index-e.htm)
          rindex0 = data < 20.0
          rindex1 = np.logical_and(data >=  20.0, data <= 100.0)
          rindex2 = np.logical_and(data > 100.0, data < 128.0)
          rindex3 = np.logical_and(data >= 128.0, data <= 191.0)
          rindex4 = data > 191.0
          rd[rindex0] = data[rindex0] * 4.0
          rd[rindex1] = 80.0 - (data[rindex1] - 20.0)
          rd[rindex3] = (data[rindex3] - 128.0) * 4.0
          rd[rindex4] = data[rindex4] * 0.0 + 255.0
          rd = rd/255.0

          gindex0 = data < 45.0
          gindex1 = np.logical_and(data >= 45.0, data <= 130.0)
          gindex2 = np.logical_and(data > 130.0, data < 192.0)
          gindex3 = data >= 192.0
          gn[gindex1] = (data[gindex1] - 45.0)*3.0
          gn[gindex2] = data[gindex2] * 0.0 + 255.0
          gn[gindex3] = 252.0 - (data[gindex3] - 192.0)*4.0
          gn = gn/255.0

          bindex0 = (data < 1.0)
          bindex1 = np.logical_and(data >= 1.0, data < 86.0)
          bindex2 = np.logical_and(data >= 86.0, data <= 137.0)
          bindex3 = data > 137.0
          be[bindex1] = (data[bindex1] - 1.0)*3.0
          be[bindex2] = 255.0 - (data[bindex2] - 86.0)*5.0
          be = be/255.0

        elif cmap == 5: # BGR
          hue = 4.*(data/256.)
          hindex0 =                          hue < 1.
          hindex1 = np.logical_and(hue >= 1.,hue < 2.)
          hindex2 = np.logical_and(hue >= 2.,hue < 3.)
          hindex3 = np.logical_and(hue >= 3.,hue < 4.)

          be[hindex0] = hue[hindex0]
          gn[hindex0] = zmask[hindex0]
          rd[hindex0] = zmask[hindex0]

          gn[hindex1] = (hue-1.)[hindex1]
          rd[hindex1] = zmask[hindex1]
          be[hindex1] = fmask[hindex1]

          gn[hindex2] = fmask[hindex2]
          rd[hindex2] = (hue-2.)[hindex2]
          be[hindex2] = (3.-hue)[hindex2]

          rd[hindex3] = fmask[hindex3]
          gn[hindex3] = (4.-hue)[hindex3]
          be[hindex3] = zmask[hindex3]

        blue = np.uint8(255.*rd)
        red = np.uint8(255.*be)
        green = np.uint8(255.*gn)
        alpha = np.uint8(255.*np.ones(blue.shape))

    else: #Signed data, positive numbers green, negative numbers magenta
      red = np.zeros(data.shape)
      green = np.zeros(data.shape)
      blue = np.zeros(data.shape)
      red[sign<=0] = data[sign<=0]
      blue[sign<=0] = data[sign<=0]
      green[sign>=0] = data[sign>=0]

      red = red.astype(np.uint8)
      green = green.astype(np.uint8)
      blue = blue.astype(np.uint8)
      alpha = np.uint8(data)

  # DISPLAY RGB image
  else:

    if data.shape[-1] > 3:
      red   = data[:,:,0].astype(np.uint8)
      green = data[:,:,1].astype(np.uint8)
      blue  = data[:,:,2].astype(np.uint8)
      if(data.ndim == 3 and data.shape[-1] == 4):
          alpha = data[:,:,3].astype(np.uint8)
      else:
          alpha = 255.*np.ones(blue.shape)
    else:
        obj.log.warn("input veclen of "+str(data.shape[-1])+" is incompatible")
        return 1

  # combine data into 4D array and output from widget (not to viewport)
  h, w = red.shape[:2]
  imageTru = np.zeros((h, w, 4), dtype=np.uint8)
  imageTru[:, :, 0] = red
  imageTru[:, :, 1] = green
  imageTru[:, :, 2] = blue
  imageTru[:, :, 3] = alpha
  return imageTru

def process_slice(obj, data, dimfunc, outport, islice, jslice, kslice, i_width, j_width, k_width):

  imageTru = colorthings(obj, data, dimfunc)
  dim = list(np.shape(imageTru))

  # define colors --> I have no idea why the colors work this way but this is the correct formatting
  r = [0, 0, 255, 0]
  b = [255, 0, 0, 0]
  g = [0, 255, 0, 0]

  # define border and crosshair colors by plane; also define widths for that dimension
  if outport == 'sagittal slice':
    border = g
    hori_crosshair = b
    vert_crosshair = r
    jwidth = i_width
    iwidth = j_width
    icenter = islice
    jcenter = jslice
  elif outport == 'coronal slice':
    border = r
    hori_crosshair = b
    vert_crosshair = g
    jwidth = i_width
    iwidth = k_width
    icenter = islice
    jcenter = kslice
  elif outport == 'transverse slice':
    border = b
    hori_crosshair = r
    vert_crosshair = g
    iwidth = j_width
    jwidth = k_width
    icenter = jslice
    jcenter = kslice
    
  # add border
  img = np.stack([np.pad(imageTru[:,:,c], ([2, 2]), mode='constant', constant_values=border[c]) for c in range(4)], axis=2)

  # add crosshairs
  vert1 = jcenter - ((iwidth - 1)//2)
  vert2 = jcenter + (iwidth//2)
  hori1 = icenter - ((jwidth - 1)//2)
  hori2 = icenter + (jwidth//2)
  print("iwidth: " + str(iwidth))
  print("jwidth: " + str(jwidth))
  if iwidth == 1 and jwidth == 1:
    for i in range(2, dim[0]-1):
      if i < (icenter-5) or i > (icenter+5):
        img[i, vert1] = vert_crosshair
    for j in range(2, dim[1]-1):
      if j <(jcenter-5) or j > (jcenter+5):
        img[hori1, j] = hori_crosshair
  else:
    for i in range(2, dim[0]-1):
      img[i, vert1] = vert_crosshair
      img[i, vert2] = vert_crosshair
    for j in range(2, dim[1]-1):
      img[hori1, j] = hori_crosshair
      img[hori2, j] = hori_crosshair
  return img, imageTru

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
  """2D image viewer for real or complex NPYarrays.

  INPUT:
  2D data, real or complex
  3D uint8 ARGB data (e.g. output of another ImageDisplay node)

  OUTPUT:
  3D data of displayed image, last dimension has length 4 for ARGB byte (uint8) data

  WIDGETS:
  Complex Display - If data are complex, allows you to show Real, Imaginary, Magnitude, Phase, or "Complex" data.
      If C (Complex) is chosen, then pixel brightness reflects value magnitude, while pixel color reflects value phase.
      If input data are real-valued, this widget is hidden

  Color Map - Chooses from a number of colormaps for real-valued data.  Not available if Scalar Display is "Sign"

  Edge Pixels - only visible for complex input data with Complex Display set to "C"
      Setting this to N creates an N-pixel color ring around the image border illustrating the phase-to-color mapping

  Black Pixels - only visible for complex input data with Complex Display set to "C"
      Setting this to N creates an N-pixel black ring around the image border (but inside the Edge Pixels ring) to
      separate the Edge pixel ring from the actual data image

  Viewport - displays the image
    Double clicking on Viewport area brings up a scaling widget to change the image size, and change graphic overlay

  L W F C - (hidden by default - double click on widget area to show sliders)
            Adjust value-to-pixel brightness mapping using Level/Window or Floor/Ceiling

  Scalar Display - visible for real data, or complex data with "Complex Display" set to R, I, M, or P
    Pass uses the real data in the data-to-pixel mapping
    Mag uses the magnitude data in the data-to-pixel mapping (i.e. affects negative values only)
    Sign will display positive values in green, and the absolutele value of negative values in magenta

  Gamma - changes gamma of display function.  Default value of 1 gives linear mapping of data to pixel value
    pixel values refect value of data^gamma

  Zero Ref - visible for real data, or complex data with "Complex Display" set to R, I, M, or P
              also invisible if "Scalar Display" set to Sign
    This is used for the data-to-pixel value mapping
    --- maps the smallest value to black, and the largest value to white
    0-> maps zero to black, and the largest value to white.  All negative numbers are black
    -0- maps zero to middle gray, with the largest magnitude set to black (if negative) or white (if positive)
    <-0 maps zero to white, and the most negative value to black.  All positive numbers are white

  Fix Range
    If Auto-Range On, the data range for pixel value mapping is rescaled whenever new data appears at input
    If Fixed-Range On, the data range is fixed (and can be changed using Range Min and Range Max)

  Range Min - shows minimum data value used for mapping to pixel values
    This value can be changed if (and only if) "Fix Range" is set to "Fixed-Ranged On"

  Range Max - shows maximum data value used for mapping to pixel values
    This value can be changed if (and only if) "Fix Range" is set to "Fixed-Ranged On"
  """

  def execType(self):
    return gpi.GPI_PROCESS
  
  def initUI(self):

    # Widgets
    self.addWidget('ExclusivePushButtons','Complex Display',
                    buttons=['R','I','M','P','C'], val=4)
    self.real_cmaps = ['Gray','IceFire','Fire','Hot','HOT2','BGR']
    self.complex_cmaps = ['HSV','HSL','HUSL','CoolWarm']
    self.addWidget('ExclusivePushButtons','Color Map',
                    buttons=self.real_cmaps, val=0, collapsed=True)
    self.addWidget('SpinBox', 'Edge Pixels', min=0)
    self.addWidget('SpinBox', 'Black Pixels', min=0)
    self.addWidget('DisplayBox', 'Viewport:')
    self.addWidget('Slider', 'Axial Slice (Blue)', val=1, min=0, max=40)  # only initial values
    self.addWidget('Slider', 'Coronal Slice (Red)', val=1, min=0, max=40)
    self.addWidget('Slider', 'Sagittal Slice (Green)', val=1, min=0, max=40)
    self.addWidget('SpinBox', '# Columns', val=1)
    self.addWidget('SpinBox', '# Rows', val=1)
    self.addWidget('WindowLevel', 'L W F C:', collapsed=True)
    self.addWidget('ExclusivePushButtons','Scalar Display',
                    buttons=['Pass','Mag','Sign'], val=0)
    self.addWidget('DoubleSpinBox', 'Gamma',min=0.1,max=10,val=1,singlestep=0.05,decimals=3)
    self.addWidget('ExclusivePushButtons','Zero Ref',
                    buttons=['---','0->','-0-','<-0'], val=0)
    self.addWidget('PushButton', 'Fix Range', button_title='Auto-Range On', toggle=True)
    self.addWidget('DoubleSpinBox', 'Range Min')
    self.addWidget('DoubleSpinBox', 'Range Max')
    self.addWidget('SpinBox', 'i width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'j width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'k width', min=0, max=40, val=1)
    self.addWidget('SpinBox', 'volume', min=0, max=1, val=0)

    # IO Ports
    self.addInPort('in', 'NPYarray', drange=(2,3))
    self.addOutPort('out', 'NPYarray')
    self.addOutPort('sagittal slice', 'NPYarray')
    self.addOutPort('coronal slice', 'NPYarray')
    self.addOutPort('transverse slice', 'NPYarray')

  def validate(self):
    data = self.getData('in')
    if np.iscomplexobj(data):
      self.setAttr('Complex Display',visible=True)
      scalarvis = self.getVal('Complex Display') != 4
    else:
      self.setAttr('Complex Display',visible=False)
      scalarvis = True        
    self.setAttr('Scalar Display',visible=scalarvis)
    self.setAttr('Edge Pixels',visible=not scalarvis)
    self.setAttr('Black Pixels',visible=not scalarvis)
    self.setAttr('Zero Ref',visible=False)
    self.setAttr('# Rows', visible=False)
    self.setAttr('# Columns', visible=False)
    self.setAttr('i width', visible=False)
    self.setAttr('j width', visible=False)
    self.setAttr('k width', visible=False)
    if self.getVal('Fix Range'):
      self.setAttr('Fix Range',button_title="Fixed Range On")
    else:
      self.setAttr('Fix Range',button_title="Auto-Range On")
    return 0

  def compute(self):

    # events = self.getEvents()
    # print(str(events['_WDG_EVENT_']))

    from matplotlib import cm

    # get and format data
    data3d = self.getData('in').astype(np.float64)
    dim = list(np.shape(data3d))
    save = False

    # slicer and volume mode here
    line = self.getAttr('Viewport:', 'line')
    if line:
      red_dim = [dim[0], dim[2]]
      i0, j0 = line[0]
      i1, j1 = line[1]
      # is there an endpoint in the control box?
      if ((j0 > (red_dim[0]-1)) or (j1 > (red_dim[0]-1))) and ((i0 < (red_dim[1]-1)) or (i1 < (red_dim[1]-1))):
        if (j0 > (red_dim[0]-1)) and (j1 > (red_dim[0]-1)) and (i0 < (red_dim[1]-1)) and (i1 < (red_dim[1]-1)):
          save = True
        else:
          self.setAttr('volume', val=0)
          self.setAttr('i width', val=1)
          self.setAttr('j width', val=1)
          self.setAttr('k width', val=1)
          if (j0 > (red_dim[0]-1)) and (i0 < (red_dim[1]-1)):                 # first point is in control box
            ii = i1
            jj = j1
          else:                                                               # second point is in control box
            ii = i0
            jj = j0
      else:                       # volume mode
        self.setAttr('volume', val=1)
        # calculate centers
        ii = (i0 + i1)//2
        jj = (j0 + j1)//2
        # calculate widths
        i_width = abs(i0 - i1) + 1
        j_width = abs(j0 - j1) + 1
        # check to make sure that both endpoints are in the same zone
        if i0 > (red_dim[1]+3) and  j0 > (red_dim[0]+3) and i1 > (red_dim[1]+3) and  j1 > (red_dim[0]+3):
          self.setAttr('j width', val=i_width)
          self.setAttr('k width', val=j_width)
        elif i0 > (red_dim[1]+3) and  i1 > (red_dim[1]+3) and j0 < (red_dim[0]+3) and j1 < (red_dim[1]+3):
          self.setAttr('i width', val=j_width)
          self.setAttr('j width', val=i_width)
        elif i0 < (red_dim[1]+3) and  i1 < (red_dim[1]+3) and j0 < (red_dim[0]+3) and j1 < (red_dim[1]+3):
          self.setAttr('i width', val=j_width)
          self.setAttr('k width', val=i_width)
        else:
          self.log.error("You cannot draw a box in two different slices!")
      # update centers based on which zone we are in
      if save is not True:
        if (ii > (red_dim[1]+3)) and  (jj > (red_dim[0]+3)):
          ii -= (red_dim[1]+4)
          jj -= (red_dim[0]+4)
          self.setAttr('Coronal Slice (Red)', val=jj)
          self.setAttr('Sagittal Slice (Green)', val=ii)
        elif ii > (red_dim[1]+3):
          ii -= (red_dim[1]+4)
          self.setAttr('Axial Slice (Blue)', val=jj)
          self.setAttr('Coronal Slice (Red)', val=ii)
        else:
          self.setAttr('Axial Slice (Blue)', val=jj)
          self.setAttr('Sagittal Slice (Green)', val=ii)

    # reset default values with the values from the dimensions of the input data
    self.setAttr('Axial Slice (Blue)', max=dim[0]-1)
    self.setAttr('Coronal Slice (Red)', max=dim[1]-1)
    self.setAttr('Sagittal Slice (Green)', max=dim[2]-1)
    self.setAttr('i width', max=dim[0]-1)
    self.setAttr('j width', max=dim[1]-1)
    self.setAttr('k width', max=dim[2]-1)
    islice = self.getVal('Axial Slice (Blue)')
    jslice = self.getVal('Coronal Slice (Red)')
    kslice = self.getVal('Sagittal Slice (Green)')
    iwidth = self.getVal('i width')
    jwidth = self.getVal('j width')
    kwidth = self.getVal('k width')
    axialSlice = data3d[islice, :, :] 
    coronalSlice = data3d[:, jslice, :]
    sagittalSlice = data3d[:, :, kslice]
    maximum = np.max([np.max(sagittalSlice), np.max(coronalSlice), np.max(axialSlice)])
    minimum = np.min([np.min(sagittalSlice), np.min(coronalSlice), np.min(axialSlice)])
    self.setAttr('Range Min', val=minimum)
    self.setAttr('Range Max', val=maximum)

    # get and process slices
    greenRegion, sagittalSliceImg = process_slice(self, sagittalSlice, 0, 'sagittal slice', islice, jslice, kslice, iwidth, jwidth, kwidth)
    redRegion, coronalSliceImg = process_slice(self, coronalSlice, 0, 'coronal slice', islice, jslice, kslice, iwidth, jwidth, kwidth)
    blueRegion, axialSliceImg = process_slice(self, axialSlice, 0, 'transverse slice', islice, jslice, kslice, iwidth, jwidth, kwidth)

    # combine all slices into one image
    combine1 = np.append(redRegion, greenRegion, axis=1)
    pad = np.zeros([dim[1]+4, dim[1]+4, 4], dtype=greenRegion.dtype)
    combine2 = np.append(pad, blueRegion, axis=1)
    img = np.append(combine1, combine2, axis=0)

    # format and display image in Viewport
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    alpha = img[:, :, 3]
    h, w = red.shape[:2]
    image1 = np.zeros((h, w, 4), dtype=np.uint8)
    image1[:, :, 0] = red
    image1[:, :, 1] = green
    image1[:, :, 2] = blue
    image1[:, :, 3] = alpha
    format_ = QtGui.QImage.Format_RGB32
    image = QtGui.QImage(image1.data, w, h, format_)
    if image.isNull():
      self.log.warn("Image Viewer: cannot load image")
    self.setAttr('Viewport:', val=image)

    # output volume and slices
    if save is True or True:                            # override to always save for now
      zz = int(self.getVal('Sagittal Slice (Green)'))
      yy = int(self.getVal('Coronal Slice (Red)'))
      xx = int(self.getVal('Axial Slice (Blue)'))
      print("xx: " + str(xx))
      print("yy: " + str(yy))
      print("zz: " + str(zz))
      # retrieve volume
      if int(self.getVal('volume')) == 1:
        # get widths
        x_width = int(self.getVal('i width'))
        y_width = int(self.getVal('j width'))
        z_width = int(self.getVal('k width'))
        print("x width: " + str(x_width))
        print("y width: " + str(y_width))
        print("z width: " + str(z_width))
        # calculate ranges for x, y & z
        x0 = xx - ((x_width-1)//2)
        x1 = xx + (x_width//2)
        y0 = yy - ((y_width-1)//2)
        y1 = yy + (y_width//2)
        z0 = zz - ((z_width-1)//2)
        z1 = zz + (z_width//2)
        print("x0: " + str(x0))
        print("x1: " + str(x1))
        print("y0: " + str(y0))
        print("y1: " + str(y1))
        print("z0: " + str(z0))
        print("z1: " + str(z1))
        # output array
        if x0 == x1: x1 += 1
        elif y0 == y1: y1 += 1
        elif z0 == z1: z1 += 1
        if x_width != 1: x_width = x1 - x0
        if y_width != 1: y_width = y1 - y0
        if z_width != 1: z_width = z1 - z0
        outArray = np.reshape(np.array(data3d[x0:x1, y0:y1, z0:z1]), (x_width, y_width, z_width))
      else:                                 # added this line because it will be faster to hardcode in values for this case since they are already known
        outArray = np.reshape(np.array(data3d[xx][yy][zz]), (1, 1, 1))
      self.setData('out', outArray)
      self.setData('sagittal slice', sagittalSliceImg)
      self.setData('coronal slice', coronalSliceImg)
      self.setData('transverse slice', axialSliceImg)
    return 0th != 1: z_width = z1 - z0
        outArray = np.reshape(np.array(data3d[x0:x1, y0:y1, z0:z1]), (x_width, y_width, z_width))
      else:                                 # added this line because it will be faster to hardcode in values for this case since they are already known
        outArray = np.reshape(np.array(data3d[xx][yy][zz]), (1, 1, 1))
      self.setData('out', outArray)
      self.setData('sagittal slice', sagittalSliceImg)
      self.setData('coronal slice', coronalSliceImg)
      self.setData('transverse slice', axialSliceImg)
    return 0