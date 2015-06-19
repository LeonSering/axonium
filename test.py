import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image

SCHWELLWERT_START = 10

WHITE = (255,255,255)

def update(val):
  global schwellwert
  schwellwert = SchwellwertSlider.val
  print schwellwert
  updateMask()
  updateImg()

def updateMask():
  global mask
  mask = s >= schwellwert
  mask = 255 *mask
  print mask


schwellwert = SCHWELLWERT_START

im = Image.open('image0083.tif') #Can be many different formats.

a = np.asarray(im).copy()
r = a[:,:,0]
g = a[:,:,1]
b = a[:,:,2]

s = r + g + b


plt.figure(1)
plt.subplot(121)


plt.imshow(im)
plt.axis('off') # clear x- and y-axes

plt.subplot(122)
mask = s > schwellwert
mask = 255 * mask
a[:,:,0] = mask
a[:,:,1] = mask
a[:,:,2] = mask


out = Image.fromarray(a)

plt.imshow(out)
plt.axis('off') # clear x- and y-axes



def updateImg():
  plt.subplot(122)
  a[:,:,0] = mask
  a[:,:,1] = mask
  a[:,:,2] = mask

  out = Image.fromarray(a)
  
  plt.imshow(out)
  plt.axis('off') # clear x- and y-axes
  plt.draw()



axcolor = 'lightgoldenrodyellow'
SchwellwertSliderAxes = plt.axes([0, 0.1, 0.65, 0.03], axisbg=axcolor)
SchwellwertSlider = Slider(SchwellwertSliderAxes, 'Schwellwert', 0, 40, valinit=SCHWELLWERT_START)


  
SchwellwertSlider.on_changed(update)

plt.show()
