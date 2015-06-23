from Tkinter import *
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
from scipy import sparse
from skimage.morphology import medial_axis
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import label


#np.set_printoptions(threshold=np.nan) # gesamte Matrix anschauen

class Axonium:
  
  def __init__(self):
    self.main = Tk()
    self.image = Image.open('image0083.tif')
 
    ## Anzeigegroesse und Skalierungsfaktor.
    self.imageSize = [1040, 1392] # erst y, dann x
    self.skalierungsfaktor = 0.4
    self.displaySize = [int(self.skalierungsfaktor* self.imageSize[1]), int(self.skalierungsfaktor * self.imageSize[0])] # erst x, dann y
 
 
    ## Das Bild auf einen Wert vereinfachen.
    a = np.asarray(self.image).copy()
    r = a[:,:,0]
    g = a[:,:,1]
    b = a[:,:,2]
    self.monochrom = r + g + b #das Bild mit nurnoch einem Wert pro Pixel


    ## Das original Bild
    imageResize = self.image.resize(self.displaySize)
    imageTk = ImageTk.PhotoImage(imageResize)
    self.imageWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.imageWid.grid(row=0,column=0)
    self.image_on_canvas = self.imageWid.create_image(0,0,anchor=NW, image=imageTk)
    self.imageWid.tag_bind(self.image_on_canvas, '<ButtonPress-1>', self.image_click)
    
    ## Das rechte Bild (MASK)
    self.maskWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.maskWid.grid(row=0,column=1)
    self.mask_on_canvas = self.maskWid.create_image(0,0,anchor=NW, image=imageTk)
    self.maskWid.tag_bind(self.mask_on_canvas, '<ButtonPress-1>', self.mask_click)
    

    ## Schwellwert Slider
    self.sliderThreshold = Scale(self.main, from_=0, to=50, length=600, resolution=0.05,tickinterval=5, orient=HORIZONTAL, command=self.sliderEvent)
    self.sliderThreshold.set(13)
    self.sliderThreshold.grid(row=1, column = 0)
    

    
    ## Jump Slider
    self.sliderJump = Scale(self.main, from_=0, to=20, length=600, resolution=1,tickinterval=5, orient=HORIZONTAL, command=self.sliderEvent)
    self.sliderJump.set(10)
    self.sliderJump.grid(row=1, column = 1)
    
    ## aktuelle Auswahl.
    self.x = 0
    self.y = 0  
    self.resetSelection()
    
    self.updateMask()
    self.updateImg()
        
    self.main.mainloop()
    
    
    
    
    
  ########### Mouse Clicks ##################  
  def mask_click(self, event):
    x, y = int(event.x / self.skalierungsfaktor), int(event.y / self.skalierungsfaktor)
    print('MaskClick: x: ', x, ' y: ', y, ' label: ', self.labels[y,x])
    self.x = x
    self.y = y
    self.click()

  def image_click(self, event):
    x, y = int(event.x / self.skalierungsfaktor), int(event.y / self.skalierungsfaktor)
    print('ImageClick: x: ', x, ' y: ', y, ' label: ', self.labels[y,x])
    self.x = x
    self.y = y
    self.click()
    
  def click(self):
    if self.sliderTouched == True:
      self.updateMask()
      self.sliderTouched = False
      
    if self.labels[self.y,self.x] == -1: # Ist der Hintergrund angeklickt worden?
      self.selection = 0 * self.monochrom
    else: # waehle die Zusammenhangskomponente aus
      self.selection = (self.labels == self.labels[self.y,self.x])
  
  
    self.path = medial_axis(self.selection)
    
    minDist = 100000
    sparsePath = sparse.coo_matrix(self.path)   
    for (i,j) in zip(sparsePath.row, sparsePath.col):
      dist = (i-self.y)^2 + (j-self.x)^2
      if dist < minDist:
        nearest = (i,j)
        minDist = dist
  
    self.cellkernel = nearest ## Von hier aus soll die laenge gemessen werden.
    
    #### TO CONTINURE
  
    self.updateImg()
    
    
  
    ############## Slider-Events ###############
  def sliderEvent(self, event):
    self.sliderTouched = True
 
  ############### Mask sowie die Labels werden aktualisiert ############
  def updateMask(self):
    ## erstelle neue Mask
    self.mask = (self.monochrom  >= self.sliderThreshold.get())

    ## erstelle Labels neu. Dafuer wird bluredMask aktualisiert.
    selem = disk(self.sliderJump.get())
    bluredMask = dilation(self.mask, selem)
    self.labels = label(bluredMask, neighbors=8, background = 0)


  ################### Bild aktualisieren ###############
  def updateImg(self):
    maskScaled = 255* self.mask
    
    a = np.asarray(self.image).copy()
    a[:,:,0] = maskScaled
    a[:,:,1] = 255*self.selection
    a[:,:,2] = np.maximum(maskScaled - 255* self.selection, 0)
    maskImg = Image.fromarray(a) 

    maskResize = maskImg.resize(self.displaySize)
    self.maskTk = ImageTk.PhotoImage(maskResize)
    self.maskWid.itemconfig(self.mask_on_canvas, image = self.maskTk)

  ################### Auswahl aufheben ##################
  def resetSelection(self):
    self.selection = np.zeros(self.imageSize)

instance = Axonium()


