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
    self.sliderThreshold = Scale(self.main, from_=1, to=50, length=600, resolution=0.05,tickinterval=3, orient=HORIZONTAL, command=self.sliderEvent)
    self.sliderThreshold.set(13)
    self.sliderThreshold.grid(row=1, column = 0)
    

    
    ## Jump Slider
    self.sliderJump = Scale(self.main, from_=1, to=20, length=600, resolution=1,tickinterval=3, orient=HORIZONTAL, command=self.sliderEvent)
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
      self.resetSelection()
      self.updateImg()
      return
    # waehle die Zusammenhangskomponente aus
    self.selection = (self.labels == self.labels[self.y,self.x])
    self.showSelection = True
  
    # bestimme das Skelett
    self.skeleton = medial_axis(self.selection)
    print self.skeleton
    
    # Bestimme den Zellkern (naehchstgelegener Punkt des Skeletts zur Klickposition
    minDist = 100000
    sparsePath = sparse.coo_matrix(self.skeleton)   
    for (i,j) in zip(sparsePath.row, sparsePath.col):
      dist = (i-self.y)^2 + (j-self.x)^2
      if dist < minDist:
        nearest = (i,j)
        minDist = dist
  
    self.cellkernel = nearest ## Von hier aus soll die laenge gemessen werden.
    
    #### Do a BFS in skeleton starting at cellkernel
    self.bestChildren = {}
    self.path = np.zeros(self.imageSize)
    self.notVisited = self.skeleton.copy()
    
    def visit(pos, parent):
      print pos
      self.notVisited[pos] = False
      maxChildLength = 0
      bestChild = False
      for i in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]: # alle 8 nachbarn
        child = (pos[0] + i[0], pos[1] + i[1])
        if child != parent:
          if self.notVisited[child[0]][child[1]] == True:
            childLength = visit(child, pos)
            if childLength > maxChildLength:
              maxChildLength = childLength
              bestChild = child
      if bestChild:
        print 'bestChild', bestChild, 'with length:', maxChildLength
        self.bestChildren[pos] = bestChild
      return maxChildLength + 1
    
    self.pathLength = visit(self.cellkernel, 0)


    pos = self.cellkernel
    while(self.bestChildren.has_key(pos)):
      self.path[pos] = True
      pos = self.bestChildren[pos]

    print 'pathlength:', self.pathLength    
  #  q = Queue()
  #  q.put(self.cellkernel, 0, 0)
  #  
  #  while (q.full()):
 #     pos, parent, depth = q.get()
      
  
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
    if self.showSelection:
      bigPath = dilation(self.path, disk(2))
      a[:,:,0] = np.minimum(maskScaled + bigPath,255)
      a[:,:,1] = np.minimum(255*self.selection + bigPath,255)
      a[:,:,2] = np.minimum(np.maximum(maskScaled - 255* self.selection, 0) + bigPath, 255)
      
   #   a[self.cellkernel[0],self.cellkernel[1],0] = 100 ## Cellkernel einzeichnen
    else:
      a[:,:,0] = maskScaled
      a[:,:,1] = 0
      a[:,:,2] = maskScaled
      
    maskImg = Image.fromarray(a) 

    maskResize = maskImg.resize(self.displaySize)
    self.maskTk = ImageTk.PhotoImage(maskResize)
    self.maskWid.itemconfig(self.mask_on_canvas, image = self.maskTk)

  ################### Auswahl aufheben ##################
  def resetSelection(self):
    self.selection = np.zeros(self.imageSize)
    self.showSelection = False

instance = Axonium()


