from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
from scipy import sparse
from skimage.morphology import medial_axis
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import label
import timeit

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
    self.sliderThreshold = Scale(self.main, from_=0, to=50, length=600, resolution=0.1,tickinterval=5, orient=HORIZONTAL)
    self.sliderThreshold.bind("<ButtonRelease-1>", self.sliderThresholdEvent)
    self.sliderThreshold.set(13)
    self.sliderThreshold.grid(row=1, column = 0)
    

    
    ## Jump Slider
    self.sliderJump = Scale(self.main, from_=1, to=20, length=600, resolution=1,tickinterval=3, orient=HORIZONTAL)
    self.sliderJump.bind("<ButtonRelease-1>", self.sliderJumpEvent)
    self.sliderJump.set(10)
    self.sliderJump.grid(row=1, column = 1)
    
    ## list box
    self.scrollbar = Scrollbar(self.main, orient=VERTICAL)        
    self.scrollbar.grid(row = 0, column = 3, sticky=N+S)
    self.listbox = Listbox (self.main, height = 25, width = 10, yscrollcommand = self.scrollbar.set)
    self.listbox.grid(row = 0, column = 2,sticky=N+S)
    self.scrollbar.config(command=self.listbox.yview)
    self.listbox.bind('<<ListboxSelect>>', self.buttonToDeleteMode)
    
    ## Button der Laenge des aktuellen Pfades anzeigt und bei Klick diesen in diese in die Liste eintraegt (rechts unten)
    self.buttonLength = Button(self.main, text="0", state=DISABLED, command=self.insertLength)
    self.buttonLength.grid(row = 1, column = 2, columnspan=2, sticky=E+W)

    
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
    start = timeit.default_timer()
      
    if self.labels[self.y,self.x] == -1: # Ist der Hintergrund angeklickt worden?
      self.resetSelection()
      self.updateImg()
      return
    
    stop0 = timeit.default_timer()
    print('updateMask total: ', stop0 - start, ' sec')
       
    # waehle die Zusammenhangskomponente aus
    self.selection = (self.labels == self.labels[self.y,self.x])
    self.showSelection = True
  
    # bestimme das Skelett
    stop1 = timeit.default_timer()
    print('selection: ', stop1 - stop0, ' sec')
    self.skeleton = medial_axis(self.selection, mask = self.selection)
  #  print(self.skeleton)
    stop2 = timeit.default_timer()
    print('skeleton: ', stop2 - stop1, ' sec')
    # Bestimme den Zellkern (naehchstgelegener Punkt des Skeletts zur Klickposition
    minDist = 100000
    sparsePath = sparse.coo_matrix(self.skeleton)   
    for (i,j) in zip(sparsePath.row, sparsePath.col):
      dist = (i-self.y)*(i-self.y) + (j-self.x)*(j-self.x)
      if dist < minDist:
        nearest = (i,j)
        minDist = dist
        

    
    self.cellkernel = nearest ## Von hier aus soll die laenge gemessen werden.
    print('kern: ', self.cellkernel, ' Abstand zum Mausklick: ', minDist)
    
    stop3 = timeit.default_timer()
    print('find kernel: ', stop3 - stop2, ' sec')  
    
    #### Do a BFS in skeleton starting at cellkernel
    self.bestChildren = {}
    self.path = np.zeros(self.imageSize)
    notVisited = self.skeleton.copy()
    
    def visit(pos, parent):
      #print(pos)
      notVisited[pos] = False
      maxChildLength = 0
      bestChild = False
      
      for i in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]: # alle 8 nachbarn
        child = (pos[0] + i[0], pos[1] + i[1])
        if child[0] < 0 or child[0] >= self.imageSize[0] or child[1] < 0 or child[1] >= self.imageSize[1]: #Child liegt ausserhalb des Bildes
          continue
        if child != parent:
          if notVisited[child[0]][child[1]] == True:
            childLength = visit(child, pos)
            if childLength > maxChildLength:
              maxChildLength = childLength
              bestChild = child
      if bestChild:
       # print('bestChild', bestChild, 'with length:', maxChildLength)
        self.bestChildren[pos] = bestChild
      return maxChildLength + 1
    
    self.pathLength = visit(self.cellkernel, 0)

    stop4 = timeit.default_timer()
    print('do dfs: ', stop4 - stop3, ' sec')  

    pos = self.cellkernel
    while(pos in self.bestChildren.keys()):
      self.path[pos] = True
      pos = self.bestChildren[pos]
    print('pathlength:', self.pathLength)  
    
  #  self.updateLength()
    self.buttonToInsertMode()

    
    stop5 = timeit.default_timer()
    print('calc path: ', stop5 - stop4, ' sec')  

  #  q = Queue()
  #  q.put(self.cellkernel, 0, 0)
  #  
  #  while (q.full()):
 #     pos, parent, depth = q.get()
      

    self.updateImg()
    
    
  
    ############## Slider-Events ###############
  def sliderThresholdEvent(self, event):
    print ('ThresholdSlider Released')
    self.resetSelection()
    self.updateMask()
    self.updateImg()
    
  def sliderJumpEvent(self, event):
    print ('JumpSlider Released')
    self.updateMask()
    if self.showSelection: # falls gerade eine Selection aktiv ist (geklickt wurde)
      self.click() #updateImg included
      
      
   ############ ButtonLength event ##################
  def insertLength(self):
    self.buttonLength.config(state = DISABLED)
    if self.deleteMode == False:
      self.listbox.insert(END, self.pathLength)
    else:
      self.listbox.delete(ANCHOR)
      self.buttonToInsertMode()
      
  def buttonToInsertMode(self):
    self.buttonLength.config(text= str(self.pathLength)+  'px', state = NORMAL)
    self.deleteMode = False
    
  def buttonToDeleteMode(self, event):
    self.buttonLength.config(text= 'Löschen', state = NORMAL)
    self.deleteMode = True    
 
  ############### Mask sowie die Labels werden aktualisiert ############
  def updateMask(self):
    start = timeit.default_timer()
    ## erstelle neue Mask
    self.mask = (self.monochrom  >= self.sliderThreshold.get())

    ## erstelle Labels neu. Dafuer wird bluredMask aktualisiert.
    selem = disk(self.sliderJump.get())
    stop1 = timeit.default_timer()
    print('init updateMask: ', stop1 - start, ' sec')  
    bluredMask = dilation(self.mask, selem)
    stop2 = timeit.default_timer()
    print('bluring Mask: ', stop2 - stop1, ' sec')  
    self.labels = label(bluredMask, neighbors=8, background = 0)
    stop3 = timeit.default_timer()
    print('find components: ', stop3 - stop2, ' sec')  

  ################### Bild aktualisieren ###############
  def updateImg(self):

    maskScaled = 100* self.mask
    print('before resize')    
    a = np.asarray(self.image).copy()    
    print('middle')
    if self.showSelection:
      bigPath = dilation(self.path, disk(3))
      print('after resize')  
      a[:,:,0] = np.minimum(maskScaled + bigPath,255)
      a[:,:,1] = np.minimum(50*self.selection + bigPath,255)
      a[:,:,2] = np.minimum(np.maximum(maskScaled - 50* self.selection, 0) + bigPath, 255)
   
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
    self.buttonLength.config(text='Kein Weg', state = DISABLED)
 

instance = Axonium()


