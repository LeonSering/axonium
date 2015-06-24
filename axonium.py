import queue
import os
import pickle
from tkinter import *
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
from scipy import sparse
from skimage.morphology import medial_axis
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import label
#import timeit
import datetime


#np.set_printoptions(threshold=np.nan) # gesamte Matrix anschauen

class Axonium:
  
  def __init__(self):
    self.main = Tk()

 
    ## Anzeigegroesse und Skalierungsfaktor.
    self.imageSize = [1040, 1392] # erst y, dann x
    self.skalierungsfaktor = 0.4
    self.displaySize = [int(self.skalierungsfaktor* self.imageSize[1]), int(self.skalierungsfaktor * self.imageSize[0])] # erst x, dann y
 


    ## Das original Bild
    self.imageWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.imageWid.grid(row=2,column=2)
    self.image_on_canvas = self.imageWid.create_image(0,0,anchor=NW)
    self.imageWid.tag_bind(self.image_on_canvas, '<ButtonPress-1>', self.image_click)
    
    ## Das rechte Bild (MASK)
    self.maskWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.maskWid.grid(row=2,column=3)
    self.mask_on_canvas = self.maskWid.create_image(0,0,anchor=NW)
    self.maskWid.tag_bind(self.mask_on_canvas, '<ButtonPress-1>', self.mask_click)
    

    ## Schwellwert Slider
    self.sliderThreshold = Scale(self.main, from_=0, to=50, length=500, resolution=0.1,tickinterval=5, orient=HORIZONTAL)
    self.sliderThreshold.bind("<ButtonRelease-1>", self.sliderThresholdEvent)
    self.sliderThreshold.set(13)
    self.sliderThreshold.grid(row=3, column = 2)
    

    
    ## Jump Slider
    self.sliderJump = Scale(self.main, from_=1, to=20, length=500, resolution=1,tickinterval=3, orient=HORIZONTAL)
    self.sliderJump.bind("<ButtonRelease-1>", self.sliderJumpEvent)
    self.sliderJump.set(10)
    self.sliderJump.grid(row=3, column = 3)
    
    ## list box for length
    self.scrollbarLength = Scrollbar(self.main, orient=VERTICAL)        
    self.scrollbarLength.grid(row = 2, column = 5, sticky=N+S)
    self.listboxLength = Listbox (self.main, height = 10, width = 10, yscrollcommand = self.scrollbarLength.set)
    self.listboxLength.grid(row = 2, column = 4,sticky=E+N+S)
    self.scrollbarLength.config(command=self.listboxLength.yview)
    self.listboxLength.bind('<<ListboxSelect>>', self.buttonToDeleteMode)
    
    ## Button der Laenge des aktuellen Pfades anzeigt und bei Klick diesen in diese in die Liste eintraegt (rechts unten)
    self.buttonLength = Button(self.main, text="0", state=DISABLED, command=self.insertLength)
    self.buttonLength.grid(row = 3, column = 4, columnspan=2, sticky=E+W)
    
    ## filelist
    self.scrollbarFiles = Scrollbar(self.main, orient=VERTICAL)        
    self.scrollbarFiles.grid(row = 2, column = 1, sticky=N+S)
    self.listboxFiles = Listbox (self.main, height = 10, width = 12, yscrollcommand = self.scrollbarFiles.set)
    self.listboxFiles.grid(row = 2, column = 0,sticky=E+N+S)
    self.scrollbarFiles.config(command=self.listboxFiles.yview)
    self.listboxFiles.bind('<<ListboxSelect>>', self.loadFile)
    
    ## Open File Button
    self.buttonOpenFile = Button(self.main, text="Öffne Ordner", command=self.selectFolder)
    self.buttonOpenFile.grid(row = 3, column = 0, columnspan = 2, sticky=E+W)

    
    ## aktuelle Auswahl.
    self.x = 0
    self.y = 0
    
    self.folderPath = ''
    os.mkdir("backup")
    self.initiateImage('') #load a black image
    
    self.loadStatus()
    

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(timestamp)
    
    self.backupName = "backup/"+timestamp+".txt"

      
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
    if self.labels[self.y,self.x] == -1: # Ist der Hintergrund angeklickt worden?
      self.resetSelection()
      self.updateImg()
      return
       
    # waehle die Zusammenhangskomponente aus
    self.selection = (self.labels == self.labels[self.y,self.x])
    self.showSelection = True
  
    # bestimme das Skelett
    self.skeleton = medial_axis(self.selection, mask = self.selection)
  #  print(self.skeleton)
    # Bestimme den Zellkern (naehchstgelegener Punkt des Skeletts zur Klickposition
    minDist = 100000
    sparsePath = sparse.coo_matrix(self.skeleton)   
    for (i,j) in zip(sparsePath.row, sparsePath.col):
      dist = (i-self.y)*(i-self.y) + (j-self.x)*(j-self.x)
      if dist < minDist:
        nearest = (i,j)
        minDist = dist
        

    
    self.cellkernel = nearest ## Von hier aus soll die laenge gemessen werden.
    print('Kern: ', self.cellkernel, ' Abstand zum Mausklick: ', minDist)
    
    #### Do a BFS in skeleton starting at cellkernel
    self.parents = {self.cellkernel:0}
    self.path = np.zeros(self.imageSize)
    notTouched = self.skeleton.copy()
    
    BFSQueue = queue.Queue()
    
    BFSQueue.put((self.cellkernel, 0, 0))
    
    maxLevel = 0
    bestEnd = self.cellkernel
    
    while(not BFSQueue.empty()):
      pos, parent, level = BFSQueue.get()
      if level > maxLevel:
        maxLevel = level
        bestEnd = pos
      
      for i in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]: # alle 8 nachbarn
        child = (pos[0] + i[0], pos[1] + i[1])
        if child[0] < 0 or child[0] >= self.imageSize[0] or child[1] < 0 or child[1] >= self.imageSize[1]: #Child liegt ausserhalb des Bildes
          continue
        if child != parent and notTouched[child[0]][child[1]] == True:
          notTouched[child[0]][child[1]] = False
          BFSQueue.put((child, pos, level + 1))
          self.parents[child] = pos

    pos = bestEnd
    self.pathLength = maxLevel + 1
    while(pos in self.parents.keys()):
      self.path[pos] = True
      pos = self.parents[pos]
    print('pathlength:', self.pathLength)  
    
  #  self.updateLength()
    self.buttonToInsertMode()

  #  q = Queue()
  #  q.put(self.cellkernel, 0, 0)
  #  
  #  while (q.full()):
 #     pos, parent, depth = q.get()
      

    self.updateImg()
    
    
  
    ############## Slider-Events ###############
  def sliderThresholdEvent(self, event):
    print ('ThresholdSlider set to', self.sliderThreshold.get())
    self.resetSelection()
    self.updateMask()
    self.updateImg()
    self.saveStatus()
    
  def sliderJumpEvent(self, event):
    print ('JumpSlider set to', self.sliderJump.get())
    self.updateMask()
    self.saveStatus()
    if self.showSelection: # falls gerade eine Selection aktiv ist (geklickt wurde)
      self.click() #updateImg included
      
      
   ############ ButtonLength event ##################
  def insertLength(self):
    self.buttonLength.config(state = DISABLED)
    if self.deleteMode == False:
      self.listboxLength.insert(END, self.pathLength)
      print("write backup")
      backup = open(self.backupName, "a")
      backup.write(str(self.pathLength)+'\n')
      backup.close()
    else:
      self.listboxLength.delete(ANCHOR)
      self.buttonToInsertMode()
      
  def buttonToInsertMode(self):
    self.buttonLength.config(text= str(self.pathLength)+  'px', state = NORMAL)
    self.deleteMode = False
    
  def buttonToDeleteMode(self, event):
    self.buttonLength.config(text= 'Löschen', state = NORMAL)
    self.deleteMode = True
    
    ############## OpenFolder #######################
    
    
  def selectFolder(self):
    self.folderPath = askdirectory(initialdir = self.folderPath)
    self.saveStatus()
    self.openFolder()
    
  def openFolder(self, selection = 0):
    self.listboxFiles.delete(0, END)
    try:
      for filename in sorted(os.listdir(self.folderPath)):
        if filename.endswith('.tif'):
          self.listboxFiles.insert(END, filename)
    except Exception:
      print("no Folder found")
    if self.listboxFiles.size() > 0:
      self.listboxFiles.selection_set(selection)
      self.loadFile(0)
    
    
    ########### Load Image #######################
  def loadFile(self, event):
    i = self.listboxFiles.curselection()
    self.saveStatus()
    if i != (): # ist etwas ausgewaehlt?
      self.initiateImage(self.folderPath + '/' + self.listboxFiles.get(i))
    
  def initiateImage(self, filepath):
    print("opening", filepath)
    self.filename = filepath
    self.image = 0
    try:
      self.image = Image.open(self.filename)
    except Exception:
      self.image = Image.new("RGB", self.imageSize)
      print(self.image)
    
    ## Das Bild auf einen Wert vereinfachen.
    a = np.asarray(self.image).copy()
    r = a[:,:,0]
    g = a[:,:,1]
    b = a[:,:,2]
    self.monochrom = r + g + b #das Bild mit nurnoch einem Wert pro Pixel
    
    ## The original Image:
    imageResize = self.image.resize(self.displaySize)
    self.imageTk = ImageTk.PhotoImage(imageResize)
    
    self.imageWid.itemconfig(self.image_on_canvas, image = self.imageTk)
    
    self.resetSelection()
    self.updateMask()
    self.updateImg()
    
    
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
    maskScaled = 100* self.mask  
    a = np.asarray(self.image).copy()    
    if self.showSelection:
      bigPath = dilation(self.path, disk(3)) # Warnung for rounding (doesnt matter)
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
    
    
  def saveStatus(self):
    status = {"folderPath" :self.folderPath, 
                  "currentFileSelection" : self.listboxFiles.curselection(), 
                  "threshold": self.sliderThreshold.get(),
                  "jump": self.sliderJump.get()}
    print("save status")
    with open('status.txt', 'wb') as handle:
      pickle.dump(status, handle)

  def loadStatus(self):
    print("load status")
    try:
      with open('status.txt', 'rb') as handle:
        status = pickle.loads(handle.read())
        self.folderPath = status["folderPath"]
        self.sliderThreshold.set(status["threshold"])
        self.sliderJump.set(status["jump"])
        self.openFolder(selection = status["currentFileSelection"])
    except Exception:
     print("No status.txt found")
      

instance = Axonium()


