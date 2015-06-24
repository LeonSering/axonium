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
from skimage.draw import line
#import timeit
import datetime


#np.set_printoptions(threshold=np.nan) # gesamte Matrix anschauen

class Axonium:
  
  def __init__(self):
    self.main = Tk()
    self.main.title("Axonium")
    self.main.resizable(False, False) #Groesse kann nicht mehr veraendert werden

 
    ## Anzeigegroesse und Skalierungsfaktor.
    self.imageSize = [1040, 1392] # erst y, dann x
    self.skalierungsfaktor = 0.4
    self.displaySize = [int(self.skalierungsfaktor* self.imageSize[1]), int(self.skalierungsfaktor * self.imageSize[0])] # erst x, dann y
 


    ## Das original Bild
    self.imageWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.imageWid.grid(row=2,column=2, columnspan = 2)
    self.image_on_canvas = self.imageWid.create_image(0,0,anchor=NW)
    self.imageWid.tag_bind(self.image_on_canvas, '<ButtonPress-1>', self.image_click)
    self.imageWid.tag_bind(self.image_on_canvas, '<ButtonPress-3>', self.beginDrawing)
    self.imageWid.tag_bind(self.image_on_canvas, '<B3-Motion>', self.drawing)
    self.imageWid.tag_bind(self.image_on_canvas, '<ButtonRelease-3>', self.endDrawing)
    
    ## Das rechte Bild (MASK)
    self.maskWid = Canvas(self.main, width = self.displaySize[0], height = self.displaySize[1])
    self.maskWid.grid(row=2,column=4, columnspan = 2)
    self.mask_on_canvas = self.maskWid.create_image(0,0,anchor=NW)
    self.maskWid.tag_bind(self.mask_on_canvas, '<ButtonPress-1>', self.mask_click)
    self.maskWid.tag_bind(self.mask_on_canvas, '<ButtonPress-3>', self.beginDrawing)
    self.maskWid.tag_bind(self.mask_on_canvas, '<B3-Motion>', self.drawing)
    self.maskWid.tag_bind(self.mask_on_canvas, '<ButtonRelease-3>', self.endDrawing)
    

    ## Schwellwert Slider
    self.labelThreshold = Label(self.main, text = "Schwellwert:")
    self.labelThreshold.grid(row=3, column = 2)
    self.sliderThreshold = Scale(self.main, from_=0, to=50, length=450, resolution=0.1,tickinterval=5, orient=HORIZONTAL)
    self.sliderThreshold.bind("<ButtonRelease-1>", self.sliderThresholdEvent)
    self.sliderThreshold.set(13)
    self.sliderThreshold.grid(row=3, column = 3)
    

    
    ## Jump Slider
    self.labelJump = Label(self.main, text = "Ausdehnung:")
    self.labelJump.grid(row=3, column = 4)
    self.sliderJump = Scale(self.main, from_=1, to=20, length=450, resolution=1,tickinterval=3, orient=HORIZONTAL)
    self.sliderJump.bind("<ButtonRelease-1>", self.sliderJumpEvent)
    self.sliderJump.set(10)
    self.sliderJump.grid(row=3, column = 5)
    
    ## list box for length
    self.scrollbarLength = Scrollbar(self.main, orient=VERTICAL)        
    self.scrollbarLength.grid(row = 2, column = 7, sticky=N+S)
    self.listboxLength = Listbox (self.main, height = 10, width = 10, yscrollcommand = self.scrollbarLength.set)
    self.listboxLength.grid(row = 2, column = 6,sticky=E+N+S)
    self.scrollbarLength.config(command=self.listboxLength.yview)
    self.listboxLength.bind('<<ListboxSelect>>', self.buttonToDeleteMode)
    
    ## Button der Laenge des aktuellen Pfades anzeigt und bei Klick diesen in diese in die Liste eintraegt (rechts unten)
    self.buttonLength = Button(self.main, text="0", state=DISABLED, command=self.insertLength)
    self.buttonLength.grid(row = 3, column = 6, columnspan=2, sticky=E+W)
    
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
    
    ## Export to Excel
    self.buttonExcel = Button(self.main, text="Excel", command=self.exportToExcel)
    self.buttonExcel.grid(row = 1, column = 6, columnspan = 2, sticky=E+W)

    ## Drawing Mode Buttons
    self.buttonPen = Button(self.main, text="Stift", background='green', state=DISABLED, command=self.selectPen, disabledforeground='black')
    self.buttonPen.grid(row = 1, column = 0, columnspan = 2, sticky=E+W)
    
    self.buttonEreaser = Button(self.main, text="Radiergummi",background = 'lightgrey', command=self.selectEreaser, disabledforeground='black')
    self.buttonEreaser.grid(row = 1, column = 2, sticky=E+W)
    
    ## aktueller Klick
    self.x = 0
    self.y = 0
    
    self.drawingX = 0
    self.drawingY = 0
    self.drawingMode = 1
    
    ## backup initialisieren:
    if not os.path.exists("backup"):
      os.makedirs("backup")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.backupName = "backup/"+timestamp+".txt"
    print(self.backupName)

    ## leeres Bild laden
    self.folderPath = ''      
   # self.initiateImage('') #load a black image
    
    ## alte Einstellungen laden:
    self.loadStatus()
  
  
    self.main.mainloop()
    
    
    
  ########### Left Mouse Clicks ##################  
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
      
    self.updateBigPath()
    self.updateImg()
    
  ############ Right Mouse Clicks ###############
  
  def beginDrawing(self, event):
    self.drawingX, self.drawingY = int(event.x / self.skalierungsfaktor), int(event.y / self.skalierungsfaktor)
    print("Start Drawing")
    
  def drawing(self, event):
    x = min(max(int(event.x / self.skalierungsfaktor), 1), self.imageSize[1] -2)
    y = min(max(int(event.y / self.skalierungsfaktor), 1), self.imageSize[0] -2)
    print("Draw Line from ", self.drawingX, self.drawingY, " to ", x, y)
    rr, cc = line(self.drawingX, self.drawingY, x, y)
    self.bluredMask[cc, rr] = self.drawingMode
    self.bluredMask[cc+1, rr] = self.drawingMode
    self.bluredMask[cc-1, rr] = self.drawingMode
    self.bluredMask[cc, rr+1] = self.drawingMode
    self.bluredMask[cc, rr-1] = self.drawingMode
    self.drawingX, self.drawingY = x, y # letzte Position aktualisieren
    self.updateImg()
    
  def endDrawing(self, event):
    self.updateLabels()
    if self.showSelection:
      self.click()
    else:
      self.x, self.y =self.drawingX, self.drawingY
      self.click()
  
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

  ########### Drawing Mode Button Events ############
  def selectPen(self):
    self.buttonPen.config(background = 'green', state = DISABLED)    
    self.buttonEreaser.config(background = 'lightgrey', state = NORMAL)
    self.drawingMode = 1
    
  def selectEreaser(self):
    self.buttonPen.config(background = 'lightgrey', state = NORMAL)    
    self.buttonEreaser.config(background = 'green', state = DISABLED)  
    self.drawingMode = 0
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
    
  ################## Export To Excel ###############
  def exportToExcel(self):
    print("Export to Excel")
    
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
  def updateMask(self):    ## erstelle neue Mask
    print("updateMask")
    self.mask = (self.monochrom  >= self.sliderThreshold.get())
    
    selem = disk(self.sliderJump.get())
    self.bluredMask = dilation(self.mask, selem) / 255
    self.updateLabels()

    
  def updateLabels(self):
    print("updateLabels")
    ## erstelle Labels neu. Dafuer wird bluredMask aktualisiert.
    self.labels = label(self.bluredMask, neighbors=8, background = 0)
    
  def updateBigPath(self):
    self.bigPath = dilation(self.path, disk(3)) # Warnung for rounding (doesnt matter)    

  ################### Bild aktualisieren ###############
  def updateImg(self):
    maskScaled = 100* self.mask  
    a = np.asarray(self.image).copy()    
    if self.showSelection:
      a[:,:,0] = np.minimum(50*self.bluredMask + maskScaled + self.bigPath,255)
      a[:,:,1] = np.minimum(50*self.selection + self.bigPath,255)
      a[:,:,2] = np.minimum(np.maximum(maskScaled - 50* self.selection, 0) + self.bigPath, 255)
   
   #   a[self.cellkernel[0],self.cellkernel[1],0] = 100 ## Cellkernel einzeichnen
    else:
      a[:,:,0] = np.minimum(50*self.bluredMask + maskScaled,255)
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


