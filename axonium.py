import queue
import os
import pickle
from tkinter import *
from tkinter.filedialog import askdirectory, asksaveasfilename
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
from scipy import sparse
from skimage.morphology import medial_axis
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import label
from skimage.draw import line, circle
from skimage.exposure import adjust_gamma
import xlsxwriter
#import timeit
import datetime


#np.set_printoptions(threshold=np.nan) # gesamte Matrix anschauen

class Axonium:
  
  def __init__(self):
    self.main = Tk()
    self.main.title("Axonium")
    self.main.resizable(False, False) #Groesse kann nicht mehr veraendert werden
 
    self.main.protocol("WM_DELETE_WINDOW", self.closing)
    ## Gibt die Praezision in Pixeln an, mit der die Wege gemessen werden sollen. Zu genau (kleine Werte) gibt Pixel-Ungenauigkeit.
    ## und endet in zu langen Wegen (da Zickzack gemessen wird.
    ## Zu grob (groesse Werte), fuert zu zu kleinen Laengen, da Abkurzungen genommen werden.
    self.precision = 10
 
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
    self.sliderThreshold = Scale(self.main, from_=0, to=100, length=450, resolution=0.1,tickinterval=10, orient=HORIZONTAL)
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
    self.listboxLength = Listbox (self.main, height = 10, width = 10, yscrollcommand = self.scrollbarLength.set, takefocus= 0)
    self.listboxLength.grid(row = 2, column = 6,sticky=E+N+S)
    self.scrollbarLength.config(command=self.listboxLength.yview)
    self.listboxLength.bind('<<ListboxSelect>>', self.buttonToDeleteMode)
    
    ## Button der Laenge des aktuellen Pfades anzeigt und bei Klick diesen in diese in die Liste eintraegt (rechts unten)
    self.buttonLength = Button(self.main, text="0", state=DISABLED, command=self.insertLength)
    self.buttonLength.grid(row = 3, column = 6, columnspan=2, sticky=E+W)
    
    ## filelist
    self.scrollbarFiles = Scrollbar(self.main, orient=VERTICAL)        
    self.scrollbarFiles.grid(row = 2, column = 1, sticky=N+S)
    self.listboxFiles = Listbox (self.main, height = 10, width = 12, yscrollcommand = self.scrollbarFiles.set, takefocus= 0)
    self.listboxFiles.grid(row = 2, column = 0,sticky=E+N+S)
    self.scrollbarFiles.config(command=self.listboxFiles.yview)
    self.listboxFiles.bind('<<ListboxSelect>>', self.loadFile)
    self.main.bind('<Down>', self.nextFile)
    self.main.bind('<Up>', self.prevFile)
    self.listboxFiles.bind('<FocusIn>', self.listboxFilesFocus)
    
    ## Open File Button
    self.buttonOpenFile = Button(self.main, text="Öffne Ordner", command=self.selectFolder)
    self.buttonOpenFile.grid(row = 3, column = 0, columnspan = 2, sticky=E+W)
    
    ## Export to Excel
    self.buttonExcel = Button(self.main, text="Excel", command=self.exportToExcel)
    self.buttonExcel.grid(row = 1, column = 6, columnspan = 2, sticky=E+W)

    ## Drawing Mode Buttons
    self.buttonPen = Button(self.main, text="Stift", background='green', state=DISABLED, command=self.selectPen, disabledforeground='black')
    self.buttonPen.grid(row = 1, column = 0, columnspan = 2, sticky=E+W)
    self.main.bind('s', self.selectPen)
    
    self.buttonEreaser = Button(self.main, text="Radiergummi",background = 'lightgrey', command=self.selectEreaser, disabledforeground='black')
    self.buttonEreaser.grid(row = 1, column = 2, sticky=E+W)
    self.main.bind('r', self.selectEreaser)
    
    self.buttonDistance = Button(self.main, text="Distanz",background = 'lightgrey', command=self.selectDistance, disabledforeground='black')
    self.buttonDistance.grid(row = 1, column = 4, sticky=E+W)
    self.main.bind('d', self.selectDistance)
    
    ## aktueller Klick
    self.x = 0
    self.y = 0
    
    self.drawingX = 0
    self.drawingY = 0
    self.drawingMode = 1
    
    self.deleteMode = False
    
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
    self.click(x,y)

  def image_click(self, event):
    x, y = int(event.x / self.skalierungsfaktor), int(event.y / self.skalierungsfaktor)
    print('ImageClick: x: ', x, ' y: ', y, ' label: ', self.labels[y,x])
    self.click(x,y)
    
  def click(self,x,y):
    if self.drawingMode == -1:
      self.measurePathLength = 0
      self.updateInsertButton()
      self.resetMeasureImg()
    else:
      self.x = x
      self.y = y
      self.findPath()
    
    self.updateImg()    
      
    
######### Laengsten Kurzesten Weg Messen ############    
  def findPath(self):
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
    i = -1
    ankor = pos
    self.pathLength = 0
    while(pos in self.parents.keys()):
      #self.path[pos] = True
      i = i+1
      if i == self.precision:
        i = 0
        self.pathLength = self.pathLength + np.sqrt((ankor[0] - pos[0])*(ankor[0] - pos[0])+(ankor[1] - pos[1])*(ankor[1] - pos[1]))
        rr, cc = line(ankor[0], ankor[1], pos[0], pos[1])
        self.path[rr,cc] = True
        ankor = pos
      pos = self.parents[pos]
    self.pathLength = self.pathLength + np.sqrt((ankor[0] - self.cellkernel[0])*(ankor[0] - self.cellkernel[0])+(ankor[1] - self.cellkernel[1])*(ankor[1] - self.cellkernel[1]))
    rr, cc = line(ankor[0], ankor[1], self.cellkernel[0], self.cellkernel[1])
    self.path[rr,cc] = True
    print('pathlength:', self.pathLength)
    print('maxLevel:', maxLevel)
    
  #  self.updateLength()
    self.buttonToInsertMode()

  #  q = Queue()
  #  q.put(self.cellkernel, 0, 0)
  #  
  #  while (q.full()):
 #     pos, parent, depth = q.get()
      
    self.updateBigPath()
    
############ Right Mouse Clicks ###############
  
  def beginDrawing(self, event):
    self.drawingX, self.drawingY = int(event.x / self.skalierungsfaktor), int(event.y / self.skalierungsfaktor)
    print("Start Drawing")
    if self.drawingMode == -1:
      self.measurePathLength = 0
      self.resetMeasureImg()
      self.buttonToInsertMode()
    
  def drawing(self, event):
    x = min(max(int(event.x / self.skalierungsfaktor), 1), self.imageSize[1] -2)
    y = min(max(int(event.y / self.skalierungsfaktor), 1), self.imageSize[0] -2)

    print("Draw Line from ", self.drawingX, self.drawingY, " to ", x, y)
    rr, cc = line(self.drawingX, self.drawingY, x, y)
    if self.drawingMode == -1:
      if max(x-self.drawingX, y-self.drawingY,self.drawingX - x, self.drawingY - y) < self.precision:
        print("Movement too short")
        return
      self.measureImage[cc, rr] = [255,255,255]
      self.measureImage[cc+1, rr] = [255,255,255]
      self.measureImage[cc-1, rr] = [255,255,255]
      self.measureImage[cc, rr+1] = [255,255,255]
      self.measureImage[cc, rr-1] = [255,255,255]
      self.measurePathLength = self.measurePathLength + np.sqrt((self.drawingX - x)*(self.drawingX - x)+(self.drawingY - y)*(self.drawingY - y))
      print(self.measurePathLength)
      self.buttonToInsertMode()
      self.drawingX, self.drawingY = x, y # letzte Position aktualisieren
    else:
      self.bluredMask[cc, rr] = self.drawingMode
      self.bluredMask[cc+1, rr] = self.drawingMode
      self.bluredMask[cc-1, rr] = self.drawingMode
      self.bluredMask[cc, rr+1] = self.drawingMode
      self.bluredMask[cc, rr-1] = self.drawingMode
      self.drawingX, self.drawingY = x, y # letzte Position aktualisieren
    self.updateImg()
    
  def endDrawing(self, event):
    if self.drawingMode == -1:
      x = min(max(int(event.x / self.skalierungsfaktor), 1), self.imageSize[1] -2)
      y = min(max(int(event.y / self.skalierungsfaktor), 1), self.imageSize[0] -2)
      rr, cc = line(self.drawingX, self.drawingY, x, y)
      self.measureImage[cc, rr] = [255,255,255]
      self.measureImage[cc+1, rr] = [255,255,255]
      self.measureImage[cc-1, rr] = [255,255,255]
      self.measureImage[cc, rr+1] = [255,255,255]
      self.measureImage[cc, rr-1] = [255,255,255]
      self.measurePathLength = self.measurePathLength + np.sqrt((self.drawingX - x)*(self.drawingX - x)+(self.drawingY - y)*(self.drawingY - y))
      print(self.measurePathLength)
      self.buttonToInsertMode()     
    else:
      self.updateLabels()
      if self.showSelection:
        self.findPath()
      else:
        self.x, self.y =self.drawingX, self.drawingY
        self.findPath()
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
      self.findPath() #updateImg included
    self.updateImg()

########### Drawing Mode Button Events ############
  def selectPen(self, event = 0):
    self.buttonPen.config(background = 'green', state = DISABLED)    
    self.buttonEreaser.config(background = 'lightgrey', state = NORMAL)
    self.buttonDistance.config(background = 'lightgrey', state = NORMAL)
    if self.drawingMode == -1:
      self.drawingMode = 1
      self.updateImg()
    self.drawingMode = 1
    self.updateInsertButton()
    
  def selectEreaser(self, event = 0):
    self.buttonPen.config(background = 'lightgrey', state = NORMAL)    
    self.buttonEreaser.config(background = 'green', state = DISABLED)  
    self.buttonDistance.config(background = 'lightgrey', state = NORMAL)
    if self.drawingMode == -1:
      self.drawingMode = 0
      self.updateImg()
    self.drawingMode = 0
    self.updateInsertButton()
    
  def selectDistance(self, event = 0):
    self.buttonPen.config(background = 'lightgrey', state = NORMAL)    
    self.buttonEreaser.config(background = 'lightgrey', state = NORMAL)
    self.buttonDistance.config(background = 'green', state = DISABLED)
    self.resetMeasureImg()
    if self.drawingMode != -1:
      self.drawingMode = -1
      self.updateImg()
      self.measurePathLength = 0
    self.drawingMode = -1
    self.updateInsertButton()

############ ButtonLength event ##################
  def insertLength(self, event = 0):
    if self.buttonLength["state"] == DISABLED:
      return
    self.buttonLength.config(state = DISABLED)
    if self.deleteMode == False:
      if self.drawingMode == -1:
        length = int(self.measurePathLength)
      else:
        length = int(self.pathLength)
      self.listboxLength.insert(END, length)
      print("write backup")
      backup = open(self.backupName, "a")
      backup.write(str(length)+'\n')
      backup.close()
    else:
      self.listboxLength.delete(ANCHOR)
      self.buttonToInsertMode()
      
  def updateInsertButton(self):
    if self.drawingMode == -1:
      if self.measurePathLength == 0:
        self.buttonLength.config(text='Kein Weg', state = DISABLED)  
      else:
        self.buttonLength.config(text= str(int(self.measurePathLength))+  'px', state = NORMAL)
    else:
      if self.showSelection == False:
        self.buttonLength.config(text='Kein Weg', state = DISABLED)
      else:
        self.buttonLength.config(text= str(int(self.pathLength))+  'px', state = NORMAL)
        
  def buttonToInsertMode(self):
    if self.drawingMode == -1:
      length = int(self.measurePathLength)
    else:
      length = int(self.pathLength)
    self.buttonLength.config(text= str(length)+  'px', state = NORMAL)
    self.buttonLength.focus_set()
    self.deleteMode = False
    
  def buttonToDeleteMode(self, event):
    self.buttonLength.config(text= 'Löschen', state = NORMAL)
    self.deleteMode = True
    
################## Export To Excel ###############
  def exportToExcel(self):
    excelFileName = asksaveasfilename(initialdir = self.folderPath, defaultextension=".xlsx")
    excelFile = xlsxwriter.Workbook(excelFileName)
    worksheet = excelFile.add_worksheet()
    i = 1
    for value in self.listboxLength.get(0, END):
      print(value)
      worksheet.write('A'+str(i), value)
      i = i+1
    excelFile.close()
    print("Export to Excel")
    
############## OpenFolder #######################
  def selectFolder(self):
    folderPath = askdirectory(initialdir = self.folderPath)
    if folderPath != '':
      self.folderPath = folderPath
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
      self.loadFile()
    
    
########### Load Image #######################
  ## Falls Datei mit Klick ausgewahlt wurde, soll
  #der Focus nicht auf der listbox bleiben(wegen Pfeiltasten)
  def listboxFilesFocus(self, event = 0): 
    self.buttonOpenFile.focus_set()    
    
  def loadFile(self, event = 0):
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
    
    ## als Array abspeichern
    self.arrayImg = np.asarray(self.image).copy()
    ## Das Bild auf einen Wert vereinfachen.
    r = self.arrayImg[:,:,0]
    g = self.arrayImg[:,:,1]
    b = self.arrayImg[:,:,2]
    self.monochrom = r + g + b #das Bild mit nurnoch einem Wert pro Pixel
    

    
    ## The original Image:
    imageResize = self.image.resize(self.displaySize)
    self.imageTk = ImageTk.PhotoImage(imageResize)
    
    self.imageWid.itemconfig(self.image_on_canvas, image = self.imageTk)
    self.measureImageOriginal = adjust_gamma(np.asarray(self.image).copy(), 0.75)
    self.resetSelection()
    self.resetMeasureImg()
    self.updateMask()
    self.updateImg()
    
  def nextFile(self, event = 0):
    i = self.listboxFiles.curselection()
    print('nextFile')
    if len(i) > 0 and self.listboxFiles.size()-1 > i[0]:
      self.listboxFiles.selection_clear(i)
      self.listboxFiles.selection_set((i[0]+1))
      self.loadFile()
      
  def prevFile(self, event = 0):
    i = self.listboxFiles.curselection()
    print('prevFile')
    if len(i) > 0 and self.listboxFiles.size() > 0:
      self.listboxFiles.selection_clear(i)
      self.listboxFiles.selection_set((i[0]-1))
      self.loadFile()
    
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
    if self.drawingMode == -1:
      maskImg = Image.fromarray(self.measureImage)       
      maskResize = maskImg.resize(self.displaySize)
      self.maskTk = ImageTk.PhotoImage(maskResize)
      self.maskWid.itemconfig(self.mask_on_canvas, image = self.maskTk)    
    else:
      maskScaled = 100* self.mask   
      if self.showSelection:
        self.arrayImg[:,:,0] = np.minimum(50*self.bluredMask + maskScaled + self.bigPath,255)
        self.arrayImg[:,:,1] = np.minimum(50*self.selection + self.bigPath,255)
        self.arrayImg[:,:,2] = np.minimum(np.maximum(maskScaled - 50* self.selection, 0) + self.bigPath, 255)

        rr, cc = circle(self.cellkernel[1], self.cellkernel[0], 7)
        self.arrayImg[cc,rr,0] = 0
        self.arrayImg[cc,rr,1] = 255 
        self.arrayImg[cc,rr,2] = 255
    #   a[self.cellkernel[0],self.cellkernel[1],0] = 100 ## Cellkernel einzeichnen
      else:
        self.arrayImg[:,:,0] = np.minimum(50*self.bluredMask + maskScaled,255)
        self.arrayImg[:,:,1] = 0
        self.arrayImg[:,:,2] = maskScaled
    
      maskImg = Image.fromarray(self.arrayImg)

      maskResize = maskImg.resize(self.displaySize)

      self.maskTk = ImageTk.PhotoImage(maskResize)
      self.maskWid.itemconfig(self.mask_on_canvas, image = self.maskTk)
      
      
  def resetMeasureImg(self):

    self.measureImage = self.measureImageOriginal.copy()

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
 
################## On Closing ###########
  def closing(self):
    if messagebox.askokcancel("Quit", "Wirklich schließen?"):
        self.main.destroy()

instance = Axonium()


