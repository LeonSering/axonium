# Axonium

Axonium is a tool for automatically measuring the length of [axons](https://en.wikipedia.org/wiki/Axon), the nerve fiber connection neurons in a biological brain.

During her studies in the neurology, my girlfriend had to measure the length of a lot of axons. They used ImageScope to load the TIFF-image and then measure the length (in pixel) by a simple ruler tool and then enter the value into Excel.

As it was necessary to measures thousands of axons, and each measurement took quite long and was error prune, I decided to write a small tool, to make this automatically.

The basic idea is the following:

1. First isolate the axons by making it monochrome, i.e., each pixel is either false (black background) or true (colored foreground = axons) based on an adjustable threshold.

2. Then skeletonize the axons with the [medial axis transformation](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis). Basically, the axon (connected components in the monochrome image) are shrunk to have a width of one pixel.

3. Finally, the longest path within the skeleton is determined by a [breadth-first search (BFS)](https://en.wikipedia.org/wiki/Breadth-first_search).

Since axon images can get wild, I added a couple of tools (a pen to connect interrupted axons and an eraser to separate two intersecting axons) and the functionality to still measure by hand.

Finally, the length is easily stored and can be exported to an Excel file.

# Requirements

python3 with pip

# Install

1. Set up a python virtual environment
   
   ```bash
   python3 -m venv env
   ```

2. Start the virtual environment
   
   ```bash
   source env/bin/activate
   ```

3. Install the python dependencies
   
   ```bash
   pip install -r requirements.txt
   ```

(To terminate the virtual environment afterward, use the command: `deactivate`)

# Usage

1. Start the GUI with 
   
   ```bash
   python3 axonium.py
   ```

2. Open a folder with TIFF-images of axons (e.g. the images in examples_images/)

3. Select an image on the left panel.

4. The left canvas always shows the original image.

5. Set the threshold to a level such that the axons are clearly visible and isolated on the right canvas.

6. Set the jump value, such that the same axon is connected, but two different axons are not connected.

7. Right-click on an axon, to measure the length of it. The measured distance always starts at from an axon-point closest to the mouse click and ends at the furthest axon-point.

8. Click on the pixel-button in the lower right corner (or press space) to add the length to the left list.

9. If the axon is not well-connected, but you cannot increase the jump value, use the Pen (click the button or press 'e') to connect the components by drawing a line with the left mouse button.

10. If two axons are connected, you can use the eraser (click on the button or press 'r') to separate them by drawing a cut with the left mouse button.

11. To measure the distance by hand, use the "Distance"-button or press 'd'. Use the left mouse button to draw a line. This disables the automatic distance measure (right click). With the slider next to the "Distance"-button, you can change the gamma-value for better visuals.

12. All distances can be exported to the Excel-format .xlsx by pressing the "Excel"-button.
