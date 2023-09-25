import numpy as np
from skimage.draw import circle_perimeter, disk
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageTk, ImageGrab
import tkinter
import os

coords_x = []
coords_y = []
indx = 1
set_width = 500 # include prompting for this

def image_quantification(path):
    global indx
    print("Beginning analysis...")

    files = Path(path).glob('**/*.tif')
    
    for imgpath in files:
        image_name = os.path.basename(imgpath).split('.')[0]   
        rgb, grayscale, annotated = read_images(imgpath, image_name)
        rgb_circles, rgb_annotated = find_circles(rgb, grayscale, annotated, coords_x, coords_y, image_name)
        show_annotated_image(rgb_annotated, path, image_name)
        write_excel(coords_x, coords_y, rgb_circles, path, image_name)

        indx = 1
        coords_x.clear()
        coords_y.clear()
    
    print("Analysis complete.")


def read_images(imgpath, image_name):
    global image, canvas, draw

    print("Fetching image file from folder...")
    window = tkinter.Tk(className=image_name)

    image = Image.open(imgpath)
    ratio = set_width / image.size[0]
    new_height = int(image.size[1]*ratio)

    image = image.resize((set_width, new_height))

    annotated = image
    draw = ImageDraw.Draw(annotated)

    rgb = np.array(image)
    rgb = rgb[...,:3]

    grayscale = image.convert('L')
    grayscale = np.array(grayscale)
    
    canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
    canvas.pack()
    image_tk = ImageTk.PhotoImage(image)
    
    canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)
    canvas.bind("<Button-1>", callback)
    tkinter.mainloop()

    annotated = np.array(annotated)
    annotated = annotated[...,:3]

    return rgb, grayscale, annotated
    
def callback(event):
    global indx

    coords_x.append(event.x)
    coords_y.append(event.y)
    canvas.create_text((event.x, event.y), text=str(indx))
    draw.text((event.x, event.y), text=str(indx), fill=(0,0,0))
    indx += 1
        

def find_circles(rgb, grayscale, annotated, coords_x, coords_y, image_name):
    radii = np.zeros(len(coords_x))
    rgb_of_circles = np.zeros((len(coords_x),3))
    
    for i in range(len(coords_x)):
        traffic_light = False
        
        while(traffic_light == False):
            try:
                if abs(int(grayscale[coords_y[i],coords_x[i]]) - int(grayscale[int(coords_y[i] + radii[i]),coords_x[i]])) >= 30:
                    circy, circx = circle_perimeter(coords_y[i], coords_x[i], int(radii[i])-3,shape=grayscale.shape)
                    rr,cc = disk((coords_y[i], coords_x[i]), radii[i],shape=grayscale.shape)
                    region = rgb[rr,cc,:] # rr is y indices, cc is x indices
                    rgb_of_circles[i,:] = np.mean(region, axis=0)
                    annotated[circy, circx] = (0,0,0)
                    
                    traffic_light = True
                else:
                    radii_test = radii[i] + 3
                    radii[i] = min(radii_test,len(grayscale[1,:]))
            except IndexError:
                print('Image error at image', image_name)
                break

    return rgb_of_circles, annotated

def show_annotated_image(rgb_annotated, path, image_name):
    name = image_name + ".png"
    new_path = os.path.join(path, 'annotated_images',name)
    annotated = Image.fromarray(rgb_annotated)
    annotated.show(title=image_name) 
    annotated.save(new_path) 


def write_excel(coords_x, coords_y, rgb_of_circles, path, image_name):
    all_data = pd.DataFrame(rgb_of_circles)
    all_data.columns = ['R','G','B']
    all_data['x positions'] = coords_x
    all_data['y positions'] = coords_y

    name = image_name + ".xlsx"
    new_path = os.path.join(path, 'image_RGB',name)
    all_data.to_excel(new_path)

