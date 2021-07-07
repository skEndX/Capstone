# import cv2
# import numpy as np

# def loadImageFromPath(imgPath):
#         try:
#             # gif 처리
#             if str(imgPath).lower().endswith('.gif'):
#                 gif = cv2.VideoCapture(imgPath)
#                 ret, frame = gif.read()  # ret=True if it finds a frame else False.
#                 cv2.imshow('capstone', frame)
#                 if ret:
#                     return frame
#             else:
#                 return cv2.imread(imgPath)
#         except Exception as e:
#             print(e)
#             return None

# def main():
#     print ("Main Function")
#     gif = cv2.VideoCapture("C:/Capstone/loading.gif")
#     ret, frame = gif.read()  # ret=True if it finds a frame else False.
#     while ret:
# 	# something to do 'frame'
#     # ...
#     # 다음 frame 읽음
#         ret, frame = gif.read()
#         #cv2.imshow('capstone', frame)
    

# if __name__ == "__main__":
# 	main()

import tkinter as tk
from PIL import Image, ImageTk
from itertools import count

class ImageLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []
        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()
    def unload(self):
        self.config(image=None)
        self.frames = None
    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)
            #self.window.attributes('-fullscreen', True)  
root = tk.Tk()
lbl = ImageLabel(root)
lbl.pack()
lbl.load('loading.gif')
root.mainloop()