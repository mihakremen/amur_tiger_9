import tkinter as tkr
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil
import json


class ImageSorterUI:
    window = tkr.Tk()

    json_path = 'C:\\Users\\user\\PycharmProjects\\tigers\\atrw_anno_pose_train\\keypoint_train.json'
    with open(json_path) as f:
        json_data = json.load(f)

    def __init__(self):
        self.image_paths = []
        self.current_index = 0
        self.sides = ['right', 'left', 'other']
        self.current_image = None

        self.window.title("ImageSorter")
        for i in range(4):
            self.window.grid_rowconfigure(i, weight=1)
            self.window.grid_columnconfigure(i, weight=1)

        self.canvas = tkr.Canvas(self.window, width=1024, height=768)
        self.canvas.grid(row=0, column=0, columnspan=4, sticky=tkr.NSEW)
        self.canvas.bind("<Key>", self.key_event)

        tkr.Label(text="Клавиша 1 - право:").grid(row=1, column=0, sticky=tkr.NSEW)
        self.entry_path_coupls = tkr.Entry(self.window)
        self.entry_path_coupls.grid(row=1, column=1, sticky=tkr.NSEW)
        tkr.Label(text="Клавиша 2 - лево:").grid(row=1, column=2, sticky=tkr.NSEW)
        self.entry_path_rolls = tkr.Entry(self.window)
        self.entry_path_rolls.grid(row=1, column=3, sticky=tkr.NSEW)

        tkr.Label(text="Клавиша 3 - другое:").grid(row=2, column=0, sticky=tkr.NSEW)
        self.entry_rolls_lbrolls = tkr.Entry(self.window)
        self.entry_rolls_lbrolls.grid(row=2, column=1, sticky=tkr.NSEW)
        tkr.Label(text="Клавиша 4 - пропустить:").grid(row=2, column=2, sticky=tkr.NSEW)
        self.entry_coupl_rolls = tkr.Entry(self.window)
        self.entry_coupl_rolls.grid(row=2, column=3, sticky=tkr.NSEW)

        self.label = tkr.Label(self.window, text="Выберите директорию с фотографиями:").grid(row=4, column=0,
                                                                                             columnspan=2,
                                                                                             sticky=tkr.NSEW)
        self.load_button = tkr.Button(self.window, text="Выбрать директорию", command=self.load_directory)
        self.load_button.grid(row=4, column=2, columnspan=2, sticky=tkr.NSEW)

        self.window.mainloop()

    def load_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.image_paths = [f'{directory}/{filename}' for filename in os.listdir(directory)]
            if self.image_paths:
                self.load_image()

    def load_image(self):
        if self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            print(image_path)
            self.current_image = Image.open(image_path)
            self.photo = ImageTk.PhotoImage(self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkr.NW)
            self.canvas.focus_set()
        else:
            self.label.config(text="Все фотографии отсортированы.")
            self.load_button.config(state="disabled")

    def key_event(self, event):
        key = event.keysym
        if key == "space":
            self.current_index += 1
            self.load_image()
        elif key in ["1", "2", "3"]:
            category = int(key) - 1
            if category >= 0:
                self.save_photo(self.sides[category], 'C:/Users/user/PycharmProjects/tigers/atrw_pose_train/labeled')
                self.current_index += 1
                self.load_image()

    def save_photo(self, side, save_path):
        filename = int(os.path.basename(self.image_paths[self.current_index]).split('.')[0].strip("0"))
        for idx, annotation in enumerate(self.json_data['annotations']):
            if annotation['image_id'] == filename:
                # width = annotation['bbox'][2]
                # heigth = annotation['bbox'][3]
                self.json_data['annotations'][idx].update(side=side)
                print(self.json_data['annotations'][idx])
                with open('labeled.json', 'w') as outfile:
                    json.dump(self.json_data['annotations'], outfile)
                original_path = self.image_paths[self.current_index]
                shutil.move(original_path, save_path)

if __name__ == "__main__":
    app = ImageSorterUI()
