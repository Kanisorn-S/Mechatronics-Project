import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os, re

class SidebarApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="white")

        # Sidebar dimensions
        self.sidebar_width = 230
        self.extra_canvas_width = 180
        self.sidebar_visible = False
        self.extra_canvas_visible = False

        # Main canvas frame with green border
        self.main_canvas_frame = tk.Frame(self.root, highlightbackground="green", highlightthickness=5)
        self.main_canvas_frame.pack(fill="both", expand=True)

        # Main canvas for webcam display
        self.main_canvas = tk.Canvas(self.main_canvas_frame, bg="white")
        self.main_canvas.pack(fill="both", expand=True)

        # Sidebar canvas for controls
        self.sidebar_canvas = tk.Canvas(self.root, width=self.sidebar_width, height=root.winfo_screenheight(), bg="gray")

        # Sidebar toggle button to show/hide sidebar
        self.toggle_button = tk.Button(self.root, text="<", command=self.toggle_sidebar, font=("Arial", 16))
        self.update_button_position()

        # Webcam control buttons on sidebar
        self.on_off_button = tk.Button(self.sidebar_canvas, text="On/Off", command=self.toggle_camera, font=("Arial", 15))
        self.on_off_button.place(x=100, y=50, anchor="center")

        self.freeze_button = tk.Button(self.sidebar_canvas, text="Freeze/Unfreeze", command=self.toggle_freeze, font=("Arial", 15))
        self.freeze_button.place(x=100, y=100, anchor="center")

        self.capture_button = tk.Button(self.sidebar_canvas, text="Capture", command=self.capture_image, font=("Arial", 15))
        self.capture_button.place(x=100, y=150, anchor="center")

        self.nut_selection_button = tk.Button(self.sidebar_canvas, text="Nut selection", command=self.toggle_nut_selection, font=("Arial", 15))
        self.nut_selection_button.place(x=100, y=200, anchor="center")

        # Extra canvas for nut selection checkboxes
        self.extra_canvas = tk.Canvas(self.root, width=self.extra_canvas_width, height=root.winfo_screenheight(), bg="lightgray")
        self.extra_canvas_widgets = []
        self.create_extra_canvas()

        # Table canvas on sidebar for displaying selected nuts
        self.table_canvas = tk.Canvas(self.sidebar_canvas, width=self.sidebar_width - 40, bg="white")
        self.table_canvas.place(x=10, y=300, height=root.winfo_screenheight() * 0.65)

        # Scrollbar configuration
        self.scrollbar_frame = tk.Frame(self.sidebar_canvas, bg="gray")
        self.scrollbar_frame.place(x=self.sidebar_width - 50, y=300, height=root.winfo_screenheight() * 0.65)

        self.scrollbar = tk.Scrollbar(self.scrollbar_frame, orient="vertical", command=self.table_canvas.yview)
        self.scrollbar.pack(fill="y", expand=True)
        self.table_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.table_frame = tk.Frame(self.table_canvas, bg="white")
        self.table_canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        # Initialize table and selection data
        self.selected_nuts = []
        self.create_table()
        self.update_table()

    def create_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        headers = ["Nut type", "Quantity"]
        for col, header_text in enumerate(headers):
            header = tk.Label(self.table_frame, text=header_text, borderwidth=1, relief="solid", width=10, height=2)
            header.grid(row=0, column=col, padx=2, pady=2)

    def update_table(self):
        self.create_table()
        for row, nut in enumerate(self.selected_nuts, start=1):
            nut_type, quantity = nut, 0
            tk.Label(self.table_frame, text=nut_type, borderwidth=1, relief="solid", width=10, height=2).grid(row=row, column=0)
            tk.Label(self.table_frame, text=quantity, borderwidth=1, relief="solid", width=10, height=2).grid(row=row, column=1)
        self.table_frame.update_idletasks()
        self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_canvas.place_forget()
            self.toggle_button.config(text="<")
        else:
            x_pos = self.root.winfo_screenwidth() - self.sidebar_width - 5
            self.sidebar_canvas.place(x=x_pos, y=5)
            self.toggle_button.config(text=">")
        self.sidebar_visible = not self.sidebar_visible
        self.update_button_position()

    def update_button_position(self):
        button_y = self.root.winfo_screenheight() // 2
        button_x = self.root.winfo_screenwidth() - (self.sidebar_width if self.sidebar_visible else 0)
        self.toggle_button.place(x=button_x, y=button_y, anchor="e")

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.main_canvas.delete("all")
            if self.cap:
                self.cap.release()
        else:
            self.camera_on = True
            self.freeze = False
            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def update_frame(self):
        if self.camera_on and not self.freeze:
            ret, frame = self.cap.read()
            if ret:
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                new_width = self.root.winfo_screenwidth() - self.sidebar_width
                new_height = int(new_width / aspect_ratio)
                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(img)
                self.main_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                self.main_canvas.image = img_tk
            self.root.after(100, self.update_frame)

    def toggle_freeze(self):
        self.freeze = not self.freeze
        if not self.freeze:
            self.update_frame()

    def capture_image(self):
        if self.camera_on:
            ret, frame = self.cap.read()
            if ret:
                filename = "captured_image.png"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Image Captured", f"Image saved as {filename}")

    def toggle_nut_selection(self):
        if self.extra_canvas_visible:
            self.extra_canvas.place_forget()
            self.nut_selection_button.config(text="Nut selection")
        else:
            x_position = self.root.winfo_screenwidth() - self.sidebar_width -55
            self.extra_canvas.place(x=x_position, y=5)
            self.nut_selection_button.config(text="Close Nut Selection")
        self.extra_canvas_visible = not self.extra_canvas_visible

    def create_extra_canvas(self):
        nut_options = [f"M{i}" for i in range(1, 21)]
        for i, nut in enumerate(nut_options):
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.extra_canvas, font=("Arial", 15), text=nut, variable=var, command=lambda n=nut, v=var: self.toggle_nut(n, v))
            checkbox.grid(row=i, column=0, sticky="w")
            self.extra_canvas_widgets.append(checkbox)

    def toggle_nut(self, nut, var):
        if var.get():
            if nut not in self.selected_nuts:
                self.selected_nuts.append(nut)
                self.selected_nuts.sort(key=lambda x: int(re.search(r'\d+', x).group()))
                self.update_table()
        else:
            if nut in self.selected_nuts:
                self.selected_nuts.remove(nut)
                self.update_table()

# Create the root Tkinter window
root = tk.Tk()

# Create SidebarApp instance
app = SidebarApp(root)

# Start the main loop
root.mainloop()
