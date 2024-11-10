# Import necessary libraries
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os

# Define the SidebarApp class
class SidebarApp:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.attributes("-fullscreen", True)  # Make the window fullscreen
        self.root.configure(bg="black")

        # Main canvas that fills the screen
        self.main_canvas = tk.Canvas(self.root, bg="black")
        self.main_canvas.pack(fill="both", expand=True)

        # Sidebar width
        self.sidebar_width = 200
        self.sidebar_visible = False  # Sidebar starts hidden

        # Sidebar canvas, initially off-screen to the right, overlapping the main canvas
        self.sidebar_canvas = tk.Canvas(
            self.root, width=self.sidebar_width, height=root.winfo_screenheight(), bg="gray"
        )
        self.sidebar_canvas.place(x=root.winfo_screenwidth(), y=0)

        # Toggle button on the middle-right side of the screen
        self.toggle_button = tk.Button(
            self.root, text="<", command=self.toggle_sidebar, font=("Arial", 16)
        )
        self.update_button_position()

        # Webcam controls
        self.camera_on = False
        self.freeze = False
        self.cap = None

        # Sidebar buttons
        self.on_off_button = tk.Button(self.sidebar_canvas, text="On/Off", command=self.toggle_camera, font=("Arial", 12))
        self.on_off_button.place(x=50, y=50, anchor="center")

        self.freeze_button = tk.Button(self.sidebar_canvas, text="Freeze/Unfreeze", command=self.toggle_freeze, font=("Arial", 12))
        self.freeze_button.place(x=50, y=100, anchor="center")

        self.capture_button = tk.Button(self.sidebar_canvas, text="Capture", command=self.capture_image, font=("Arial", 12))
        self.capture_button.place(x=50, y=150, anchor="center")

    # Toggle the sidebar visibility
    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_canvas.place(x=self.root.winfo_screenwidth(), y=0)
            self.toggle_button.config(text="<")
        else:
            self.sidebar_canvas.place(x=self.root.winfo_screenwidth() - self.sidebar_width, y=0)
            self.toggle_button.config(text=">")
        self.sidebar_visible = not self.sidebar_visible
        self.update_button_position()

    # Update the position of the toggle button
    def update_button_position(self):
        screen_height = self.root.winfo_screenheight()
        button_y = screen_height // 2
        button_x = self.root.winfo_screenwidth() - (self.sidebar_width if self.sidebar_visible else 0)
        self.toggle_button.place(x=button_x, y=button_y, anchor="e")

    # Toggle the camera on and off
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

    # Update the frame from the webcam
    def update_frame(self):
        if self.camera_on and not self.freeze:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(img)
                self.main_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                self.main_canvas.image = img_tk
            self.root.after(10, self.update_frame)

    # Toggle the freeze state of the camera
    def toggle_freeze(self):
        if self.camera_on:
            self.freeze = not self.freeze
            if not self.freeze:
                self.update_frame()

    # Capture an image from the webcam
    def capture_image(self):
        if self.camera_on:
            ret, frame = self.cap.read()
            if ret:
                filename = "captured_image.png"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Image Captured", f"Image saved as {filename}")
            else:
                messagebox.showwarning("Capture Failed", "Could not capture image.")
        else:
            messagebox.showwarning("Camera Off", "Please turn on the camera first.")

# Create the main window
root = tk.Tk()
app = SidebarApp(root)
root.mainloop()
