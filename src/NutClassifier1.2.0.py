import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os, re, sys
from detection import detect_nuts, process_nuts, predict_nut_types, crop_regions
from utils.postprocess import adjust_coordinate


class SidebarApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", False)  # Set the window to fullscreen
        #self.root.state("zoomed")
        self.root.configure(bg="black")  # Set background color to black

        # Get the screen resolution
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Sidebar dimensions
        self.sidebar_width = 260
        self.sidebar_visible = False  # Sidebar initially hidden
        self.extra_canvas_visible = False  # Extra canvas initially hidden

        # Main canvas for webcam display
        self.main_canvas = tk.Canvas(self.root, bg="black")
        self.main_canvas.pack(fill="both", expand=True)  # Fill the entire window

        # Sidebar canvas with camera control buttons
        self.sidebar_canvas = tk.Canvas(self.root, width=self.sidebar_width, height=root.winfo_screenheight(), bg="gray")
        self.sidebar_canvas.place(x=root.winfo_screenwidth(), y=0)  # Place out of bounds initially

        # Sidebar toggle button to open/close sidebar
        self.toggle_button = tk.Button(self.root, text="<", command=self.toggle_sidebar, font=("Arial", 16))
        self.update_button_position()  # Position the toggle button

        # Webcam controls
        self.camera_on = False  # Camera initially off
        self.freeze = False  # Freeze frame initially off
        self.cap = None  # Video capture object

        # Sidebar buttons
        self.on_off_button = tk.Button(self.sidebar_canvas, text="On/Off", command=self.toggle_camera, font=("Arial", 15))
        self.on_off_button.place(x=100, y=50, anchor="center")

        self.freeze_button = tk.Button(self.sidebar_canvas, text="Freeze/Unfreeze", command=self.toggle_freeze, font=("Arial", 15))
        self.freeze_button.place(x=100, y=100, anchor="center")

        self.capture_button = tk.Button(self.sidebar_canvas, text="Capture", command=self.capture_image, font=("Arial", 15))
        self.capture_button.place(x=100, y=150, anchor="center")

        self.nut_selection_button = tk.Button(self.sidebar_canvas, text="Nut selection", command=self.toggle_nut_selection, font=("Arial", 15))
        self.nut_selection_button.place(x=100, y=200, anchor="center")

        # Extra canvas for buttons and checkboxes (initially out of bounds)
        self.extra_canvas = tk.Canvas(self.root, width=200, height=root.winfo_screenheight(), bg="lightgray")
        self.extra_canvas_widgets = []
        self.create_extra_canvas()  # Populate extra canvas with checkboxes
        self.extra_canvas.place(x=self.root.winfo_screenwidth(), y=0)  # Start out of bounds

        # Table for displaying selected nuts
        self.table_canvas = tk.Canvas(self.sidebar_canvas, width=self.sidebar_width - 60, bg="white")
        self.table_canvas.place(x=10, y=300, height=root.winfo_screenheight() * 0.65)

        # Scrollbar and setup
        self.scrollbar_frame = tk.Frame(self.sidebar_canvas, bg="gray")
        self.scrollbar_frame.place(x=self.sidebar_width - 50, y=300, height=root.winfo_screenheight() * 0.65)

        self.scrollbar = tk.Scrollbar(self.scrollbar_frame, orient="vertical", command=self.table_canvas.yview)
        self.scrollbar.pack(fill="y", expand=True)
        self.table_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.table_frame = tk.Frame(self.table_canvas, bg="white")
        self.table_canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        # Initialize table and selection data
        self.selected_nuts = []
        self.nut_quantities = {
            f"M{i}": 0 for i in range(1, 21)
        }
        self.create_table()  # Create table headers
        self.update_table()  # Populate table with initial data
        self.circles = []  # Store references to drawn circles

        # Nuts color and size
        self.sizes = ["M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20"]
        self.colors = {
            "M3": "blue",
            "M4": "red",
            "M5": "green",
            "M6": "yellow",
            "M7": "purple",
            "M8": "orange",
            "M9": "cyan",
            "M10": "magenta",
            "M11": "brown",
            "M12": "pink",
            "M13": "gray",
            "M14": "black",
            "M15": "white",
            "M16": "blue",
            "M17": "red",
            "M18": "green",
            "M19": "yellow",
            "M20": "purple"
        }
        self.radiuses = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    def create_table(self):
        # Clear existing table contents
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Create table headers
        headers = ["Nut type", "Quantity", "Color"]
        for col, header_text in enumerate(headers):
            header = tk.Label(self.table_frame, text=header_text, borderwidth=1, relief="solid", width=10, height=2)
            header.grid(row=0, column=col, padx=2, pady=2)

    def update_table(self):
        self.create_table()  # Recreate table headers
        for row, nut in enumerate(self.selected_nuts, start=1):
            nut_type, quantity = nut, self.nut_quantities[nut]
            color = self.colors[nut_type]
            tk.Label(self.table_frame, text=nut_type, borderwidth=1, relief="solid", width=10, height=2).grid(row=row, column=0)
            tk.Label(self.table_frame, text=quantity, borderwidth=1, relief="solid", width=10, height=2).grid(row=row, column=1)
            color_label = tk.Label(self.table_frame, text=color, borderwidth=1, relief="solid", width=10, height=2, bg=color)
            color_label.grid(row=row, column=2)
            color_label.bind("<Button-1>", lambda e, nut=nut_type: self.change_color(nut))
        self.table_frame.update_idletasks()
        self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))

    def change_color(self, nut):
        # Change the color of the nut
        color = tk.colorchooser.askcolor(title=f"Choose color for {nut}")[1]
        if color:
            self.colors[nut] = color
            self.update_table()  # Update table to reflect the new color

    def toggle_sidebar(self):
        # Toggle sidebar visibility
        if self.sidebar_visible:
            self.sidebar_canvas.place(x=self.root.winfo_screenwidth(), y=0)  # Hide sidebar
            self.toggle_button.config(text="<")
        else:
            x_pos = self.root.winfo_screenwidth() - self.sidebar_width
            self.sidebar_canvas.place(x=x_pos, y=0)  # Show sidebar
            self.toggle_button.config(text=">")
        self.sidebar_visible = not self.sidebar_visible
        self.update_button_position()  # Update toggle button position

    def update_button_position(self):
        # Update the position of the toggle button
        button_y = self.root.winfo_screenheight() // 2
        button_x = self.root.winfo_screenwidth() - (self.sidebar_width if self.sidebar_visible else 0)
        self.toggle_button.place(x=button_x, y=button_y, anchor="e")

    def toggle_camera(self):
        # Toggle camera on/off
        if self.camera_on:
            self.camera_on = False
            self.main_canvas.delete("all")  # Clear the main canvas
            if self.cap:
                self.cap.release()  # Release the video capture object
        else:
            self.cap = cv2.VideoCapture(0)  # Open the default camera
            while not self.cap.isOpened():
                print("Camera not detected. Trying again...")
            print("Camera detected.")
            self.camera_on = True
            self.freeze = False
            self.update_frame()  # Start updating frames

    def update_frame(self):
        # Update the frame from the camera
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
                # Redraw circles to ensure they remain visible
                for circle in self.circles:
                    self.main_canvas.tag_raise(circle)
            self.root.after(100, self.update_frame)  # Schedule the next frame update

    def toggle_freeze(self):
        # Toggle freeze/unfreeze frame
        self.freeze = not self.freeze
        if not self.freeze:
            self.update_frame()  # Resume frame updates if unfreezing

    def capture_image(self):
        # Capture and save the current frame
        if self.camera_on:
            self.clear_nut_quantities()
            ret, frame = self.cap.read()
            if not ret:
                return
            crop_region = crop_regions["all"]
            detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes = detect_nuts(frame, crop_region)
            nuts, center_Y = process_nuts(detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes, crop_region)
            predictions = predict_nut_types(nuts)
            print(predictions)
            self.clear_circles()  # Clear existing circles
            for i, center in enumerate(center_Y):
                if self.sizes[predictions[i]] in self.selected_nuts:
                    self.draw_circle(center, predictions[i])
            self.update_table_with_predictions(predictions)  # Update table with predictions

    def clear_nut_quantities(self):
        # Clear all nut quantities
        for nut in self.nut_quantities:
            self.nut_quantities[nut] = 0
    
    def update_table_with_predictions(self, predictions):
        # Update the table according to predictions
        nut_sizes = [f"M{i}" for i in range(3, 21)]
        print(nut_sizes)
        for prediction in predictions:
            nut_size = nut_sizes[prediction]
            self.nut_quantities[nut_size] += 1

        self.update_table()

    def clear_circles(self):
        # Clear all drawn circles
        for circle in self.circles:
            self.main_canvas.delete(circle)
        self.circles = []

    def toggle_nut_selection(self):
        # Toggle nut selection canvas visibility
        if self.extra_canvas_visible:
            self.extra_canvas.place(x=self.root.winfo_screenwidth(), y=0)  # Hide extra canvas
            self.extra_canvas_visible = False
            self.nut_selection_button.config(text="Nut selection")
        else:
            # Attach extra_canvas next to sidebar when visible
            x_position = self.root.winfo_screenwidth() - self.sidebar_width - 55
            self.extra_canvas.place(x=x_position, y=0)
            for widget in self.extra_canvas_widgets:
                widget.grid()  # Show all widgets in the extra canvas
            self.extra_canvas_visible = True
            self.nut_selection_button.config(text="Close Nut Selection")

    def create_extra_canvas(self):
        # Create checkboxes for nut selection
        nut_options = [f"M{i}" for i in range(1, 21)]
        for i, nut in enumerate(nut_options):
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.extra_canvas, font=("Arial", 15), text=nut, variable=var, command=lambda n=nut, v=var: self.toggle_nut(n, v))
            checkbox.grid(row=i, column=0, sticky="w")
            self.extra_canvas_widgets.append(checkbox)

    def toggle_nut(self, nut, var):
        # Toggle nut selection
        if var.get():
            if nut not in self.selected_nuts:
                self.selected_nuts.append(nut)
                # Sort with custom key that extracts numeric part
                self.selected_nuts.sort(key=lambda x: int(re.search(r'\d+', x).group()))
                self.update_table()  # Update table with selected nuts
        else:
            if nut in self.selected_nuts:
                self.selected_nuts.remove(nut)
                self.update_table()  # Update table after removing nut
    
    def draw_circle(self, center, ind=0):
        x_offset = 125
        y_offset = 20
        y_scale = 25
        size = self.sizes[ind]
        color = self.colors[size]
        radius = self.radiuses[ind]
        x, y = center
        x, y = adjust_coordinate(x, y, self.screen_width, self.screen_height, x_offset, y_offset, y_scale)
        circle = self.main_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        text = self.main_canvas.create_text(x, y + radius + 10, text=f"{size}", fill=color)
        self.circles.extend([circle, text])  # Store references to the circle and text


# Create the root Tkinter window
root = tk.Tk()

# Create SidebarApp instance
app = SidebarApp(root)

# Start the main loop
root.mainloop()
