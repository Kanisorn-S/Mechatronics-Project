import tkinter as tk
import win32gui
import win32con
import win32api
from tkinter import colorchooser
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os, re, sys
import threading  # Import threading module
from detection import detect_nuts, process_nuts, predict_nut_types, crop_regions
from utils.postprocess import adjust_coordinate


class SidebarApp:
    def __init__(self, root):

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
            "M16": "light blue",
            "M17": "light green",
            "M18": "light yellow",
            "M19": "light cyan",
            "M20": "dark red"
        }
        self.radiuses = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]

        self.root = root
        self.root.attributes("-fullscreen", True)  # Set the window to fullscreen
        #self.root.state("zoomed")
        self.root.configure(bg="black")  # Set background color to black

        # Get the screen resolution
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Sidebar dimensions
        self.sidebar_width = 360  # Increased width to ensure the entire table is visible
        self.sidebar_visible = False  # Sidebar initially hidden
        self.extra_canvas_visible = False  # Extra canvas initially hidden

        # Main canvas for webcam display
        self.main_canvas = tk.Canvas(self.root, width=self.screen_width, height=self.screen_height, bg="white", highlightbackground="black", highlightthickness=10)
        self.main_canvas.pack(fill="both", expand=True)  # Fill the entire window

        # Sidebar canvas with camera control buttons
        self.sidebar_canvas = tk.Canvas(self.root, width=self.sidebar_width, height=root.winfo_screenheight(), bg="#808080", highlightthickness=0)
        self.sidebar_canvas.place(x=root.winfo_screenwidth(), y=0)  # Place out of bounds initially
        # Make Sidebar Transparent
        hwnd = self.sidebar_canvas.winfo_id()
        colorKey = win32api.RGB(128, 128, 128)
        wnd_exstyle = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        new_exstyle = wnd_exstyle | win32con.WS_EX_LAYERED
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_exstyle)
        win32gui.SetLayeredWindowAttributes(hwnd, colorKey, 200, win32con.LWA_ALPHA)

        self.loading_popup = None
        self.loading_bar = None
        self.loading_label = None

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

        self.capture_button = tk.Button(self.sidebar_canvas, text="Capture", command=self.capture_image, font=("Arial", 15))
        self.capture_button.place(x=100, y=100, anchor="center")

        self.sort_button = tk.Button(self.sidebar_canvas, text="Sort", command=self.sort_nuts, font=("Arial", 15))
        self.sort_button.place(x=200, y=100, anchor="center")

        self.clear_button = tk.Button(self.sidebar_canvas, text="Clear", command=self.toggle_freeze, font=("Arial", 15))
        self.clear_button.place(x=100, y=150, anchor="center")


        self.nut_selection_button = tk.Button(self.sidebar_canvas, text="Nut selection", command=self.toggle_nut_selection, font=("Arial", 15))
        self.nut_selection_button.place(x=100, y=200, anchor="center")

        # Extra canvas for buttons and checkboxes (initially out of bounds)
        self.extra_canvas = tk.Canvas(self.root, width=200, height=root.winfo_screenheight(), bg="lightgray")
        self.extra_canvas_widgets = []
        self.create_extra_canvas()  # Populate extra canvas with checkboxes
        self.extra_canvas.place(x=self.root.winfo_screenwidth(), y=0)  # Start out of bounds

        # Table for displaying selected nuts
        self.table_canvas = tk.Canvas(self.sidebar_canvas, width=self.sidebar_width - 60, bg="white")
        self.table_canvas.place(x=10, y=250, height=root.winfo_screenheight() * 0.65)  # Move the table up

        # Scrollbar and setup
        self.scrollbar_frame = tk.Frame(self.sidebar_canvas, bg="gray")
        self.scrollbar_frame.place(x=self.sidebar_width - 50, y=250, height=root.winfo_screenheight() * 0.65)  # Move the scrollbar up

        self.scrollbar = tk.Scrollbar(self.scrollbar_frame, orient="vertical", command=self.table_canvas.yview)
        self.scrollbar.pack(fill="y", expand=True)
        self.table_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.table_frame = tk.Frame(self.table_canvas, bg="white")
        self.table_canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        self.table_canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Bind mouse wheel to scroll

        # Initialize table and selection data
        self.selected_nuts = [f"M{i}" for i in range(3, 6)]  # Select all nut sizes by default
        self.nut_quantities = {
            f"M{i}": 0 for i in range(3, 21)
        }
        self.create_table()  # Create table headers
        self.update_table()  # Populate table with initial data
        self.circles = []  # Store references to drawn circles

        self.update_frame_id = None
        
        # Save nuts variables
        self.center_Y = []
        self.predictions = []
        
        # Debug
        self.cap_count = 0
        self.file_path = './debug/images/'


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
            color_label = tk.Label(self.table_frame, borderwidth=1, relief="solid", width=10, height=2, bg=color)
            color_label.grid(row=row, column=2)
            color_label.bind("<Button-1>", lambda e, nut=nut_type: self.change_color(nut))
        self.table_frame.update_idletasks()
        self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))

    def change_color(self, nut):
        # Change the color of the nut
        color = colorchooser.askcolor(title=f"Choose color for {nut}")[1]
        if color:
            self.colors[nut] = color
            self.update_table()  # Update table to reflect the new color
            self.update_frame()

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
            self.main_canvas.config(highlightbackground="black", highlightthickness=10)  # Add thicker and brighter green border
        else:
            self.show_loading_popup()
            threading.Thread(target=self.initialize_camera).start()  # Run initialize_camera in a separate thread
            self.update_loading_bar(0)  # Start updating the loading bar

    def show_loading_popup(self):
        # Create a popup window with a loading bar
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.geometry("300x150")
        self.loading_popup.title("Initializing Camera")
        self.loading_popup.transient(self.root)
        self.loading_popup.grab_set()
        self.loading_popup.attributes("-topmost", True)

        # Center the popup window
        x = (self.root.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.root.winfo_screenheight() // 2) - (150 // 2)
        self.loading_popup.geometry(f"+{x}+{y}")

        self.loading_label = tk.Label(self.loading_popup, text="Initializing Camera..", font=("Arial", 12))
        self.loading_label.pack(pady=10)

        self.loading_bar = tk.Canvas(self.loading_popup, width=200, height=20, bg="white", highlightthickness=1, highlightbackground="black")
        self.loading_bar.pack(pady=10)
        self.loading_bar.create_rectangle(0, 0, 0, 20, fill="green", tags="progress")

    def update_loading_bar(self, value):
        # Update the loading bar
        self.loading_bar.coords("progress", 0, 0, value * 2, 20)
        if value <= 100:
            self.root.after(570, self.update_loading_bar, value + 1)
        else:
            self.loading_label.config(text="Camera Ready")
            self.root.after(500, self.close_loading_popup)

    def close_loading_popup(self):
        # Close the loading popup
        if self.loading_popup:
            self.loading_popup.destroy()
            self.loading_popup = None

    def initialize_camera(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(1)  # Open the default camera
        self.camera_on = True
        self.freeze = False
        self.main_canvas.config(highlightbackground="#00FF00", highlightthickness=10)  # Add thicker and brighter green border
        self.update_frame()  # Start updating frames

    def update_frame(self):
        # Update the frame from the camera
        if self.camera_on and not self.freeze:
            ret, frame = self.cap.read()
            if ret:
                # height, width = frame.shape[:2]
                # aspect_ratio = width / height
                # new_width = self.root.winfo_screenwidth() - self.sidebar_width
                # new_height = int(new_width / aspect_ratio)
                # frame = cv2.resize(frame, (new_width, new_height))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # img = Image.fromarray(frame)
                # img_tk = ImageTk.PhotoImage(img)
                # self.main_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                # self.main_canvas.image = img_tk
                # Redraw circles to ensure they remain visible
                for circle in self.circles:
                    self.main_canvas.tag_raise(circle)
            self.update_frame_id = self.root.after(100, self.update_frame)  # Schedule the next frame update

    def toggle_freeze(self):
        # Toggle freeze/unfreeze frame
        # self.freeze = not self.freeze
        # if not self.freeze:
        #     self.update_frame()  # Resume frame updates if unfreezing
        self.clear_circles()
        self.clear_nut_quantities()
        self.update_table()

    def capture_image(self):
        # Capture and save the current frame
        if self.camera_on:
            self.clear_circles()
            self.clear_nut_quantities()
            ret, frame = self.cap.read()
            if not ret:
                return
            crop_region = crop_regions["machine"]
            detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes = detect_nuts(frame, crop_region)
            nuts, center_Y = process_nuts(detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes, crop_region)
            self.center_Y = center_Y
            predictions = predict_nut_types(nuts)
            self.predictions = predictions
            cv2.imwrite(f"{self.file_path}image_{self.cap_count}.jpg", detected)
            self.cap_count += 1
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
        for prediction in predictions:
            nut_size = nut_sizes[prediction]
            self.nut_quantities[nut_size] += 1

        self.update_table()

    def clear_circles(self):
        # Clear all drawn circles
        if hasattr(self, "circles"):
            for circle in self.circles:
                self.main_canvas.delete(circle)
            self.circles = []
            self.main_canvas.update_idletasks()

    def toggle_nut_selection(self):
        # Toggle nut selection canvas visibility
        if self.extra_canvas_visible:
            self.extra_canvas.place(x=self.root.winfo_screenwidth(), y=0)  # Hide extra canvas
            self.extra_canvas_visible = False
            self.nut_selection_button.config(text="Nut selection")
        else:
            # Attach extra_canvas next to sidebar when visible
            x_position = self.root.winfo_screenwidth() - self.sidebar_width - 255
            self.extra_canvas.place(x=x_position, y=0)
            for widget in self.extra_canvas_widgets:
                widget.grid()  # Show all widgets in the extra canvas
            self.extra_canvas_visible = True
            self.nut_selection_button.config(text="Close Nut Selection")

    def create_extra_canvas(self):
        # Create checkboxes for nut selection
        self.select_all_var = tk.BooleanVar(value=False)
        select_all_checkbox = tk.Checkbutton(self.extra_canvas, font=("Arial", 15), text="Select/Deselect All", variable=self.select_all_var, command=self.toggle_select_all)
        select_all_checkbox.grid(row=0, column=0, sticky="w")
        self.extra_canvas_widgets.append(select_all_checkbox)

        nut_options = [f"M{i}" for i in range(3, 21)]
        self.nut_vars = {}
        for i, nut in enumerate(nut_options, start=1):
            if i < 4:
                var = tk.BooleanVar(value=True)
            else:
                var = tk.BooleanVar(value=False)
            checkbox = tk.Checkbutton(self.extra_canvas, font=("Arial", 15), text=nut, variable=var, command=lambda n=nut, v=var: self.toggle_nut(n, v))
            checkbox.grid(row=i, column=0, sticky="w")
            self.extra_canvas_widgets.append(checkbox)
            self.nut_vars[nut] = var

    def toggle_select_all(self):
        # Toggle select/deselect all nuts
        select_all = self.select_all_var.get()
        for nut, var in self.nut_vars.items():
            var.set(select_all)
            if select_all and nut not in self.selected_nuts:
                self.selected_nuts.append(nut)
            elif not select_all and nut in self.selected_nuts:
                self.selected_nuts.remove(nut)
        self.update_table()  # Update table with selected nuts

    def toggle_nut(self, nut, var):
        # Toggle nut selection
        if var.get():
            if nut not in self.selected_nuts:
                self.selected_nuts.append(nut)
                self.selected_nuts.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort with custom key that extracts numeric part
        else:
            if nut in self.selected_nuts:
                self.selected_nuts.remove(nut)
        self.update_table()  # Update table with selected nuts
    
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

    def _on_mousewheel(self, event):
        # Scroll the table with the mouse wheel
        self.table_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def sort_nuts(self):
        # Placeholder function for sorting nuts
        pass


# Create the root Tkinter window
root = tk.Tk()

# Create SidebarApp instance
app = SidebarApp(root)

# Start the main loop
root.mainloop()
