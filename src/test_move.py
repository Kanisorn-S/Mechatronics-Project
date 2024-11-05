import tkinter as tk
import pyautogui
import random

# Function to move the mouse cursor to a specific position
def move_cursor(x, y):
    pyautogui.moveTo(x, y)

# Function to simulate receiving new coordinates
def update_coordinates():
    # Simulating new random coordinates (for demonstration)
    new_x = random.randint(0, 1920)  # Assuming a screen width of 1920
    new_y = random.randint(0, 1080)  # Assuming a screen height of 1080
    move_cursor(new_x, new_y)
    
    # Schedule the next update
    # root.after(1000, update_coordinates)  # Update every 1000 ms (1 second)

if __name__ == "__main__":
  # # Create the Tkinter window
  # root = tk.Tk()
  # root.title("Cursor Movement Example")

  # # Start updating coordinates immediately
  # update_coordinates()

  # # Start the Tkinter event loop
  # root.mainloop()
  move_cursor(945 * 2, 225 * 2)
