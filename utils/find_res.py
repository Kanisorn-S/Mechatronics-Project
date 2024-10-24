from screeninfo import get_monitors

def get_display_resolutions():
    monitors = get_monitors()
    for i, monitor in enumerate(monitors):
        print(f"Monitor {i + 1}:")
        print(f"    Width: {monitor.width}")
        print(f"    Height: {monitor.height}")
        print(f"    Display name: {monitor.name}")

if __name__ == "__main__":
    get_display_resolutions()
