from machine import UART, Pin
import utime

# Initialize UART communication
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# Function to parse received data
def parse_moves(data):
    moves = []
    lines = data.split('\n')
    for line in lines:
        if line.strip():
            try:
                x, y = map(float, line.split(','))
                moves.append((x, y))
            except ValueError:
                print(f"Invalid move format: {line}")
    return moves

# Define GPIO pins for Motor 1 and Motor 2
motor1_step_pin = Pin(20, Pin.OUT)
motor1_dir_pin = Pin(21, Pin.OUT)
motor2_step_pin = Pin(17, Pin.OUT)
motor2_dir_pin = Pin(18, Pin.OUT)

# Function to control a single step on the motor
def step_motor(step_pin, delay=5000):
    step_pin.value(1)  # Send a pulse to the step pin
    utime.sleep_us(delay)  # Delay in microseconds
    step_pin.value(0)  # Set the pin low
    utime.sleep_us(delay)  # Delay after pulse

# Function to move the motors based on dx and dy simultaneously
def move_motors(dx, dy, steps_per_unit):
    # Calculate the number of steps required for each motor based on dx and dy
    steps_x = int(abs(dx) * steps_per_unit)
    steps_y = int(abs(dy) * steps_per_unit)

    # Set direction for motor 1 (X-axis)
    if dx < 0:  # Negative X direction
        print('-x')
        motor1_dir_pin.value(0)  # Clockwise
        motor2_dir_pin.value(0)
    elif dx > 0:  # Positive X direction
        print('x')
        motor1_dir_pin.value(1)  # Counterclockwise
        motor2_dir_pin.value(1)

    # Set direction for motor 2 (Y-axis)
    if dy > 0:  # Positive Y direction
        print('y')
        motor2_dir_pin.value(1)  # Clockwise
        motor1_dir_pin.value(0)
    elif dy < 0:  # Negative Y direction
        print('-y')
        motor2_dir_pin.value(0)  # Counterclockwise
        motor1_dir_pin.value(1) 
    x = steps_x
    y = steps_y
    # Synchronize steps for both motors
    while x > 0 or y > 0:
      
        step_motor(motor1_step_pin)
        x -= 1
        step_motor(motor2_step_pin)
        y -= 1

# Function to calculate changes in x and y between consecutive path points
def calculate_deltas(path):
    deltas = []
    for i in range(1, len(path)):
        prev_x, prev_y = path[i - 1]
        curr_x, curr_y = path[i]

        # Calculate changes in x and y with inverted y-axis
        delta_x = curr_x - prev_x
        delta_y = (curr_y - prev_y)

        deltas.append((delta_x, delta_y))
    return deltas

def execute_moves(deltas):
    # Execute movement based on calculated deltas
    for i, (dx, dy) in enumerate(deltas, start=1):
        print(f"Step {i}: Moving Δx = {dx:.2f}, Δy = {dy:.2f}")
        move_motors(dx, dy, steps_per_unit)
# Example path with adjusted coordinates
# sweep_path = [
#     (400, 400), (400, 500),(500,500)]
# Calculate deltas with inverted Y-axis
# deltas = calculate_deltas(sweep_path)

# Steps per distance unit (to be determined experimentally)
steps_per_unit = 1.117  # Adjust this as needed

buffer = ""
while True:
    if uart.any():
        buffer += uart.read().decode('utf-8')
        if '\n' in buffer:
            deltas = parse_moves(buffer)
            print(deltas)
            execute_moves(deltas)
            buffer = ""
