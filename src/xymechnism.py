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
def step_motor(step_pin, delay=1000):
    """
    Sends a pulse to the step pin for a step in the motor.
    """
    step_pin.value(1)  # Send a pulse to the step pin
    utime.sleep_us(delay)  # Delay in microseconds
    step_pin.value(0)  # Set the pin low
    utime.sleep_us(delay)  # Delay after pulse

# Function to move the motors based on dx and dy
def move_motors(dx, dy, steps_per_unit):
    """
    Controls the motors to follow the specified delta_x and delta_y.
    steps_per_unit defines the number of motor steps per distance unit.
    """
    # Calculate the number of steps required for each motor based on dx and dy
    steps_x = int(abs(dx) * steps_per_unit)
    steps_y = int(abs(dy) * steps_per_unit)

    # Set direction for motor 1 (X-axis)
    if dx < 0:  # Negative X direction
        motor1_dir_pin.value(0)  # Clockwise
    elif dx > 0:  # Positive X direction
        motor1_dir_pin.value(1)  # Counterclockwise

    # Set direction for motor 2 (Y-axis) - Adjusted for downward positive Y direction
    if dy < 0:  # Negative Y direction (Upward movement)
        motor2_dir_pin.value(0)  # Clockwise (Upward)
    elif dy > 0:  # Positive Y direction (Downward movement)
        motor2_dir_pin.value(1)  # Counterclockwise (Downward)

    # Move both motors simultaneously by stepping both motors at the same time
    for _ in range(max(steps_x, steps_y)):  # Loop to move both motors together
        if steps_x > 0:
            step_motor(motor1_step_pin)  # Move motor 1 (X-axis)
            steps_x -= 1
        if steps_y > 0:
            step_motor(motor2_step_pin)  # Move motor 2 (Y-axis)
            steps_y -= 1

# Function to calculate changes in x and y between consecutive path points
def calculate_deltas(path):
    """
    Takes a list of (x, y) coordinates representing the sweep path and 
    returns a list of changes (delta_x, delta_y) between each point.
    """
    deltas = []
    for i in range(1, len(path)):
        prev_x, prev_y = path[i - 1]
        curr_x, curr_y = path[i]

        # Calculate changes in x and y
        delta_x = curr_x - prev_x
        delta_y = curr_y - prev_y

        deltas.append((delta_x, delta_y))
    return deltas

def execute_moves(deltas):
# Move along the calculated path
    for i, (dx, dy) in enumerate(deltas, start=1):
        print(f"Step {i}: Moving Δx = {dx:.2f}, Δy = {dy:.2f}")
        move_motors(dx, dy, steps_per_unit)
# Example sweep path (generated from previous code)
# sweep_path = [
#     (0, 0), (0, 298.54247966)
# ]

# Calculate deltas
# deltas = calculate_deltas(sweep_path)

# Print the results
# print("Changes in x and y between consecutive points:")
# for i, (dx, dy) in enumerate(deltas, start=1):
#     print(f"Step {i}: Δx = {dx:.2f}, Δy = {dy:.2f}")

# Steps per distance unit (to be determined experimentally)
steps_per_unit = 1.117  # Placeholder; adjust this value when the correct steps per unit is known

buffer = ""
while True:
    if uart.any():
        buffer += uart.read().decode('utf-8')
        if '\n' in buffer:
            moves = parse_moves(buffer)
            print(moves)
            execute_moves(moves)
            buffer = ""