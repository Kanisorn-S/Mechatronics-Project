import serial
import time

# Replace '/dev/ttyACM0' with the correct port for your system (COMx on Windows)
pico_port = 'COM5'  # For Linux/Mac (or COMx on Windows)
baud_rate = 115200

# Initialize serial connection
ser = serial.Serial(pico_port, baud_rate)

# Send data to Pico and print the response
try:
    while True:
        data = "1"
        ser.write(data.encode())  # Send encoded string
        print("Sent:", data)
        
        # Wait for the Pico to respond
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()  # Read the response and decode it
            print("Received:", response)  # Print the echoed message from Pico
        
        time.sleep(1)  # Wait a bit before sending again
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    ser.close()  # Close the serial connection
