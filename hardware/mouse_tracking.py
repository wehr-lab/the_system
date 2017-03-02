from inputs import devices
from threading import Timer
import numpy as np

####################################
# PWM Setup

PWM_CLOCK       = "30000" # Sampling rate of Intan Board
SAMPLE_INTERVAL = 0.1     # Change the PWM duty cycle every 100ms

# Handle for getting mouse events
mouse = devices.mice[0]

# Convenience function to manipulate PWM
def set(property, value):
    try:
        f = open("/sys/class/rpi-pwm/pwm0/" + property, 'w')
        f.write(value)
        f.close()
    except:
        print("Error writing " + property + " to value: " + value)

# Use to set movement to "0"
def baseline():
    set("duty", "50")

# Initialize PWM
set("delayed", "0")
set("mode", "pwm")
set("frequency", PWM_CLOCK)
set("active", "1")

####################################
# Asynchonously update the pwm inside the while loop
# Recursively calls itself every SAMPLE_INTERVAL seconds.
def update_pwm():
    if movements:
        # Make a copy and clear movements so new movements aren't added here.
        move_local = movements
        global movements  # Yeah I know, but since this code is freestanding this is fine...
        movements = []

        # Average recent movements and set them.
        # You might be able to just write the values as fast as you can sample them
        # Writing this conservatively for the poor little pi in the meantime
        move_local = np.clip(move_local, -50, 50)
        move_local = move_local.mean(dtype=np.int)
        move_local += 50
        set('duty', str(move_local))
        Timer(SAMPLE_INTERVAL, update_pwm).start()

    else:
        baseline()
        Timer(SAMPLE_INTERVAL, update_pwm).start()

####################################
# Run the loop, monitor for movements.
movements = []
update_pwm()  # Call once to start the recursive update
while 1:
    events = mouse.read()
    movements.extend([event.state for event in events if event.code == "REL_Y"])


