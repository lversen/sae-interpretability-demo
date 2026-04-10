"""
This script tests the interactive plotting capability with a simple example.
Run this first to verify your matplotlib setup can display plots interactively.
"""
import matplotlib
# Force matplotlib to use an interactive backend
matplotlib.use('TkAgg')  # Try this first - most compatible across platforms

import numpy as np
import matplotlib.pyplot as plt
import time

# Enable interactive mode
plt.ion()

# Create a simple figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))
ax.set_title("Interactive Plot Test")

# Animation loop to simulate training
for i in range(10):
    # Update the data
    line.set_ydata(np.sin(x + i/5))
    
    # Redraw the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Show the plot without blocking
    plt.show(block=False)
    
    # Give the GUI time to update
    plt.pause(0.5)
    
    print(f"Step {i+1}/10 - If you can see the plot updating, interactive mode is working!")

print("\nTest completed!")
print("If you did NOT see a plot window with an updating sine wave, try these troubleshooting steps:")
print("1. Make sure you have a working display environment")
print("2. Try changing the matplotlib backend in the code to one of these:")
print("   - 'Qt5Agg' (requires PyQt5)")
print("   - 'TkAgg' (requires Tkinter)")
print("   - 'GTK3Agg' (requires GTK)")
print("   - 'macosx' (for MacOS)")
print("3. Install the required dependencies for your chosen backend")
print("4. If using a remote server, ensure you have X11 forwarding enabled")

# Keep the window open at the end
plt.ioff()
plt.show()