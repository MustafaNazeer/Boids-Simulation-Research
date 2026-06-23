
## Steps

- Go to the simulation folder & Start the live demo:

```
cd ~/career/work/research/dr-patino/meetings/9-june && python3 -m sim.boids.live
```

A window opens and the flock animates in real time. The title bar shows the
current polarization (heading alignment) and the number of clusters as the run
goes.

## What you are looking at

- Blue dots: the boids (the flock).
- Grey circles: obstacles the flock steers around.
- Red star: the predator. When it gets close, the flock splits and scatters,
  then streams back together.

## To stop

Close the window. That ends the program and returns you to the terminal.

## If the window does not appear

The live view needs a graphical display and the TkAgg backend, which depends on
Pillow with ImageTk. On this machine that is already installed. If you ever move
to a fresh machine and see an ImageTk error, install Pillow into your user space
once:

```
pip install --user --break-system-packages pillow
```

Then run step 2 again. If there is no display at all (for example a remote
terminal with no screen), use the saved animation instead:

```
xdg-open ~/career/work/research/dr-patino/meetings/9-june/deliverables/flock_escape.gif
```
