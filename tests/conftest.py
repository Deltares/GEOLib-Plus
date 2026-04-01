import matplotlib

# Use non-interactive backend for headless environments (CI, no DISPLAY/Tk)
# This must run before any import of matplotlib.pyplot in the test process.
matplotlib.use("Agg")
