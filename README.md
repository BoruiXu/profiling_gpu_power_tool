## A tool for tracking the GPU power/energy consumption.

### USAGE:

```bash
mkdir build
cmake ..
make -j

./power\_profiling -g 0 -c "python your\_test.py"
```

### Parameter:
```bash
-t profiling time length

-i sample time interval

-m metirc (instant power: 1, average power: 2, energy: 3, GPU SM frequency: 4)

-g gpu index (e.g. 0,1)

-c "running command the program you want to test" 
```
