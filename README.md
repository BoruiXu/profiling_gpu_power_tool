## A tool for tracking the GPU power/energy consumption.


### Compile
```bash
mkdir build
cmake ..
make -j
```
### USAGE:
```bash
./power_profiling -g 0 -c "python your_test.py"
```

### Parameter:
Must provide with -g and -c parameters.
```
-t profiling time length (seconds, default: 100 seconds)

-i sample time interval (microsecond, default: 100000 usec)

-m metirc (instant power: 1, average power: 2, energy: 3, GPU SM frequency: 4, default: 1)

-g gpu index (e.g. 0,1)

-c "running command the program you want to test" 
```
