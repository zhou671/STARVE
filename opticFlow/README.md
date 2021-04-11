These files are to produce forward and backward optic flows.

`./consistencyChecker/`, `makeOptFlow.sh`, `run-deepflow.sh`
are copied from [manuelruder/artistic-videos](https://github.com/manuelruder/artistic-videos).
But bash commands are rewritten as Python codes with the subprocess package.

`deepflow2-static` is copied from [DeepFLow2](https://thoth.inrialpes.fr/src/deepflow/).

`deepmatching-static`, `viz.py` are copied form [DeepMatching](https://thoth.inrialpes.fr/src/deepmatching/).
The `print` finctions in `viz.py` is modified to accommodate for Python3.x.

To compile the files in `./consistencyChecker`, run
```bash
cd consistencyChecker
make
cd ..
```