### Playground to test Tensorflow Eager

Install (MacOS):
```
conda env create -f environment_macos.yml -n tf_eager
source activate tf_eager
```
Run eager:
```
python main.py
```
Run graph:
1. Edit use_eager to False
2. Run:
```
python main.py
```
