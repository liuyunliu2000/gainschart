# gainschart module for predictive modeling



# alternatively by manual upload to testpypi
python -m pip install --upgrade pip
pip install --upgrade twine
python3 -m pip install --upgrade build
python -m build
cd gith
ls
cd github
ls
cd gainschart
ls
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
cd ..
pip install -i https://test.pypi.org/simple/ gainschart==0.1.0

## question about the naming of it folder structure
import gainschart.gs as g
g.combineimages()

why it works this way:
import pandas as pd
