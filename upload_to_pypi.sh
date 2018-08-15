python3 setup.py sdist bdist_wheel
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
pip uninstall jeteloss
pip install jeteloss
cd examples/
python example1.py
