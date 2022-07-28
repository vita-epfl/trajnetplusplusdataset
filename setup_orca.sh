## Additional Requirements (ORCA)
wget https://github.com/sybrenstuvel/Python-RVO2/archive/master.zip
unzip master.zip
rm master.zip

## Setting up ORCA (steps provided in the Python-RVO2 repo)
cd Python-RVO2-main/
pip install cmake
pip install cython
python setup.py build
python setup.py install
cd ../
rm -rf Python-RVO2-main/
