#/bin/bash
pip uninstall FeatureSpaceTree
rm ./dist/FeatureSpaceTree-.01.tar.gz
rm ./dist/FeatureSpaceTree-.01-py2.7.egg
python setup.py sdist bdist_egg
pip install ./dist/FeatureSpaceTree-.01.tar.gz 

python setup.py install


# To perform a succesful Re-instalation
pip uninstall FeatureSpaceTree
python setup.py sdist
pip install ./dist/FeatureSpaceTree-.01.tar.gz --upgrade



# Wheel 
pip uninstall FeatureSpaceTree
python setup.py bdist_wheel
pip install FeatureSpaceTree-1-py2-none-any.whl 
