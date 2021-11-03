
# 1. Update local packages for distribution
python3 -m pip install --upgrade setuptools wheel
python3 -m pip install --upgrade twine
#python3 -m pip install --user --upgrade setuptools wheel
#python3 -m pip install --user --upgrade twine

# 2. Open setup.py and change the version, e.g., version='1.0.3'.
python3 increment_setup_version.py

# 3. It is sometimes neccessary to delete the build/ and dist/ folder
rm -r build/*
rm -r dist/*

# 4. Create distribution packages on your local machine, and check
# the dist/ directory for the new version files
python3 setup.py sdist bdist_wheel

# 5. Upload the distribution files to https://pypi.org/ server
python3 -m twine upload dist/*

