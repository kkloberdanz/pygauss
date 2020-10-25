set -e

rm -rf build dist gauss.egg-info || true

python3 setup.py sdist bdist_wheel
