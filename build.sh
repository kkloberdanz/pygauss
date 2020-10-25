set -e

rm -rf build dist || true

python3 setup.py sdist bdist_wheel
