# Sample Data

This directory is for sample/synthetic test data.

## Generating Sample Data

Run the demo script to generate sample visualizations:

```bash
cd semiconductor-test-toolkit
pip install -r requirements.txt
python examples/demo.py

Sample Data Format
The toolkit expects STDF V4 binary files from:

Advantest V93000 (SmarTest)
Teradyne J750/UltraFLEX (IG-XL)
Creating Synthetic STDF

# Use the test suite to generate minimal STDF files
python -m pytest tests/test_stdf_parser.py -v


## Add to Repo

```powershell
# Create folder if not exists
mkdir -p data/sample

# Create the file
# (paste content above into data/sample/README.md)

# Git add
git add data/sample/README.md
git commit -m "docs: Add sample data README with NDA guidelines"
