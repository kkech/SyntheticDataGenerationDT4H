import os
import sys

# Tests run from any cwd: put the repo root (parent of tests/) on the path
# so `import pipeline...` resolves to this checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
