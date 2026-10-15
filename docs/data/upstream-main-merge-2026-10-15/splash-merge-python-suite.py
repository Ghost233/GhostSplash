import sys,unittest
from pathlib import Path
root=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash')
sys.path.insert(0,str(root))
from install import paths
paths.BINARY=root/'build/merge-upstream-20261015/splash'
print('Full Python discovery using merged native binary:',paths.BINARY,flush=True)
suite=unittest.defaultTestLoader.discover(str(root/'dev/tests'),pattern='test_*.py',top_level_dir=str(root))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
