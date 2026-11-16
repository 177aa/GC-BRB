"""Run with: python tests/test_baselines.py (requires python/requirements.txt)."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'python'/'run_same_input_baselines.py'
spec=importlib.util.spec_from_file_location('gg_baselines',path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng=np.random.default_rng(123)
groups=np.repeat(np.arange(60),2)
y=np.repeat(np.arange(60)%2,2)
X=rng.normal(size=(120,37));X[:,0]+=y
masks={'Train':groups<40,'Validation':(groups>=40)&(groups<50),'Test':groups>=50}
for name in ['xgb','mlp']:
    a=mod.fit_one(name,X,y,groups,masks,123,smoke=True)
    changed=y.copy();changed[masks['Test']]=1-changed[masks['Test']]
    b=mod.fit_one(name,X,changed,groups,masks,123,smoke=True)
    assert a['params']==b['params'] and a['threshold']==b['threshold']
    np.testing.assert_allclose(a['test_score'],b['test_score'],rtol=0,atol=1e-14)
    print(name+': forbidden-test-label invariance passed')
