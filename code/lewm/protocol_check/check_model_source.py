"""Compare the loaded model definition with the selected released source."""
from pathlib import Path
import ast, json

R = Path(__file__).parent

def methods(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    result[node.name + '.' + method.name] = ast.dump(method, include_attributes=False)
    return result

rows = []
for task in ['cube', 'pusht', 'reacher', 'tworoom']:
    assets = json.loads((R / f'reference/assets_{task}.json').read_text())
    runtime = Path(assets['checkpoint_source']).parent
    release = R / 'official_source'
    a, b = methods(release / 'module.py'), methods(runtime / 'module.py')
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    assert a.keys() == b.keys() and changed == ['SIGReg.forward']
    image_preprocessor_equal = methods(release / 'utils.py')['get_img_preprocessor'] == methods(runtime / 'utils.py')['get_img_preprocessor']
    jepa_equal = (runtime / 'jepa.py').read_bytes() == (release / 'jepa.py').read_bytes()
    assert jepa_equal and image_preprocessor_equal and assets['serialization_exact']
    rows.append({'task': task, 'model_class': assets['checkpoint_class'],
                 'state_tensors': assets['state_tensors'], 'serialization_exact': True,
                 'jepa_file_equal': jepa_equal,
                 'identical_inference_module_methods': sorted(k for k in a if k != 'SIGReg.forward'),
                 'image_preprocessor_function_equal': image_preprocessor_equal,
                 'nonidentical_source_files': ['module.py', 'utils.py'],
                 'differences': ['SIGReg.forward chooses proj.device instead of cuda when sampling training regularizer projections; it is not called by JEPA.get_cost.',
                                 'utils.py replaces the training-column normalizer closure by an equivalent callable class and changes checkpoint-saving callbacks. Evaluation uses the unchanged image preprocessor and the pretrained action StandardScaler.'],
                 'dataset_name': assets['dataset_name'], 'dataset_bytes': assets['dataset_bytes']})
(R / 'lewm_checks/model_source_check.json').write_text(json.dumps(rows, indent=2))
print(json.dumps({'tasks': len(rows), 'inference_paths_equal': True,
                  'training_utility_source_differences_recorded': True}))
