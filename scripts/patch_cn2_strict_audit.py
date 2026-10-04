"""Patch an exported private V5 notebook locally; never executes it.

Rejects unknown CN2 source layouts. Keeps original weights, preprocessing,
all-window policy and 25% rank fusion. The private notebook is not published
by this tool. Output must be a new file; upload only to a separate private copy.
"""
import argparse
import ast
import json
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one patch target: {old[:80]!r}")
    return source.replace(old, new, 1)


def patch_branch(source, guard_source):
    source = replace_once(source,
        '24 evenly spaced 3-slice windows -> 224',
        'all acquired-span 3-slice windows (up to 42) -> 224')
    source = replace_once(source,
        '_C_SPAN, _C_IMG, _C_RES, _C_K = (0.15, 0.85), 336, 224, 24',
        '_C_SPAN, _C_IMG, _C_RES = (0.15, 0.85), 336, 224')
    source = replace_once(source,
        'def _c_volume(uid, series, series_root, reader):',
        'def _c_volume(uid, series, series_root, reader, return_audit=False):\n    _prep_started = _ct.perf_counter()')
    source = replace_once(source,
        '    return _ctorch.stack(wins)',
        '    result = _ctorch.stack(wins)\n    if return_audit:\n        return result, mask, cs, _ct.perf_counter() - _prep_started\n    return result')
    source = replace_once(source,
        "        return _c_volume(u, _series, _sr, _readers[t])",
        "        return _c_volume(u, _series, _sr, _readers[t], return_audit=True)")
    source = replace_once(source,
        "    _labs = [c for c in _sub.columns if c != 'StudyInstanceUID']",
        """    _labs = [c for c in _sub.columns if c != 'StudyInstanceUID']
    # Failures must not leave the pre-CN2 submission eligible for submission.
    _co.unlink('/kaggle/working/submission.csv')
    if _labs != _RSNA_LABELS or _sub.StudyInstanceUID.tolist() != _RSNA_TEST_IDS:
        raise RuntimeError('CN2 parent schema or UID order mismatch')
    if not _ctorch.cuda.is_available() or _ctorch.cuda.device_count() != 2:
        raise RuntimeError('CN2 requires the reference two CUDA devices')""")
    source = replace_once(source,
        "    _ndev = max(1, _ctorch.cuda.device_count())",
        "    _ndev = _ctorch.cuda.device_count()")
    start = source.index('    _nfail = 0\n')
    end = source.index('    def _rk(v):\n', start)
    old = source[start:end]
    if '0.05 * len(_ids)' not in old or '_okrate < 0.95' not in old:
        raise ValueError('Unrecognised V5 CN2 release block')
    new = '''    _model_ids = [_co.path.basename(f) for f in _c_files]
    _expected_models = [f'raptor_ft_cn2_224_f{i}_swa.pt' for i in range(5)] + [
        'raptor_ft_cn2_224_full_swa.pt', 'raptor_ft_cn2_224_student_s42_swa.pt',
        'raptor_ft_cn2_224_student_s7_swa.pt']
    if _model_ids != _expected_models:
        raise RuntimeError('CN2 checkpoint member identity mismatch')
    _audit = StrictCohortAudit(_ids, _model_ids, 12)
    _mi = None
    try:
        # Keep only four prepared studies pending, not the whole hidden cohort.
        with _CPool(4) as _pool:
            _pending = {i: _pool.submit(_prep, u) for i, u in enumerate(_ids[:4])}
            for si, uid in enumerate(_ids):
                _mi = None
                rsna_deadline('CN2 full-cohort inference')
                x, mask, centers, prep_seconds = _pending.pop(si).result()
                _audit.record_preparation(uid, mask, centers, prep_seconds)
                if tuple(x.shape) != (len(centers), 3, 224, 224):
                    raise RuntimeError('CN2 input window shape mismatch')
                if not bool(_ctorch.isfinite(x).all()):
                    raise RuntimeError('CN2 input contains NaN/Inf')
                if si + 4 < len(_ids):
                    _pending[si + 4] = _pool.submit(_prep, _ids[si + 4])
                with _ctorch.inference_mode():
                    for _mi, (m, d) in enumerate(_models):
                        inp = x[None].to(d).half()
                        _policy = assert_cn2_fp16(m, inp)
                        _audit.precision[_model_ids[_mi]] = _policy
                        _forward_started = _ct.perf_counter()
                        pred = _ctorch.sigmoid(m(inp).float()).cpu().numpy()[0]
                        _P[_mi, si] = _audit.record_forward(uid, _model_ids[_mi], pred,
                            _ct.perf_counter() - _forward_started)
                        del inp
                del x
        _audit.assert_complete(_P, _ids, _model_ids)
    except Exception as e:
        _audit.record_failure(uid if 'uid' in locals() else _ids[0], e,
                              _model_ids[_mi] if _mi is not None else None)
        raise
    finally:
        _receipt = _audit.summary()
        _receipt['branch_elapsed_seconds'] = _ct.time() - _c_t0
        rsna_json('/kaggle/working/diagnostics/cn2_strict_audit.json', _receipt)
    print('[cn2-audit] complete:', len(_ids), 'studies;', len(_audit.records),
          'successful forwards; fallback forbidden; see diagnostics/cn2_strict_audit.json', flush=True)
    rsna_save_predictions('cn2_raw', _ids, _P, _labs)
'''
    source = replace_once(source, old, new)
    source = replace_once(source,
        "        rm = _cn.nanmean([_rk(_P[mi, :, j]) for mi in range(len(_models))], axis=0)\n        ok = _cn.isfinite(_P[:, :, j]).all(0)\n        rm = _cn.where(ok, rm, r3)",
        "        rm = _cn.mean([_rk(_P[mi, :, j]) for mi in range(len(_models))], axis=0)")
    source = replace_once(source,
        "    _out.to_csv('/kaggle/working/submission.csv.tmp', index=False)",
        "    rsna_frame(_out, _ids, _labs, 'CN2 FINAL')\n    _out.to_csv('/kaggle/working/submission.csv.tmp', index=False)")
    # The helpers run in the notebook namespace, so remove their future import.
    guard_source = guard_source.replace('from __future__ import annotations\n', '')
    guard_source = guard_source.split('if __name__ == "__main__":')[0]
    source = guard_source + '\n' + source
    compile(source, '<cn2-audited>', 'exec')
    return source


def patch_notebook(notebook, guard_source):
    matches = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if "'cn2_arm_B_strict'" not in source:
            continue
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'compile'
                 and len(node.args) == 3 and isinstance(node.args[1], ast.Constant)
                 and node.args[1].value == 'cn2_arm_B_strict']
        if len(calls) != 1 or not isinstance(calls[0].args[0], ast.Constant):
            raise ValueError('Expected one literal CN2 compile source')
        matches.append((index, source, calls[0]))
    if len(matches) != 1:
        raise ValueError('Expected exactly one V5 CN2 cell')
    index, source, call = matches[0]
    patched = patch_branch(call.args[0].value, guard_source)
    old_literal = ast.get_source_segment(source, call.args[0])
    source = replace_once(source, old_literal, repr(patched))
    compile(source, '<patched-cell>', 'exec')
    notebook['cells'][index]['source'] = source.splitlines(keepends=True)
    # Old outputs must not masquerade as evidence of a new audited run.
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    guard = Path(__file__).with_name('inference_full_volume_guard.py').read_text()
    patched = patch_notebook(json.loads(args.input.read_text()), guard)
    with args.output.open('x') as handle:
        json.dump(patched, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(f'Prepared {args.output}; no GPU execution or submission performed.')


if __name__ == '__main__':
    main()
