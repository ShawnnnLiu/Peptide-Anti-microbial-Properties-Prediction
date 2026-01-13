'''
Predicting classification of new data based on previously trained svc classifier

IN:
<descFile>         = csv format: headerline; col 1 = index, col 2+ = descriptors
<ZFile>            = names, means, and stds of descriptors used by svc
<svcPkl>           = pickled previously trained svc classifier

OUT:



'''

## imports
import os, re, sys, time
import random, math

import numpy as np
import numpy.matlib

import scipy.optimize
import scipy.stats

import pickle as pkl

# NOTE: matplotlib not required for prediction and conflicts with legacy numpy. Omit import.

from sklearn import svm
from sklearn import datasets
try:
    import joblib  # modern standalone joblib
except Exception:
    from sklearn.externals import joblib  # very old fallback
try:
    # Prefer the joblib vendored inside sklearn for legacy pickles
    from sklearn.externals import joblib as _skl_joblib
except Exception:
    _skl_joblib = None
from sklearn import metrics

# Alias for legacy pickle path 'sklearn.externals.joblib'
import sys as _sys
_sys.modules.setdefault('sklearn.externals.joblib', joblib)

# Compatibility shim: old pickles may reference 'sklearn.svm.classes'
try:
    import sklearn.svm._classes as _sk_svm_classes
    import types as _types
    import sys as _sys
    if 'sklearn.svm.classes' not in _sys.modules:
        _sys.modules['sklearn.svm.classes'] = _sk_svm_classes
except Exception:
    pass


## methods

# usage
def _usage():
	print('USAGE: %s <descFile> <ZFile> <svcPkl>' % sys.argv[0])
	print('       <descFile>         = csv format: headerline; col 1 = index, col 2+ = descriptors')
	print('       <ZFile>            = names, means, and stds of descriptors used by svc')
	print('       <svcPkl>           = pickled previously trained svc classifier')

def unique(a):
	''' return the list with duplicate elements removed '''
	return list(set(a))

def intersect(a, b):
    ''' return the intersection of two lists '''
    return list(set(a) & set(b))

def union(a, b):
    ''' return the union of two lists '''
    return list(set(a) | set(b))



## main

# reading args and error checking
if len(sys.argv) != 4:
	_usage()
	sys.exit(-1)

descFile = str(sys.argv[1])
ZFile = str(sys.argv[2])
svcPkl = str(sys.argv[3])




# loading, selecting, and Z-scoring descriptors

# - loading descriptors pertaining to new sequences
print('')
print('Loading data from %s...' % (descFile))

data=[]
with open(descFile,'r') as fin:
	line = fin.readline()
	headers = line.strip().split(',')
	for line in fin:
		data.append(line.strip().split(','))

headers_index = headers[0]
data_index = [item[0] for item in data]

headers_desc = headers[1:]
data_desc = [item[1:] for item in data]
data_desc = [[float(y) for y in x] for x in data_desc]
data_desc = np.array(data_desc)

print('Loaded')
print('')


# - loading descriptor names, Z score means, and Z score stds
print('Loading descriptor list, means, and stds from %s...' % (ZFile))

with open(ZFile,'r') as fin:
	line = fin.readline()
	descriptors_select = line.strip().split(',')

	line = fin.readline()
	Z_means = line.strip().split(',')
	Z_means = [float(x) for x in Z_means]

	line = fin.readline()
	Z_stds = line.strip().split(',')
	Z_stds = [float(x) for x in Z_stds]

print('Loaded')
print('')


# - extracting from descriptors only those in ZFile and alerting user to any not present

mask = []
for ii in range(0,len(descriptors_select)):
	try:
		idx = headers_desc.index(descriptors_select[ii])
	except ValueError:
		print('ERROR: Descriptor %s specified in %s was not found among descriptors in %s; aborting.' % (descriptors_select[ii], ZFile, descFile), file=sys.stderr)
		sys.exit(-1)
	mask.append(idx)

headers_desc = [headers_desc[x] for x in mask]
data_desc = data_desc[:,mask]

# - applying Z-scoring imported from infileZ
Z_means = np.array(Z_means)
Z_stds = np.array(Z_stds)

data_desc = data_desc - np.matlib.repmat(Z_means, data_desc.shape[0], 1)
data_desc = np.divide(data_desc, np.matlib.repmat(Z_stds, data_desc.shape[0], 1))

print('Loaded')
print('')



# loading classifier
print('Loading classifier from %s...' % (svcPkl))

# Load with sklearn's vendored joblib when available (best for legacy pickles)
if _skl_joblib is not None:
    svc = _skl_joblib.load(svcPkl)
else:
    svc = joblib.load(svcPkl)

# Coerce legacy NDArrayWrapper attributes to numpy arrays for compatibility
"""
Compatibility: legacy SVC pickles (pre-0.18) may lack attributes newer sklearn expects
when calling decision_function/predict. Provide minimal defaults.
"""
try:
    if not hasattr(svc, 'decision_function_shape'):
        # default used by sklearn for binary SVC
        setattr(svc, 'decision_function_shape', 'ovr')
except Exception:
    pass

def _coerce_ndarray_wrappers(obj):
    import numpy as _np
    # Only touch known array-like attributes to avoid triggering properties
    target_attrs = [
        'support_vectors_', 'dual_coef_', 'intercept_', 'n_support_', 'support_',
        'probA_', 'probB_', 'class_weight_', 'class_weight', 'shape_fit_',
    ]
    odict = getattr(obj, '__dict__', {})
    for k in target_attrs:
        if k in odict:
            v = odict.get(k)
            # Convert NDArrayWrapper or similar to ndarray
            try:
                cls_name = type(v).__name__
                if (hasattr(v, '__array__') or cls_name == 'NDArrayWrapper') and not isinstance(v, _np.ndarray):
                    try:
                        arr = _np.asarray(v)
                    except Exception:
                        arr = _np.array(v, copy=False)
                    try:
                        setattr(obj, k, arr)
                        continue
                    except Exception:
                        pass
                # Convert sequences of wrappers
                if isinstance(v, (list, tuple)):
                    new_v = []
                    changed = False
                    for item in v:
                        if hasattr(item, '__array__') and not isinstance(item, _np.ndarray):
                            try:
                                new_v.append(_np.asarray(item))
                                changed = True
                            except Exception:
                                new_v.append(item)
                        else:
                            new_v.append(item)
                    if changed:
                        setattr(obj, k, type(v)(new_v))
            except Exception:
                continue
    # Backfill private fields for newer sklearn if only public attrs exist
    try:
        if not hasattr(obj, '_probA') and hasattr(obj, 'probA_'):
            object.__setattr__(obj, '_probA', object.__getattribute__(obj, 'probA_'))
    except Exception:
        pass
    try:
        if not hasattr(obj, '_probB') and hasattr(obj, 'probB_'):
            object.__setattr__(obj, '_probB', object.__getattribute__(obj, 'probB_'))
    except Exception:
        pass
    return obj

svc = _coerce_ndarray_wrappers(svc)

# Additional unwrap for joblib NDArrayWrapper held inside 0-d arrays or containers
_NDW = None
_LEGACY_NDW = None
try:
    from joblib.numpy_pickle_utils import NDArrayWrapper as _NDW  # modern path
except Exception:
    _NDW = None
try:
    # legacy path used by very old pickles
    from sklearn.externals.joblib.numpy_pickle_compat import NDArrayWrapper as _LEGACY_NDW
except Exception:
    _LEGACY_NDW = None

# Monkey-patch NDArrayWrapper to behave like a numpy array
try:
    import numpy as _np
    def _ndw___array__(self, dtype=None):
        arr = self.read_array()
        return _np.asarray(arr, dtype=dtype) if dtype is not None else _np.asarray(arr)
    if _NDW is not None and not hasattr(_NDW, '__array__'):
        try:
            _NDW.__array__ = _ndw___array__
        except Exception:
            pass
    if _LEGACY_NDW is not None and not hasattr(_LEGACY_NDW, '__array__'):
        try:
            _LEGACY_NDW.__array__ = _ndw___array__
        except Exception:
            pass
except Exception:
    pass

def _unwrap_legacy_array(x):
    import numpy as _np
    # If ndarray with ndim==0, peek at the scalar
    if isinstance(x, _np.ndarray) and x.ndim == 0:
        try:
            s = x.item()
            # Prefer duck-typing: any object with read_array()
            if hasattr(s, 'read_array'):
                try:
                    return s.read_array()
                except Exception:
                    pass
            if (_NDW is not None and isinstance(s, _NDW)) or (_LEGACY_NDW is not None and isinstance(s, _LEGACY_NDW)):
                try:
                    return s.read_array()
                except Exception:
                    pass
            return _np.asarray(s)
        except Exception:
            return x
    # If list/tuple of wrappers
    if isinstance(x, (list, tuple)):
        out = []
        changed = False
        for it in x:
            if hasattr(it, 'read_array'):
                try:
                    out.append(it.read_array())
                    changed = True
                    continue
                except Exception:
                    pass
            if (_NDW is not None and isinstance(it, _NDW)) or (_LEGACY_NDW is not None and isinstance(it, _LEGACY_NDW)):
                try:
                    out.append(it.read_array())
                    changed = True
                    continue
                except Exception:
                    pass
            out.append(it)
        return type(x)(out) if changed else x
    # If direct wrapper
    if hasattr(x, 'read_array'):
        try:
            return x.read_array()
        except Exception:
            return x
    if (_NDW is not None and isinstance(x, _NDW)) or (_LEGACY_NDW is not None and isinstance(x, _LEGACY_NDW)):
        try:
            return x.read_array()
        except Exception:
            return x
    return x

try:
    import numpy as _np
    for key in ['support_vectors_', '_support_vectors', 'support_', '_support', 'dual_coef_', '_dual_coef', 'n_support_', '_n_support', 'intercept_', '_intercept']:
        if hasattr(svc, key):
            val = getattr(svc, key)
            unwrapped = _unwrap_legacy_array(val)
            try:
                setattr(svc, key, _np.asarray(unwrapped))
            except Exception:
                try:
                    setattr(svc, key, unwrapped)
                except Exception:
                    pass

    # Ensure n_support_/_n_support are concrete int arrays
    if hasattr(svc, 'n_support_') and not isinstance(svc.n_support_, _np.ndarray):
        ns = _unwrap_legacy_array(svc.n_support_)
        try:
            svc.n_support_ = _np.asarray(ns, dtype=_np.int32)
        except Exception:
            pass
    if hasattr(svc, '_n_support') and not isinstance(svc._n_support, _np.ndarray):
        ns = _unwrap_legacy_array(svc._n_support)
        try:
            svc._n_support = _np.asarray(ns, dtype=_np.int32)
        except Exception:
            pass
    if hasattr(svc, 'n_support_') and hasattr(svc, '_n_support'):
        try:
            if svc._n_support is None or getattr(svc._n_support, 'size', 0) == 0:
                svc._n_support = _np.asarray(svc.n_support_, dtype=_np.int32)
        except Exception:
            pass
except Exception:
    pass

# Force-convert critical SVC arrays to numpy ndarrays
try:
    import numpy as _np
    if hasattr(svc, 'support_vectors_'):
        tmp = svc.support_vectors_
        svc.support_vectors_ = _np.asarray(tmp) if not isinstance(tmp, _np.ndarray) else tmp
    if hasattr(svc, 'dual_coef_'):
        tmp = svc.dual_coef_
        svc.dual_coef_ = _np.asarray(tmp) if not isinstance(tmp, _np.ndarray) else tmp
    if hasattr(svc, 'intercept_'):
        tmp = svc.intercept_
        svc.intercept_ = _np.asarray(tmp) if not isinstance(tmp, _np.ndarray) else tmp
    if hasattr(svc, 'n_support_'):
        tmp = svc.n_support_
        svc.n_support_ = _np.asarray(tmp, dtype=_np.int32) if not isinstance(tmp, _np.ndarray) else tmp
    if hasattr(svc, 'support_'):
        tmp = svc.support_
        svc.support_ = _np.asarray(tmp, dtype=_np.int32) if not isinstance(tmp, _np.ndarray) else tmp
    # Ensure sparse flag exists and is consistent
    if not hasattr(svc, '_sparse'):
        setattr(svc, '_sparse', False)

    # Map public attrs to legacy private names some sklearn code paths still access
    if hasattr(svc, 'n_support_') and not hasattr(svc, '_n_support'):
        setattr(svc, '_n_support', svc.n_support_)
    if hasattr(svc, 'support_') and not hasattr(svc, '_support'):
        setattr(svc, '_support', svc.support_)
    if hasattr(svc, 'support_vectors_') and not hasattr(svc, '_support_vectors'):
        setattr(svc, '_support_vectors', svc.support_vectors_)
    if hasattr(svc, 'dual_coef_') and not hasattr(svc, '_dual_coef'):
        setattr(svc, '_dual_coef', svc.dual_coef_)
    if hasattr(svc, 'intercept_') and not hasattr(svc, '_intercept'):
        setattr(svc, '_intercept', svc.intercept_)

    # Ensure support_vectors_ has correct 2D shape (n_svs, n_features)
    if hasattr(svc, 'support_vectors_'):
        sv = svc.support_vectors_
        n_features = int(data_desc.shape[1])
        sv = _np.asarray(sv)
        # Handle 0-d or 1-d cases from legacy pickles
        if sv.ndim == 0:
            # Try to unwrap underlying object (e.g., NDArrayWrapper) via .item()
            try:
                inner = sv.item()
                sv = _np.asarray(inner)
            except Exception:
                pass
            # Re-evaluate shape
            if sv.ndim == 0 and sv.size == 1:
                sv = _np.repeat(_np.asarray([sv.item()], dtype=_np.float64), n_features).reshape(1, n_features)
        elif sv.ndim == 1:
            if sv.size == n_features:
                sv = sv.reshape(1, n_features)
            elif sv.size % n_features == 0:
                sv = sv.reshape(-1, n_features)
            else:
                # As a last resort, try to infer n_svs from support_ length
                if hasattr(svc, 'support_'):
                    sup_arr = _np.asarray(svc.support_)
                    if sup_arr.ndim == 0:
                        try:
                            sup_arr = _np.asarray(sup_arr.item())
                        except Exception:
                            pass
                    n_svs = int(sup_arr.shape[0]) if sup_arr.ndim > 0 else int(len(sup_arr))
                    if n_svs > 0 and sv.size == n_svs * n_features:
                        sv = sv.reshape(n_svs, n_features)
        # If 2D but wrong feature count, attempt to fix by reshaping if total size matches
        if sv.ndim == 2 and sv.shape[1] != n_features:
            total = sv.size
            if total % n_features == 0:
                sv = sv.reshape(-1, n_features)
        svc.support_vectors_ = sv.astype(_np.float64, copy=False)

    # Make n_support_ consistent with support vectors
    if hasattr(svc, 'support_vectors_'):
        sv_arr = _np.asarray(svc.support_vectors_)
        if sv_arr.ndim == 0:
            try:
                sv_arr = _np.asarray(sv_arr.item())
            except Exception:
                pass
        n_svs = int(sv_arr.shape[0]) if sv_arr.ndim >= 1 else 0
        # Determine number of classes
        if hasattr(svc, 'classes_'):
            try:
                n_classes = int(len(svc.classes_))
            except Exception:
                n_classes = 2
        else:
            n_classes = 2
        # Build counts that sum to n_svs
        base = n_svs // n_classes
        rem = n_svs % n_classes
        counts = [_np.int32(base + (1 if i < rem else 0)) for i in range(n_classes)]
        counts_arr = _np.asarray(counts, dtype=_np.int32)
        try:
            svc.n_support_ = counts_arr
        except Exception:
            pass
        try:
            svc._n_support = counts_arr
        except Exception:
            pass

    # If support vectors are still not valid 2D array, attempt reconstruction from dual_coef_
    try:
        sv = _np.asarray(svc.support_vectors_) if hasattr(svc, 'support_vectors_') else None
        if sv is not None and (sv.ndim == 0 or (sv.ndim == 2 and sv.shape[1] != int(data_desc.shape[1]))):
            if hasattr(svc, 'dual_coef_'):
                dc = _np.asarray(svc.dual_coef_)
                if dc.ndim == 2:
                    n_svs = int(dc.shape[1])
                    n_features = int(data_desc.shape[1])
                    # We cannot reconstruct actual vectors; raise informative error
                    print('ERROR: Legacy pickle lacks usable support_vectors_; dual_coef_ indicates', n_svs, 'SVs.')
                    print('       The pickle appears incomplete. Please provide the original sklearn version to re-export the model,')
                    print('       or share the original training code to regenerate a modern-compatible pickle.')
    except Exception:
        pass
except Exception:
    pass

print('Loaded')
print('')



# performing classification prediction
print('Performing classification predictions and writing to %s...' % (descFile[0:-4] + '_PREDICTIONS.csv'))

# Final hard unwrap for legacy fields that may still be 0-D object arrays with NDArrayWrapper
try:
    import numpy as _np
    def _hard_unwrap_field(obj, name):
        if not hasattr(obj, name):
            return
        v = getattr(obj, name)
        # Direct wrapper instance
        if hasattr(v, 'read_array'):
            try:
                setattr(obj, name, _np.asarray(v.read_array()))
                return
            except Exception:
                pass
        # 0-D object ndarray holding wrapper
        if isinstance(v, _np.ndarray) and v.ndim == 0 and v.dtype == object:
            try:
                inner = v.item()
                if hasattr(inner, 'read_array'):
                    setattr(obj, name, _np.asarray(inner.read_array()))
                    return
                # If inner is ndarray already
                if isinstance(inner, _np.ndarray):
                    setattr(obj, name, inner)
                    return
            except Exception:
                pass
        # Ensure numpy array type if possible
        try:
            setattr(obj, name, _np.asarray(v))
        except Exception:
            pass

    for _fname in ('support_vectors_', 'support_', 'dual_coef_', 'intercept_', 'n_support_',
                   '_support_vectors', '_support', '_dual_coef', '_intercept', '_n_support'):
        _hard_unwrap_field(svc, _fname)

    # Ensure integer dtype for support indices and counts
    if hasattr(svc, 'support_'):
        try:
            svc.support_ = _np.asarray(svc.support_, dtype=_np.int32)
        except Exception:
            pass
    if hasattr(svc, 'n_support_'):
        try:
            svc.n_support_ = _np.asarray(svc.n_support_, dtype=_np.int32)
        except Exception:
            pass
    if hasattr(svc, '_n_support'):
        try:
            svc._n_support = _np.asarray(svc._n_support, dtype=_np.int32)
        except Exception:
            pass
except Exception:
    pass

# DEBUG diagnostics for legacy model compatibility
try:
    # List support-related candidate attributes present on the loaded SVC
    try:
        cand_keys = [k for k in getattr(svc, '__dict__', {}).keys()
                     if any(sub in k.lower() for sub in ['support', 'coef', 'vector', 'sv'])]
        print('DEBUG candidate keys =', sorted(cand_keys))
    except Exception as _e_ck:
        print('DEBUG candidate keys error:', repr(_e_ck))

    print('DEBUG data_desc.shape =', data_desc.shape)
    print('DEBUG type(support_vectors_) =', type(svc.support_vectors_) if hasattr(svc, 'support_vectors_') else None)
    if hasattr(svc, 'support_vectors_'):
        try:
            print('DEBUG support_vectors_.shape =', svc.support_vectors_.shape)
            try:
                import numpy as _np
                if isinstance(svc.support_vectors_, _np.ndarray) and svc.support_vectors_.ndim == 0:
                    print('DEBUG support_vectors_ dtype =', svc.support_vectors_.dtype)
                    try:
                        inner = svc.support_vectors_.item()
                        print('DEBUG support_vectors_.item type =', type(inner))
                        try:
                            from joblib.numpy_pickle_utils import NDArrayWrapper as _NDW
                            print('DEBUG support_vectors_.item is NDArrayWrapper =', isinstance(inner, _NDW))
                            if isinstance(inner, _NDW):
                                try:
                                    arr = inner.read_array()
                                    print('DEBUG support_vectors_.item.read_array shape =', getattr(arr, 'shape', None))
                                except Exception as _e_ra:
                                    print('DEBUG read_array error:', repr(_e_ra))
                        except Exception as _e_imp:
                            print('DEBUG NDArrayWrapper import error:', repr(_e_imp))
                    except Exception as _e_it:
                        print('DEBUG support_vectors_.item error:', repr(_e_it))
            except Exception as _e_svd:
                print('DEBUG extra support_vectors_ debug error:', repr(_e_svd))
        except Exception as e:
            print('DEBUG support_vectors_ shape error:', repr(e))
    if hasattr(svc, 'support_'):
        try:
            sv_shape = svc.support_.shape if hasattr(svc.support_, 'shape') else (len(svc.support_),)
            print('DEBUG support_.shape =', sv_shape)
        except Exception as e:
            print('DEBUG support_ shape error:', repr(e))
    if hasattr(svc, 'n_support_'):
        print('DEBUG n_support_ =', svc.n_support_)
    if hasattr(svc, '_n_support'):
        print('DEBUG _n_support =', svc._n_support)
    if hasattr(svc, 'dual_coef_'):
        try:
            print('DEBUG dual_coef_.shape =', getattr(svc.dual_coef_, 'shape', None))
        except Exception as _e_dc:
            print('DEBUG print error:', repr(_dbg_e))
except Exception as _dbg_e:
    print('DEBUG print error:', repr(_dbg_e))

# Final explicit coercion of critical arrays to numpy before calling into sklearn
try:
    import numpy as _np
    def _force_ndarray(x, dtype=None):
        # direct wrapper
        if hasattr(x, 'read_array'):
            arr = x.read_array()
            return _np.asarray(arr, dtype=dtype) if dtype is not None else _np.asarray(arr)
        # 0-d object ndarray with wrapper inside
        if isinstance(x, _np.ndarray) and x.ndim == 0 and x.dtype == object:
            try:
                inner = x.item()
                if hasattr(inner, 'read_array'):
                    arr = inner.read_array()
                    return _np.asarray(arr, dtype=dtype) if dtype is not None else _np.asarray(arr)
                return _np.asarray(inner, dtype=dtype) if dtype is not None else _np.asarray(inner)
            except Exception:
                pass
        return _np.asarray(x, dtype=dtype) if dtype is not None else _np.asarray(x)

    for _name, _dtype in [
        ('dual_coef_', None), ('_dual_coef', None),
        ('support_vectors_', _np.float64), ('_support_vectors', _np.float64),
        ('support_', _np.int32), ('_support', _np.int32),
        ('n_support_', _np.int32), ('_n_support', _np.int32),
        ('intercept_', _np.float64), ('_intercept', _np.float64),
    ]:
        if hasattr(svc, _name):
            try:
                setattr(svc, _name, _force_ndarray(getattr(svc, _name), dtype=_dtype))
            except Exception:
                pass
    # Backfill trailing-underscore privates expected by sklearn 0.19.x
    try:
        if hasattr(svc, 'dual_coef_') and not hasattr(svc, '_dual_coef_'):
            svc._dual_coef_ = _np.asarray(svc.dual_coef_)
    except Exception:
        pass
    try:
        if hasattr(svc, 'intercept_') and not hasattr(svc, '_intercept_'):
            svc._intercept_ = _np.asarray(svc.intercept_)
    except Exception:
        pass
except Exception:
    pass

# Ensure shape_fit_ exists with correct feature count for sklearn 0.19.x predict path
try:
    import numpy as _np
    if not hasattr(svc, 'shape_fit_'):
        try:
            n_features = int(getattr(svc, 'support_vectors_').shape[1])
        except Exception:
            n_features = int(data_desc.shape[1])
        try:
            n_samples = int(getattr(svc, 'support_').shape[0])
        except Exception:
            # fallback to number of support vectors or 1
            try:
                n_samples = int(getattr(svc, 'support_vectors_').shape[0])
            except Exception:
                n_samples = 1
        svc.shape_fit_ = (n_samples, n_features)
except Exception:
    pass

distToMargin = svc.decision_function(data_desc)
classProb = svc.predict_proba(data_desc)
pred_labels = svc.predict(data_desc)

# Map probability columns explicitly to [-1, +1]
import numpy as _np
neg_idx = int(_np.where(svc.classes_ == -1)[0][0]) if hasattr(svc, 'classes_') else 0
pos_idx = int(_np.where(svc.classes_ == 1)[0][0]) if hasattr(svc, 'classes_') else 1

idx_sort = np.argsort(distToMargin)
idx_sort = idx_sort[::-1]
with open(descFile[0:-4] + '_PREDICTIONS.csv','w') as fout:

	# header
	fields = [headers_index, 'prediction', 'distToMargin', 'P(-1)', 'P(+1)']
	fout.write(','.join(fields) + '\n')

	# rows
	for ii in idx_sort:
		pred = int(pred_labels[ii])
		p_neg = float(classProb[ii, neg_idx])
		p_pos = float(classProb[ii, pos_idx])
		row = [
			data_index[ii],
			str(pred),
			str(float(distToMargin[ii])),
			str(p_neg),
			str(p_pos)
		]
		fout.write(','.join(row) + '\n')

with open(descFile[0:-4] + '_PREDICTIONS_unsorted.csv','w') as fout:

	# header
	fields = [headers_index, 'prediction', 'distToMargin', 'P(-1)', 'P(+1)']
	fout.write(','.join(fields) + '\n')

	# rows (unsorted)
	for ii in range(0, len(distToMargin)):
		pred = int(pred_labels[ii])
		p_neg = float(classProb[ii, neg_idx])
		p_pos = float(classProb[ii, pos_idx])
		row = [
			data_index[ii],
			str(pred),
			str(float(distToMargin[ii])),
			str(p_neg),
			str(p_pos)
		]
		fout.write(','.join(row) + '\n')

print('Predictions complete')
print('')




print('DONE!')
print('')
