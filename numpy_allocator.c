#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <string.h>

typedef struct {
	PyObject_HEAD
	void **ptr;
} PyCFuncPtrObject;

typedef struct {
	PyCFuncPtrObject *calloc;
	PyCFuncPtrObject *free;
	PyCFuncPtrObject *malloc;
	PyCFuncPtrObject *realloc;
} PyDataMem_Funcs;

static void tp_finalize(PyObject *cls) {
	PyObject_DelAttrString(cls, "_handler_");
}

typedef void *(PyDataMem_ReallocFunc)(void *ptr, size_t new_size);

static void *safe_realloc(void *ctx, void *ptr, size_t new_size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *new_ptr = ((PyDataMem_ReallocFunc *) *((PyDataMem_Funcs *) ctx)->realloc->ptr)(ptr, new_size);
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return new_ptr;
}

static void *default_realloc(void *ctx, void *ptr, size_t new_size) {
	return realloc(ptr, new_size);
}

typedef void *(PyDataMem_MallocFunc)(size_t size);

static void *safe_malloc(void *ctx, size_t size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *ptr = ((PyDataMem_MallocFunc *) *((PyDataMem_Funcs *) ctx)->malloc->ptr)(size);
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return ptr;
}

static void *default_malloc(void *ctx, size_t size) {
	return malloc(size);
}

typedef void (PyDataMem_FreeFunc)(void *ptr, size_t size);

static void safe_free(void *ctx, void *ptr, size_t size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	((PyDataMem_FreeFunc *) *((PyDataMem_Funcs *) ctx)->free->ptr)(ptr, size);
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
}

static void default_free(void *ctx, void *ptr, size_t size) {
	free(ptr);
}

typedef void *(PyDataMem_CallocFunc)(size_t nelem, size_t elsize);

static void *safe_calloc(void *ctx, size_t nelem, size_t elsize) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *ptr = ((PyDataMem_CallocFunc *) *((PyDataMem_Funcs *) ctx)->calloc->ptr)(nelem, elsize);
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return ptr;
}

static void *default_calloc(void *ctx, size_t nelem, size_t elsize) {
	return calloc(nelem, elsize);
}

static void handler_destructor(PyObject *_handler_) {
	PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(_handler_, "mem_handler");
	if (!handler) {
		return;
	}

	Py_XDECREF(((PyDataMem_Funcs *) handler->allocator.ctx)->realloc);

	Py_XDECREF(((PyDataMem_Funcs *) handler->allocator.ctx)->malloc);

	Py_XDECREF(((PyDataMem_Funcs *) handler->allocator.ctx)->free);

	Py_XDECREF(((PyDataMem_Funcs *) handler->allocator.ctx)->calloc);

	free(handler->allocator.ctx);

	free(handler);
}

static int tp_init(PyObject *cls, PyObject *args, PyObject *kwds) {
	PyDataMem_Handler *handler = (PyDataMem_Handler *) calloc(1, sizeof(PyDataMem_Handler));
	if (!handler) {
		PyErr_NoMemory();

		return -1;
	}

	handler->allocator.ctx = calloc(1, sizeof(PyDataMem_Funcs));
	if (!handler->allocator.ctx) {
		free(handler);

		PyErr_NoMemory();

		return -1;
	}

	PyObject *_handler_ = PyCapsule_New(handler, "mem_handler", handler_destructor);
	if (!_handler_) {
		free(handler->allocator.ctx);

		free(handler);

		return -1;
	}

	strncpy(handler->name, _PyType_Name((PyTypeObject *) cls), sizeof(((PyDataMem_Handler *) NULL)->name) - 1);

	PyObject *_calloc_ = PyObject_GetAttrString(cls, "_calloc_");
	if (!_calloc_ || _calloc_ == Py_None) {
		Py_XDECREF(_calloc_);
		handler->allocator.calloc = default_calloc;
	} else {
		((PyDataMem_Funcs *) handler->allocator.ctx)->calloc = (PyCFuncPtrObject *) _calloc_;
		handler->allocator.calloc = safe_calloc;
	}

	PyObject *_free_ = PyObject_GetAttrString(cls, "_free_");
	if (!_free_ || _free_ == Py_None) {
		Py_XDECREF(_free_);
		handler->allocator.free = default_free;
	} else {
		((PyDataMem_Funcs *) handler->allocator.ctx)->free = (PyCFuncPtrObject *) _free_;
		handler->allocator.free = safe_free;
	}

	PyObject *_malloc_ = PyObject_GetAttrString(cls, "_malloc_");
	if (!_malloc_ || _malloc_ == Py_None) {
		Py_XDECREF(_malloc_);
		handler->allocator.malloc = default_malloc;
	} else {
		((PyDataMem_Funcs *) handler->allocator.ctx)->malloc = (PyCFuncPtrObject *) _malloc_;
		handler->allocator.malloc = safe_malloc;
	}

	PyObject *_realloc_ = PyObject_GetAttrString(cls, "_realloc_");
	if (!_realloc_ || _realloc_ == Py_None) {
		Py_XDECREF(_realloc_);
		handler->allocator.realloc = default_realloc;
	} else {
		((PyDataMem_Funcs *) handler->allocator.ctx)->realloc = (PyCFuncPtrObject *) _realloc_;
		handler->allocator.realloc = safe_realloc;
	}

	int error = PyObject_SetAttrString(cls, "_handler_", _handler_);
	Py_DECREF(_handler_);
	if (error) {
		return -1;
	}

	return 0;
}

static PyObject *handles(PyObject *cls, PyObject *args) {
	while (args && PyArray_Check(args)) {
		if (PyArray_CHKFLAGS((PyArrayObject *) args, NPY_ARRAY_OWNDATA)) {
			PyObject *_handler_ = PyArray_HANDLER((PyArrayObject *) args);
			if (!_handler_) {
				PyErr_SetString(PyExc_RuntimeError, "no memory handler found but OWNDATA flag set");
				return NULL;
			}

			PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(_handler_, "mem_handler");
			if (!handler) {
				return NULL;
			}

			if (strncmp(handler->name, _PyType_Name((PyTypeObject *) cls), sizeof(((PyDataMem_Handler *) NULL)->name))) {
				Py_RETURN_FALSE;
			}

			Py_RETURN_TRUE;
		}

		args = PyArray_BASE((PyArrayObject *) args);
	}

	PyErr_SetString(PyExc_ValueError, "argument must be an ndarray");
	return NULL;
}

PyObject *var;

static PyObject *PyContextVar_Pop(PyObject *var) {
	PyObject *list;
	if (PyContextVar_Get(var, NULL, &list)) {
		return NULL;
	}

	PyObject *capsule = PySequence_GetItem(list, PySequence_Size(list) - 1);
	if (!capsule) {
		Py_DECREF(list);

		return NULL;
	}

	int error = PySequence_DelItem(list, PySequence_Size(list) - 1);
	Py_DECREF(list);
	if (error) {
		Py_DECREF(capsule);

		return NULL;
	}

	return capsule;
}

static PyObject *__exit__(PyObject *cls, PyObject *args) {
	PyObject *new_handler = PyContextVar_Pop(var);
	if (!new_handler) {
		return NULL;
	}

	PyObject *old_handler = PyDataMem_SetHandler(new_handler);
	Py_DECREF(new_handler);
	if (!old_handler) {
		return NULL;
	}
	Py_DECREF(old_handler);

	Py_RETURN_NONE;
}

static int PyContextVar_Push(PyObject *var, PyObject *capsule) {
	PyObject *list;
	if (PyContextVar_Get(var, NULL, &list)) {
		return -1;
	}

	int error = PyList_Append(list, capsule);
	Py_DECREF(list);
	if (error) {
		return -1;
	}

	return 0;
}

static PyObject *__enter__(PyObject *cls, PyObject *args) {
	PyObject *new_handler = PyObject_GetAttrString(cls, "_handler_");
	if (!new_handler) {
		return NULL;
	}

	PyObject *old_handler = PyDataMem_SetHandler(new_handler);
	Py_DECREF(new_handler);
	if (!old_handler) {
		return NULL;
	}

	int error = PyContextVar_Push(var, old_handler);
	Py_DECREF(old_handler);
	if (error) {
		return NULL;
	}

	Py_RETURN_NONE;
}

static PyMethodDef tp_methods[] = {
	{"__enter__", __enter__, METH_NOARGS, NULL},
	{"__exit__", __exit__, METH_VARARGS, NULL},
	{"handles", handles, METH_O, NULL},
	{NULL, NULL, 0, NULL},
};

static PyTypeObject type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "numpy_allocator.type",
	.tp_methods = tp_methods,
	.tp_base = &PyType_Type,
	.tp_init = tp_init,
	.tp_finalize = tp_finalize,
};

static int exec_module(PyObject *module) {
	PyObject *list = PyList_New(0);
	if (!list) {
		return -1;
	}

    var = PyContextVar_New("var", list);
    Py_DECREF(list);
    if (!var) {
		return -1;
	}

	if (PyType_Ready(&type)) {
		return -1;
	}

	Py_INCREF(&type);

	if (PyModule_AddObject(module, _PyType_Name(&type), (PyObject *) &type)) {
		Py_DECREF(&type);

		return -1;
	}

	return 0;
}

static PyModuleDef_Slot m_slots[] = {
	{Py_mod_exec, exec_module},
	{0, NULL},
};

static PyModuleDef def = {
	PyModuleDef_HEAD_INIT,
	.m_name = "numpy_allocator",
	.m_slots = m_slots,
};

PyMODINIT_FUNC PyInit_numpy_allocator(void) {
	import_array();

	return PyModuleDef_Init(&def);
}
