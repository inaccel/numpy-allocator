#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <string.h>

typedef struct {
	PyObject_HEAD
	void **ptr;
} PyCFuncPtrObject;

static void tp_finalize(PyObject *cls) {
	PyObject *_handler_ = PyObject_GetAttrString(cls, "_handler_");
	Py_XDECREF(_handler_);
	if (!_handler_) {
		return;
	}

	PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(_handler_, NULL);
	if (!handler) {
		return;
	}

	Py_DECREF(_handler_);

	free(handler);
}

static void *default_zeroed_alloc(size_t nelems, size_t elsize) {
	return calloc(nelems, elsize);
}

static void *default_realloc(void *ptr, size_t size) {
	return realloc(ptr, size);
}

static void default_free(void *ptr, size_t size) {
	free(ptr);
}

static void *default_alloc(size_t size) {
	return malloc(size);
}

static int tp_init(PyObject *cls, PyObject *args, PyObject *kwds) {
	PyDataMem_Handler *handler = (PyDataMem_Handler *) calloc(1, sizeof(PyDataMem_Handler));
	if (!handler) {
		PyErr_NoMemory();

		return -1;
	}

	strncpy(handler->name, _PyType_Name((PyTypeObject *) cls), sizeof(((PyDataMem_Handler *) NULL)->name) - 1);

	PyCFuncPtrObject *_alloc_ = (PyCFuncPtrObject *) PyObject_GetAttrString(cls, "_alloc_");
	Py_XDECREF(_alloc_);
	if (!_alloc_) {
		handler->alloc = (PyDataMem_AllocFunc *) default_alloc;
	} else {
		handler->alloc = (PyDataMem_AllocFunc *) *_alloc_->ptr;
	}

	PyCFuncPtrObject *_free_ = (PyCFuncPtrObject *) PyObject_GetAttrString(cls, "_free_");
	Py_XDECREF(_free_);
	if (!_free_) {
		handler->free = (PyDataMem_FreeFunc *) default_free;
	} else {
		handler->free = (PyDataMem_FreeFunc *) *_free_->ptr;
	}

	PyCFuncPtrObject *_realloc_ = (PyCFuncPtrObject *) PyObject_GetAttrString(cls, "_realloc_");
	Py_XDECREF(_realloc_);
	if (!_realloc_) {
		handler->realloc = (PyDataMem_ReallocFunc *) default_realloc;
	} else {
		handler->realloc = (PyDataMem_ReallocFunc *) *_realloc_->ptr;
	}

	PyCFuncPtrObject *_zeroed_alloc_ = (PyCFuncPtrObject *) PyObject_GetAttrString(cls, "_zeroed_alloc_");
	Py_XDECREF(_zeroed_alloc_);
	if (!_zeroed_alloc_) {
		handler->zeroed_alloc = (PyDataMem_ZeroedAllocFunc *) default_zeroed_alloc;
	} else {
		handler->zeroed_alloc = (PyDataMem_ZeroedAllocFunc *) *_zeroed_alloc_->ptr;
	}

	PyObject *_handler_ = PyCapsule_New(handler, NULL, NULL);
	if (!_handler_) {
		free(handler);

		return -1;
	}

	if (PyObject_SetAttrString(cls, "_handler_", _handler_)) {
		Py_DECREF(_handler_);

		free(handler);

		return -1;
	}

	return 0;
}

static PyObject *handles(PyObject *cls, PyObject *args) {
	if (!PyArray_Check(args)) {
		return NULL;
	}

	PyArrayObject *obj = (PyArrayObject *) args;

	while (PyArray_BASE(obj)) {
		if (!PyArray_Check(PyArray_BASE(obj))) {
			return NULL;
		}

		obj = (PyArrayObject *) PyArray_BASE(obj);
	}

	if (strncmp(PyDataMem_GetHandlerName(obj), _PyType_Name((PyTypeObject *) cls), sizeof(((PyDataMem_Handler *) NULL)->name))) {
		Py_RETURN_FALSE;
	}

	Py_RETURN_TRUE;
}

static PyObject *__swap__(PyObject *cls, PyObject *args) {
	PyObject *_handler_ = PyObject_GetAttrString(cls, "_handler_");
	Py_XDECREF(_handler_);
	if (!_handler_) {
		return NULL;
	}

	PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(_handler_, NULL);
	if (!handler) {
		return NULL;
	}

	handler = (PyDataMem_Handler *) PyDataMem_SetHandler(handler);

	if (PyCapsule_SetPointer(_handler_, handler)) {
		return NULL;
	}

	return Py_None;
}

static PyMethodDef tp_methods[] = {
	{"__enter__", __swap__, METH_NOARGS, NULL},
	{"__exit__", __swap__, METH_VARARGS, NULL},
	{"handles", handles, METH_O, NULL},
	{NULL, NULL, 0, NULL},
};

static PyTypeObject allocator_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "numpy_allocator.base_allocator",
	.tp_methods = tp_methods,
	.tp_base = &PyType_Type,
	.tp_init = tp_init,
	.tp_finalize = tp_finalize,
};

static int exec_module(PyObject *module) {
	if (PyType_Ready(&allocator_type)) {
		return -1;
	}

	Py_INCREF(&allocator_type);

	if (PyModule_AddObject(module, _PyType_Name(&allocator_type), (PyObject *) &allocator_type)) {
		Py_DECREF(&allocator_type);

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
