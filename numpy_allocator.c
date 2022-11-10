#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#include <numpy/arrayobject.h>
#include <string.h>

typedef struct {
	PyObject_HEAD
	void **ptr;
} PyCFuncPtrObject;

typedef struct {
	PyObject *self;
	PyCFuncPtrObject *calloc;
	PyCFuncPtrObject *free;
	PyCFuncPtrObject *malloc;
	PyCFuncPtrObject *realloc;
} PyDataMem_Funcs;

typedef void *(PyDataMemType_ReallocFunc)(void *ptr, size_t new_size);

typedef void *(PyDataMemBaseObject_ReallocFunc)(PyObject *self, void *ptr, size_t new_size);

static void *safe_realloc(void *ctx, void *ptr, size_t new_size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *new_ptr;
	if (((PyDataMem_Funcs *) ctx)->self) {
		new_ptr = ((PyDataMemBaseObject_ReallocFunc *) *((PyDataMem_Funcs *) ctx)->realloc->ptr)(((PyDataMem_Funcs *) ctx)->self, ptr, new_size);
	} else {
		new_ptr = ((PyDataMemType_ReallocFunc *) *((PyDataMem_Funcs *) ctx)->realloc->ptr)(ptr, new_size);
	}
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return new_ptr;
}

static void *default_realloc(void *ctx, void *ptr, size_t new_size) {
	return realloc(ptr, new_size);
}

typedef void *(PyDataMemType_MallocFunc)(size_t size);

typedef void *(PyDataMemBaseObject_MallocFunc)(PyObject *self, size_t size);

static void *safe_malloc(void *ctx, size_t size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *ptr;
	if (((PyDataMem_Funcs *) ctx)->self) {
		ptr = ((PyDataMemBaseObject_MallocFunc *) *((PyDataMem_Funcs *) ctx)->malloc->ptr)(((PyDataMem_Funcs *) ctx)->self, size);
	} else {
		ptr = ((PyDataMemType_MallocFunc *) *((PyDataMem_Funcs *) ctx)->malloc->ptr)(size);
	}
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return ptr;
}

static void *default_malloc(void *ctx, size_t size) {
	return malloc(size);
}

typedef void (PyDataMemType_FreeFunc)(void *ptr, size_t size);

typedef void (PyDataMemBaseObject_FreeFunc)(PyObject *self, void *ptr, size_t size);

static void safe_free(void *ctx, void *ptr, size_t size) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	if (((PyDataMem_Funcs *) ctx)->self) {
		((PyDataMemBaseObject_FreeFunc *) *((PyDataMem_Funcs *) ctx)->free->ptr)(((PyDataMem_Funcs *) ctx)->self, ptr, size);
	} else {
		((PyDataMemType_FreeFunc *) *((PyDataMem_Funcs *) ctx)->free->ptr)(ptr, size);
	}
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
}

static void default_free(void *ctx, void *ptr, size_t size) {
	free(ptr);
}

typedef void *(PyDataMemType_CallocFunc)(size_t nelem, size_t elsize);

typedef void *(PyDataMemBaseObject_CallocFunc)(PyObject *self, size_t nelem, size_t elsize);

static void *safe_calloc(void *ctx, size_t nelem, size_t elsize) {
	PyObject *type;
	PyObject *value;
	PyObject *traceback;
	if (PyGILState_Check()) {
		PyErr_Fetch(&type, &value, &traceback);
	}
	void *ptr;
	if (((PyDataMem_Funcs *) ctx)->self) {
		ptr = ((PyDataMemBaseObject_CallocFunc *) *((PyDataMem_Funcs *) ctx)->calloc->ptr)(((PyDataMem_Funcs *) ctx)->self, nelem, elsize);
	} else {
		ptr = ((PyDataMemType_CallocFunc *) *((PyDataMem_Funcs *) ctx)->calloc->ptr)(nelem, elsize);
	}
	if (PyGILState_Check()) {
		PyErr_Restore(type, value, traceback);
	}
	return ptr;
}

static void *default_calloc(void *ctx, size_t nelem, size_t elsize) {
	return calloc(nelem, elsize);
}

static void handler_destructor(PyObject *handler) {
	PyDataMem_Handler *mem_handler = (PyDataMem_Handler *) PyCapsule_GetPointer(handler, "mem_handler");
	if (!mem_handler) {
		return;
	}

	Py_XDECREF(((PyDataMem_Funcs *) mem_handler->allocator.ctx)->realloc);

	Py_XDECREF(((PyDataMem_Funcs *) mem_handler->allocator.ctx)->malloc);

	Py_XDECREF(((PyDataMem_Funcs *) mem_handler->allocator.ctx)->free);

	Py_XDECREF(((PyDataMem_Funcs *) mem_handler->allocator.ctx)->calloc);

	Py_XDECREF(((PyDataMem_Funcs *) mem_handler->allocator.ctx)->self);

	free(mem_handler->allocator.ctx);

	free(mem_handler);
}

static PyObject *handler(PyObject *allocator, PyObject *args) {
	if (PyObject_HasAttrString(allocator, "_handler_")) {
		return PyObject_GetAttrString(allocator, "_handler_");
	} else {
		PyDataMem_Handler *mem_handler = (PyDataMem_Handler *) calloc(1, sizeof(PyDataMem_Handler));
		if (!mem_handler) {
			PyErr_NoMemory();

			return NULL;
		}

		mem_handler->allocator.ctx = calloc(1, sizeof(PyDataMem_Funcs));
		if (!mem_handler->allocator.ctx) {
			free(mem_handler);

			PyErr_NoMemory();

			return NULL;
		}

		PyObject *handler = PyCapsule_New(mem_handler, "mem_handler", handler_destructor);
		if (!handler) {
			free(mem_handler->allocator.ctx);

			free(mem_handler);

			return NULL;
		}

		PyObject *name = PyObject_Str(allocator);
		if (!name) {
			Py_DECREF(handler);

			return NULL;
		}
		strncpy(mem_handler->name, PyUnicode_AsUTF8(name), sizeof(((PyDataMem_Handler *) NULL)->name) - 1);
		Py_DECREF(name);

		mem_handler->version = 1;

		if (!PyType_Check(allocator)) {
			Py_INCREF(allocator);
			((PyDataMem_Funcs *) mem_handler->allocator.ctx)->self = allocator;
		}

		if (PyObject_HasAttrString(allocator, "_calloc_")) {
			PyObject *_calloc_ = PyObject_GetAttrString(allocator, "_calloc_");
			if (!_calloc_) {
				Py_DECREF(handler);

				return NULL;
			} else if (_calloc_ == Py_None) {
				Py_DECREF(_calloc_);
				mem_handler->allocator.calloc = default_calloc;
			} else {
				((PyDataMem_Funcs *) mem_handler->allocator.ctx)->calloc = (PyCFuncPtrObject *) _calloc_;
				mem_handler->allocator.calloc = safe_calloc;
			}
		} else {
			mem_handler->allocator.calloc = default_calloc;
		}

		if (PyObject_HasAttrString(allocator, "_free_")) {
			PyObject *_free_ = PyObject_GetAttrString(allocator, "_free_");
			if (!_free_) {
				Py_DECREF(handler);

				return NULL;
			} else if (_free_ == Py_None) {
				Py_DECREF(_free_);
				mem_handler->allocator.free = default_free;
			} else {
				((PyDataMem_Funcs *) mem_handler->allocator.ctx)->free = (PyCFuncPtrObject *) _free_;
				mem_handler->allocator.free = safe_free;
			}
		} else {
			mem_handler->allocator.free = default_free;
		}

		if (PyObject_HasAttrString(allocator, "_malloc_")) {
			PyObject *_malloc_ = PyObject_GetAttrString(allocator, "_malloc_");
			if (!_malloc_) {
				Py_DECREF(handler);

				return NULL;
			} else if (_malloc_ == Py_None) {
				Py_DECREF(_malloc_);
				mem_handler->allocator.malloc = default_malloc;
			} else {
				((PyDataMem_Funcs *) mem_handler->allocator.ctx)->malloc = (PyCFuncPtrObject *) _malloc_;
				mem_handler->allocator.malloc = safe_malloc;
			}
		} else {
			mem_handler->allocator.malloc = default_malloc;
		}

		if (PyObject_HasAttrString(allocator, "_realloc_")) {
			PyObject *_realloc_ = PyObject_GetAttrString(allocator, "_realloc_");
			if (!_realloc_) {
				Py_DECREF(handler);

				return NULL;
			} else if (_realloc_ == Py_None) {
				Py_DECREF(_realloc_);
				mem_handler->allocator.realloc = default_realloc;
			} else {
				((PyDataMem_Funcs *) mem_handler->allocator.ctx)->realloc = (PyCFuncPtrObject *) _realloc_;
				mem_handler->allocator.realloc = safe_realloc;
			}
		} else {
			mem_handler->allocator.realloc = default_realloc;
		}

		if (PyObject_SetAttrString(allocator, "_handler_", handler)) {
			Py_DECREF(handler);

			return NULL;
		}

		return handler;
	}
}

static PyObject *handles(PyObject *allocator, PyObject *array) {
	if (!PyArray_Check(array)) {
		PyErr_SetString(PyExc_ValueError, "argument must be an ndarray");
		return NULL;
	}

	while (array && PyArray_Check(array)) {
		if (PyArray_CHKFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA)) {
			PyObject *array_handler = PyArray_HANDLER((PyArrayObject *) array);
			if (!array_handler) {
				PyErr_SetString(PyExc_RuntimeError, "no memory handler found but OWNDATA flag set");
				return NULL;
			}

			PyObject *allocator_handler = handler(allocator, array);
			if (!allocator_handler) {
				return NULL;
			}
			Py_DECREF(allocator_handler);

			if (array_handler != allocator_handler) {
				Py_RETURN_FALSE;
			}

			Py_RETURN_TRUE;
		}

		array = PyArray_BASE((PyArrayObject *) array);
	}

	Py_RETURN_FALSE;
}

static PyObject *var;

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

static PyObject *__exit__(PyObject *allocator, PyObject *args) {
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

static PyObject *__enter__(PyObject *allocator, PyObject *args) {
	PyObject *new_handler = handler(allocator, args);
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

	Py_INCREF(allocator);
	return allocator;
}

static PyMethodDef tp_methods[] = {
	{"__enter__", __enter__, METH_NOARGS, NULL},
	{"__exit__", __exit__, METH_VARARGS, NULL},
	{"handler", handler, METH_NOARGS, NULL},
	{"handles", handles, METH_O, NULL},
	{NULL, NULL, 0, NULL},
};

static PyObject *tp_str(PyObject *allocator) {
	PyObject *__name__ = PyObject_GetAttrString(allocator, "__name__");
	if (!__name__) {
		return NULL;
	}

	PyObject *allocator_str = PyObject_Str(__name__);
	Py_DECREF(__name__);
	return allocator_str;
}

static PyTypeObject type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "numpy_allocator.type",
	.tp_str = tp_str,
	.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT,
	.tp_methods = tp_methods,
	.tp_base = &PyType_Type,
};

static PyTypeObject object = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "numpy_allocator.object",
	.tp_str = tp_str,
	.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT,
	.tp_methods = tp_methods,
	.tp_base = &PyBaseObject_Type,
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

	object.tp_new = PyBaseObject_Type.tp_new;
	if (PyType_Ready(&object)) {
		Py_DECREF(var);

		return -1;
	}

	Py_INCREF(&object);

	if (PyModule_AddObject(module, "object", (PyObject *) &object)) {
		Py_DECREF(&object);

		Py_DECREF(var);

		return -1;
	}

	type.tp_new = PyType_Type.tp_new;
	if (PyType_Ready(&type)) {
		Py_DECREF(&object);

		Py_DECREF(var);

		return -1;
	}

	Py_INCREF(&type);

	if (PyModule_AddObject(module, "type", (PyObject *) &type)) {
		Py_DECREF(&type);

		Py_DECREF(&object);

		Py_DECREF(var);

		return -1;
	}

	if (PyObject_SetAttrString(module, "default_handler", PyDataMem_DefaultHandler)) {
		Py_DECREF(&type);

		Py_DECREF(&object);

		Py_DECREF(var);

		return -1;
	}

	return 0;
}

static PyModuleDef_Slot m_slots[] = {
	{Py_mod_exec, exec_module},
	{0, NULL},
};

static PyObject *set_handler(PyObject *module, PyObject *handler) {
	if (handler == Py_None) {
		return PyDataMem_SetHandler(NULL);
	} else {
		return PyDataMem_SetHandler(handler);
	}
}

static PyObject *get_handler(PyObject *module, PyObject *args) {
	PyObject *array = NULL;
	if (!PyArg_ParseTuple(args, "|O:get_handler", &array)) {
		return NULL;
	}

	if (array) {
		if (!PyArray_Check(array)) {
			PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
			return NULL;
		}

		while (array && PyArray_Check(array)) {
			if (PyArray_CHKFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA)) {
				PyObject *array_handler = PyArray_HANDLER((PyArrayObject *) array);
				if (!array_handler) {
					PyErr_SetString(PyExc_RuntimeError, "no memory handler found but OWNDATA flag set");
					return NULL;
				}

				Py_INCREF(array_handler);
				return array_handler;
			}

			array = PyArray_BASE((PyArrayObject *) array);
		}

		Py_RETURN_NONE;
	} else {
		return PyDataMem_GetHandler();
	}
}

static PyMethodDef m_methods[] = {
	{"get_handler", get_handler, METH_VARARGS, NULL},
	{"set_handler", set_handler, METH_O, NULL},
	{NULL, NULL, 0, NULL},
};

static PyModuleDef def = {
	PyModuleDef_HEAD_INIT,
	.m_name = "numpy_allocator",
	.m_methods = m_methods,
	.m_slots = m_slots,
};

PyMODINIT_FUNC PyInit_numpy_allocator(void) {
	import_array();

	return PyModuleDef_Init(&def);
}
