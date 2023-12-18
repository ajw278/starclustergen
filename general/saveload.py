import pickle
import  os

def save_obj(obj, name):
	with open('obj/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
	
class ModuleUnpickler(pickle.Unpickler):
	def set_module(self, module_new):
		self.module = module_new

	def find_class(self, module, name):
		if module == "__main__":
			module = self.module
		return super().find_class(module, name)


def load_obj_main(name, module_):
	with open('obj/' + name + '.pkl', 'rb') as f:
		unpickler = ModuleUnpickler(f)
		unpickler.set_module(module_)
		return unpickler.load()
