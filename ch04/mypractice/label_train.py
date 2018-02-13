
import os


SRC_DIR		= 'train_1'
TRAIN_LABEL   	= 'SPAMTrain.label'
SPAM_DIR	= 'spam'
HAM_DIR		= 'ham'	

def create_dir():
	if os.path.exists(SPAM_DIR) is False:
		os.mkdir(SPAM_DIR)

	if os.path.exists(HAM_DIR)  is False:
		os.mkdir(HAM_DIR)


def delete_dir():
	import shutil	
	if os.path.exists(SPAM_DIR) is False:
		shutil.rmtree(SPAM_DIR)

	if os.path.exists(HAM_DIR)  is False:
		shutil.rmtree(HAM_DIR)

#TODO
def load_label():
	return None,None

#TODO
def copy_files(labels,files):
	pass

#TODO, maybe we could pre-handle the raw data here.
delete_dir()
create_dir()
labels,files = load_label()
copy_files(labels,files)
print 'Finished!'
