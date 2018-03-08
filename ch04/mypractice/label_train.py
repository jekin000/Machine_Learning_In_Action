
import os


SRC_DIR		= 'src_train'
TRAIN_LABEL   	= 'SPAMTrain.label'
SPAM_DIR	= 'spam'
HAM_DIR		= 'ham'	
TRAIN_DIR	= 'TRAINING'

def create_dir():
	if os.path.exists(SPAM_DIR) is False:
		os.mkdir(SPAM_DIR)

	if os.path.exists(HAM_DIR)  is False:
		os.mkdir(HAM_DIR)


def delete_dir():
	import shutil	
	if os.path.exists(SPAM_DIR) is True:
		shutil.rmtree(SPAM_DIR)

	if os.path.exists(HAM_DIR)  is True:
		shutil.rmtree(HAM_DIR)

def load_label():
	label_names = []
	with open(TRAIN_LABEL,'r') as f:
		label_names = f.readlines()

	labels = []	
	names  = []
	for each_ln in  label_names:
		tmp = each_ln.strip().split(' ')
		labels.append(tmp[0])
		names.append(tmp[1])
		
	return labels,names

def copy_files(labels,files):
	import shutil
	file_max_cnt  = len(labels)
	for i in range(file_max_cnt):
		file_path_name = os.path.join(TRAIN_DIR,files[i])
		if int(labels[i]) == 1:
			dst_path_name = os.path.join('ham',files[i])
		else:
			dst_path_name = os.path.join('spam',files[i])
		
		shutil.copyfile(file_path_name,dst_path_name)

#TODO, maybe we could pre-handle the raw data here.
if __name__ == '__main__':
	import ExtractContent 
	ExtractContent.ExtractBodyFromDir(SRC_DIR,TRAIN_DIR)
	delete_dir()
	create_dir()
	labels,files = load_label()
	copy_files(labels,files)
	print 'Finished!'
