
# for the spam engine it is nearly 83%
import traceback
import subprocess
import time
import os

LABEL_FILE = 'SPAMTest.label'
LOG_PATH   = ''
SPAM_DIR   = 'TESTING'
SPAM_SCORE = 0.0



def clean_log():
	try:
		with open(LOG_PATH,'w') as f:
			f.write('')
	except BaseException as e:
		traceback.print_exc()
		return None

def read_score():
	import re
	try:
		with open(LOG_PATH,'r') as f:
			lines = f.readlines()
			for l in lines:
				if 'SNAP,get score' in l:
					#() -> group(1)
					#.? -> - or None
					#\d+.?\d+  -> 3.5 or 3
					score = re.search('SNAP,get score=\[(.?\d+.?\d+)\]',l).group(1)
					return float(score)
	except BaseException as e:
		traceback.print_exc()
		return None


def send_mail(emlname):
	#You could not use ' in the command	
	#cmd = ["./temail","-SMTP=127.0.0.1","-FROM='12345@trend.com.tw'","-TO='dd@gmail.com;cc@localhost.com'",emlname]
	cmd = ["./temail","-SMTP=127.0.0.1","-FROM=12345@trend.com.tw","-TO=dd@gmail.com;cc@localhost.com",emlname]
	#use os.system still work
	try:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		(stdout, stderr) = proc.communicate()
	except BaseException as e:
		traceback.print_exc()

def judge_score(sc):
	if sc > SPAM_SCORE:
		return 1
	else:
		return 0
	
def judge_eml(path,emlname):
	clean_log()
	send_mail(os.path.join(path,emlname))
	time.sleep(2)
	return '{} {}'.format(judge_score(read_score()),emlname)


if __name__ == '__main__':
	emls = os.listdir(SPAM_DIR)	
	emls_sorted = sorted(emls)

	with open(LABEL_FILE,'r') as f:
		all_labels = f.readlines()

	for eml in emls_sorted:
		res = 	judge_eml(SPAM_DIR,eml)
		i = 0
		for each_label in all_labels:
			if eml in each_label:	
				all_labels[i] = res+'\n'
			i += 1

	with open(LABEL_FILE,'w') as f:
		f.writelines(all_labels)

	exit(0)

