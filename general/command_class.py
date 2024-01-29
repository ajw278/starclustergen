from __future__ import print_function


import subprocess, threading




class Command(object):
	def __init__(self, cmd):
		self.cmd = cmd
		self.process = None


	def run(self, timeout):
		def target():
			print('Thread started: {0}'.format(self.cmd))
			#ulimit -s unlimited; 
			self.process = subprocess.Popen("exec " +self.cmd, shell=True)
			self.process.communicate()
			print('Thread finished')

		thread = threading.Thread(target=target)
		thread.start()

		thread.join(timeout)
		if thread.is_alive():
			print('Terminating process')
			self.process.terminate()
			thread.join()
		print(self.process.returncode)
