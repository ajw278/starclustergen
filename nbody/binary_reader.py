import os
import pandas as pd
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir)
sys.path.append(scriptdir+'/../general')

from common import *
import saveload


class BinarySnapshot:
	def __init__(self, run_dir=None, fname='simulation'):

		self.home_dir = os.getcwd()
		self.out = fname
		self.compiled=False
		if not self.load(self.home_dir):
			if run_dir  is None:
				self.run_dir = self.home_dir+'/run_dir'
			else:
				self.run_dir = run_dir
			self.header_cols =  ['File', 'NPAIRS', 'MODEL', 'NRUN', 'N', 'NC', 'NMERGE', 'Time[NB]', 'RSCALE[NB]', 'RTIDE[NB]', 'RC[NB]',
				'Time[Myr]', 'ETIDC[NB]', '0']
			self.dtypes = [str, int, int, int, int, float, float, float, float, float, float, float, float, float]
			self.snapshots_info = pd.DataFrame(columns=self.header_cols)
			self.snapshots_data = []
			self.get_flist(self.run_dir)
			self.save(self.home_dir)
		
		self.home_dir = os.getcwd()
		if run_dir  is None:
			self.run_dir = self.home_dir+'/run_dir'
		else:
			self.run_dir = run_dir

	def get_flist(self, run_dir):
		os.chdir(run_dir)
		snapshot_files = []
		files_tmp = glob.glob('bdat.9_'+'*')

		if len(files_tmp)==0:
			raise Exception('No binary data files found in run directory: %s'%run_dir)
		
		fle_nums = np.zeros(len(files_tmp))
		for ifle in range(len(files_tmp)):
			fle_nums[ifle] = float(files_tmp[ifle].split('_')[-1])

		ifile_srt = np.argsort(fle_nums)

		for ifile in ifile_srt:
			snapshot_files.append(files_tmp[ifile])
		self.snapshot_files = snapshot_files
	
	def save(self, home_dir):
		os.chdir(home_dir)
		if not os.path.exists('obj'):
			os.makedirs('obj')
		saveload.save_obj(self, self.out+'.binaries')
		return None

	def load(self, home_dir):
		os.chdir(home_dir)

		if os.path.exists('obj/'+self.out+'.binaries.pkl'):
			try:
				oldclass = saveload.load_obj(self.out+'.binaries')
			except:
				oldclass = saveload.load_obj_main(self.out+'.binaries', 'binary_reader')
			oldprops = oldclass.__dict__ 
			for okey in oldprops:
				setattr(self, okey, getattr(oldclass,okey))
			
			return True
		else:
			return False
		

	def create_database(self):
		if not hasattr(self, 'snapshot_files'):
			self.get_flist()

		if not self.compiled:
			for snapshot_file in self.snapshot_files:
				snapshot_path = os.path.join(self.run_dir, snapshot_file)
				header_1_info = self.extract_header_1_info(snapshot_path)
				header_1_info['File'] = snapshot_file
				irow = len(self.snapshots_info)
				self.snapshots_info.loc[irow] = header_1_info
		
		print(self.snapshots_info)
		self.compiled = True
		self.save(self.home_dir)

	def extract_header_1_info(self, snapshot_path):
	
		file_name = os.path.basename(snapshot_path)
		time_nb = float(file_name.split('_')[-1])  # Extracting Time[NB] from the file name
        

		with open(snapshot_path, 'r') as file:
			header_1_str = file.readline()
			header_1 = header_1_str.split()
			if len(header_1)<13:
				print(header_1)
				fpart = int(header_1_str[:4])
				try:
					lpart = int(header_1_str[4:8])
				except:
					lpart = -1
				header_1[0] = fpart
				header_1.insert(1, lpart)
			
			
			header_1_dict = {self.header_cols[i+1]: self.dtypes[i+1](header_1[i]) for i in range(len(header_1))}
			# Adding Time[NB] to the DataFrame
			header_1_dict['Time[NB]'] = time_nb
			return header_1_dict

	def read_snapshots(self, snapshot_index=None):
		if snapshot_index is None:
			snapshot_index = range(len(self.snapshots_info))

		for idx in snapshot_index:
			snapshot_path = os.path.join(directory, self.snapshots_info.iloc[idx]['File'])
			snapshot_data = self.read_snapshot(snapshot_path)
			self.snapshots_data.append(snapshot_data)
	
	def search_snapshot(self, attribute, value):
		"""
		Search for a snapshot by a given attribute and its value in snapshots_info.

		Parameters:
		- attribute (str): The attribute to search for.
		- value: The value to match.

		Returns:
		- pd.DataFrame or None: The data from the corresponding snapshot file if found, otherwise None.
		"""
		try:
			# Check if the attribute is a float column
			is_float_column = self.snapshots_info[attribute].dtype == float

			if is_float_column:
				# If it's a float column, find the closest value
				closest_index = np.argmin(np.abs(self.snapshots_info[attribute] - value))
				result_df = self.snapshots_info.loc[closest_index]
			else:
				# If not a float column, perform an exact match
				result_df = self.snapshots_info[self.snapshots_info[attribute] == value].iloc[0]

			if not result_df.empty:
				snapshot_file = result_df['File']
				snapshot_data = self.read_snapshot(self.run_dir+'/'+snapshot_file)
				return snapshot_data
			else:
				print("Snapshot not found for attribute {} = {}".format(attribute, value))
				return None
		except Exception as e:
			print("Error searching for snapshot:", str(e))
			return None

	def read_snapshot(self, snapshot_path):

		
		with open(snapshot_path, 'r') as file:
			# Skip the header lines
			for _ in range(3):
				file.readline()
			header = file.readline().split()
			# Read the data into a DataFrame with explicit delimiter
			data = pd.read_csv(file, delim_whitespace=True, comment='#', names=header)
		
		return data
	
	def gen_history(self, istar):
		if not self.compiled:
			print('Database not yet compiled. Executing now...')
			self.create_database()
		
		
		result_dict = {
			'bflag': [],
			't': [],
			'icomp': [],
			'mcomp': [],
			'ecc': [],
			'semi': []
		}
		

		for idx in range(len(self.snapshots_info)):
			snapshot_data = self.read_snapshot(self.run_dir+'/'+self.snapshots_info.iloc[idx]['File'])

			# Check if istar is in the binary companion's NAME (either NAME(I1) or NAME(I2))
			is_star_found = (
				(snapshot_data['NAME(I1)'] == istar) | (snapshot_data['NAME(I2)'] == istar)
			)

			result_dict['bflag'].append(is_star_found.any())  # True if the star is found in any binary
			result_dict['t'].append(self.snapshots_info.iloc[idx]['Time[NB]'])
			
			if is_star_found.any():
				# Get the index of the first binary where istar is found
				comp_index = np.where(is_star_found)[0][0]

				# Extract relevant information
				icomp = snapshot_data.iloc[comp_index]['NAME(I1)'] if istar == snapshot_data.iloc[comp_index]['NAME(I1)'] else snapshot_data.iloc[comp_index]['NAME(I2)']
				mcomp = snapshot_data.iloc[comp_index]['M1[M*]'] if istar == snapshot_data.iloc[comp_index]['NAME(I1)'] else snapshot_data.iloc[comp_index]['M2[M*]']
				ecc = snapshot_data.iloc[comp_index]['ECC']
				semi = snapshot_data.iloc[comp_index]['SEMI[AU]']

				result_dict['icomp'].append(icomp)
				result_dict['mcomp'].append(mcomp)
				result_dict['ecc'].append(ecc)
				result_dict['semi'].append(semi)
			else:
				# If istar is not found in any binary, insert np.nan for the corresponding elements
				result_dict['icomp'].append(-1)
				result_dict['mcomp'].append(np.nan)
				result_dict['ecc'].append(np.nan)
				result_dict['semi'].append(np.nan)
		for key in result_dict:
			if key =='icomp':
				result_dict[key] = np.asarray(result_dict[key], dtype=int)
			elif key=='bflag':
				result_dict[key] = np.asarray(result_dict[key], dtype=bool)
			else:
				result_dict[key] = np.asarray(result_dict[key])
		
		return result_dict

class WideBinarySnapshot:
	def __init__(self, run_dir=None, fname='simulation'):

		self.home_dir = os.getcwd()
		self.out = fname
		self.compiled=False
		if not self.load(self.home_dir):
			if run_dir  is None:
				self.run_dir = self.home_dir+'/run_dir'
			else:
				self.run_dir = run_dir
			self.header_cols =  ['File',  'Time[NB]', 'Time[Myr]', 'N']
			self.dtypes = [str, float, float, float, int]
			self.snapshots_info = pd.DataFrame(columns=self.header_cols)
			self.snapshots_data = []
			self.get_flist(self.run_dir)
			self.save(self.home_dir)
		
		self.home_dir = os.getcwd()
		if run_dir  is None:
			self.run_dir = self.home_dir+'/run_dir'
		else:
			self.run_dir = run_dir

	def get_flist(self, run_dir, nmax=10):
		os.chdir(run_dir)
		snapshot_files = []
		files_tmp = glob.glob('bwdat.19_*')

		if len(files_tmp)==0:
			raise Exception('No wide binary data files found in run directory: %s'%run_dir)
		
		if len(files_tmp)>0:
			fle_nums = np.zeros(len(files_tmp))
			for ifle in range(len(files_tmp)):
				fle_nums[ifle] = float(files_tmp[ifle].split('_')[-1])

			ifile_srt = np.argsort(fle_nums)
				
				#conf_list.append(iconf)
				#subconf_list.append([])
			for ifile in ifile_srt:
				#subconf_list.append(int(fname.split('.')[-1]))
				snapshot_files.append(files_tmp[ifile])
		self.snapshot_files = snapshot_files
	
	def save(self, home_dir):
		os.chdir(home_dir)
		if not os.path.exists('obj'):
			os.makedirs('obj')
		saveload.save_obj(self, self.out+'.widebinaries')
		return None

	def load(self, home_dir):
		os.chdir(home_dir)

		if os.path.exists('obj/'+self.out+'.widebinaries.pkl'):
			try:
				oldclass = saveload.load_obj(self.out+'.widebinaries')
			except:
				oldclass = saveload.load_obj_main(self.out+'.widebinaries', 'binary_reader')
			oldprops = oldclass.__dict__ 
			for okey in oldprops:
				setattr(self, okey, getattr(oldclass,okey))
			
			return True
		else:
			return False
		

	def create_database(self):
		if not hasattr(self, 'snapshot_files'):
			self.get_flist()

		if not self.compiled:
			for snapshot_file in self.snapshot_files:
				snapshot_path = os.path.join(self.run_dir, snapshot_file)
				header_1_info = self.extract_header_1_info(snapshot_path)
				header_1_info['File'] = snapshot_file
				irow = len(self.snapshots_info)
				self.snapshots_info.loc[irow] = header_1_info
		
		print(self.snapshots_info)
		self.compiled = True
		self.save(self.home_dir)

	def extract_header_1_info(self, snapshot_path):
	
		file_name = os.path.basename(snapshot_path)
		time_nb = float(file_name.split('_')[-1])  # Extracting Time[NB] from the file name
        

		with open(snapshot_path, 'r') as file:
			header_1_str = file.readline()
			header_1 = header_1_str.split()[5:]
			if len(header_1)<3:
				fpart = int(header_1_str[:4])
				lpart = int(header_1_str[4:8])
				header_1[0] = fpart
				header_1.insert(1, lpart)
			
			print(header_1, header_1_str)
			
			header_1_dict = {self.header_cols[i+1]: self.dtypes[i+1](header_1[i]) for i in range(len(header_1))}
			# Adding Time[NB] to the DataFrame
			header_1_dict['Time[NB]'] = time_nb
			return header_1_dict

	def read_snapshots(self, snapshot_index=None):
		if snapshot_index is None:
			snapshot_index = range(len(self.snapshots_info))

		for idx in snapshot_index:
			snapshot_path = os.path.join(directory, self.snapshots_info.iloc[idx]['File'])
			snapshot_data = self.read_snapshot(snapshot_path)
			self.snapshots_data.append(snapshot_data)
	
	def search_snapshot(self, attribute, value):
		"""
		Search for a snapshot by a given attribute and its value in snapshots_info.

		Parameters:
		- attribute (str): The attribute to search for.
		- value: The value to match.

		Returns:
		- pd.DataFrame or None: The data from the corresponding snapshot file if found, otherwise None.
		"""
		try:
			# Check if the attribute is a float column
			is_float_column = self.snapshots_info[attribute].dtype == float

			if is_float_column:
				# If it's a float column, find the closest value
				closest_index = np.argmin(np.abs(self.snapshots_info[attribute] - value))
				result_df = self.snapshots_info.loc[closest_index]
			else:
				# If not a float column, perform an exact match
				result_df = self.snapshots_info[self.snapshots_info[attribute] == value].iloc[0]

			if not result_df.empty:
				snapshot_file = result_df['File']
				snapshot_data = self.read_snapshot(self.run_dir+'/'+snapshot_file)
				return snapshot_data
			else:
				print("Snapshot not found for attribute {} = {}".format(attribute, value))
				return None
		except Exception as e:
			print("Error searching for snapshot:", str(e))
			return None

	def read_snapshot(self, snapshot_path):

		
		with open(snapshot_path, 'r') as file:
			# Skip the header lines
			for _ in range(1):
				file.readline()
			header = file.readline().split()
			# Read the data into a DataFrame with explicit delimiter
			data = pd.read_csv(file, delim_whitespace=True, comment='#', names=header)
		
		return data
	
	def gen_history(self, istar):
		if not self.compiled:
			print('Database not yet compiled. Executing now...')
			self.create_database()
		
		
		result_dict = {
			'bflag': [],
			't': [],
			'icomp': [],
			'mcomp': [],
			'ecc': [],
			'semi': []
		}
		

		for idx in range(len(self.snapshots_info)):
			snapshot_data = self.read_snapshot(self.run_dir+'/'+self.snapshots_info.iloc[idx]['File'])

			# Check if istar is in the binary companion's NAME (either NAME(I1) or NAME(I2))
			is_star_found = (
				(snapshot_data['NAME(I1)'] == istar) | (snapshot_data['NAME(I2)'] == istar)
			)

			result_dict['bflag'].append(is_star_found.any())  # True if the star is found in any binary
			result_dict['t'].append(self.snapshots_info.iloc[idx]['Time[NB]'])
			
			if is_star_found.any():
				# Get the index of the first binary where istar is found
				comp_index = np.where(is_star_found)[0][0]

				# Extract relevant information
				icomp = snapshot_data.iloc[comp_index]['NAME(I1)'] if istar == snapshot_data.iloc[comp_index]['NAME(I1)'] else snapshot_data.iloc[comp_index]['NAME(I2)']
				mcomp = snapshot_data.iloc[comp_index]['M(I1)[M*]'] if istar == snapshot_data.iloc[comp_index]['NAME(I1)'] else snapshot_data.iloc[comp_index]['M(I2)[M*]']
				ecc = snapshot_data.iloc[comp_index]['ECC']
				semi = snapshot_data.iloc[comp_index]['SEMI[AU]']

				result_dict['icomp'].append(icomp)
				result_dict['mcomp'].append(mcomp)
				result_dict['ecc'].append(ecc)
				result_dict['semi'].append(semi)
			else:
				# If istar is not found in any binary, insert np.nan for the corresponding elements
				result_dict['icomp'].append(-1)
				result_dict['mcomp'].append(np.nan)
				result_dict['ecc'].append(np.nan)
				result_dict['semi'].append(np.nan)
		for key in result_dict:
			if key =='icomp':
				result_dict[key] = np.asarray(result_dict[key], dtype=int)
			elif key=='bflag':
				result_dict[key] = np.asarray(result_dict[key], dtype=bool)
			else:
				result_dict[key] = np.asarray(result_dict[key])
		
		return result_dict

class AllBinaries:
	def __init__(self, star_names, fname='simulation', binary_snapshot_filename=None, wide_binary_snapshot_filename=None, run_dir=None):		
		self.home_dir = os.getcwd()
		self.run_dir = run_dir
		self.out = fname
		if run_dir is None:
			self.run_dir = self.home_dir +'/run_dir'
		if not self.load(self.home_dir):
			
			self.star_names = star_names
			self.binary_snapshot_filename = binary_snapshot_filename
			self.wide_binary_snapshot_filename = wide_binary_snapshot_filename
			self.times_wide_binary = self.get_times_wide_binary()
			
			# Create empty arrays for 'bflag', 't', 'icomp', 'mcomp', 'ecc', 'semi'
			num_times = len(self.times_wide_binary)
			self.num_times = num_times
			
			num_stars = len(self.star_names)
			self.num_stars = num_stars 
			
			self.bflag = np.zeros((num_times, num_stars), dtype=bool)
			self.compiled_flag = np.zeros(num_times, dtype=bool)
			self.icomp = np.full((num_times, num_stars), -1, dtype=int)
			self.mcomp = np.full((num_times, num_stars), np.nan)
			self.ecc = np.full((num_times, num_stars), np.nan)
			self.semi = np.full((num_times, num_stars), np.nan)
			
			self.save(self.home_dir)
		

	def save(self, home_dir, flag=''):
		os.chdir(home_dir)
		if not os.path.exists('obj'):
			os.makedirs('obj')
		saveload.save_obj(self, self.out+flag+'.allbinaries')
		return None

	def load(self, home_dir):
		os.chdir(home_dir)

		if os.path.exists('obj/'+self.out+'.allbinaries.pkl'):
			try:
				oldclass = saveload.load_obj(self.out+'.allbinaries')
			except:
				oldclass = saveload.load_obj_main(self.out+'.allbinaries', 'binary_reader')
			
			oldprops = oldclass.__dict__ 
			for okey in oldprops:
				setattr(self, okey, getattr(oldclass,okey))
			
			return True
		else:
			return False

	def get_times_wide_binary(self):
		# Create a WideBinarySnapshot instance to extract times
		if self.wide_binary_snapshot_filename is None:
			wide_bin_snap = WideBinarySnapshot()
		else:
			wide_bin_snap = WideBinarySnapshot(fname=self.wide_binary_snapshot_filename)
		return wide_bin_snap.snapshots_info['Time[NB]'].values
	
	def save_snap(self, home_dir, datarr, i):
		fname = home_dir+'/sim_ts/'+self.out+'_ts_%d'%i
		if not os.path.exists(home_dir+'/sim_ts'):
			os.makedirs(home_dir+'/sim_ts')
		np.save(fname, datarr)
		return None
	
	def load_snap(self, home_dir, i):
		fname = home_dir+'/sim_ts/'+self.out+'_ts_%d'%i +'.npy'

		if os.path.exists(fname):
			datarr = np.load(fname)
			return datarr
		
		return None

	def create_binary_arrays(self):
		
		# Load instances of BinarySnapshot and WideBinarySnapshot
		if self.binary_snapshot_filename is None:
			bin_snap = BinarySnapshot()
		else:
			bin_snap = BinarySnapshot(fname=self.binary_snapshot_filename)

		if self.wide_binary_snapshot_filename is None:
			wide_bin_snap = WideBinarySnapshot()
		else:
			wide_bin_snap = WideBinarySnapshot(fname=self.wide_binary_snapshot_filename)

		# Loop through times and stars to populate arrays
		for i, time in enumerate(self.times_wide_binary):
			darr = self.load_snap(self.home_dir, i)
			if not darr is None:
				self.bflag[i], self.icomp[i], self.mcomp[i], self.ecc[i], self.semi[i] = darr[:]
				self.compiled_flag[i] =True
			elif not self.compiled_flag[i]:
				wbin_data = wide_bin_snap.search_snapshot('Time[NB]', time)
				bin_data = bin_snap.search_snapshot('Time[NB]', time)
				
				for j, star_name in enumerate(self.star_names):
					is_star_found = np.any(
						(wbin_data['NAME(I1)'] == star_name) | (wbin_data['NAME(I2)'] == star_name)
					)
					if is_star_found:
						self.bflag[i, j] = True
						comp_index = np.where((wbin_data['NAME(I1)'] == star_name) | (wbin_data['NAME(I2)'] == star_name))[0][0]
						self.icomp[i, j] = wbin_data.iloc[comp_index]['NAME(I1)'] if star_name == wbin_data.iloc[comp_index]['NAME(I1)'] else wbin_data.iloc[comp_index]['NAME(I2)']
						self.mcomp[i, j] = wbin_data.iloc[comp_index]['M(I1)[M*]'] if star_name == wbin_data.iloc[comp_index]['NAME(I1)'] else wbin_data.iloc[comp_index]['M(I2)[M*]']
						self.ecc[i, j] = wbin_data.iloc[comp_index]['ECC']
						self.semi[i, j] = wbin_data.iloc[comp_index]['SEMI[AU]']
					
					else:
						is_star_found = np.any(
							(bin_data['NAME(I1)'] == star_name) | (bin_data['NAME(I2)'] == star_name)
						)
						if is_star_found:
							self.bflag[i, j] = True
							comp_index = np.where((bin_data['NAME(I1)'] == star_name) | (bin_data['NAME(I2)'] == star_name))[0][0]
							self.icomp[i, j] = bin_data.iloc[comp_index]['NAME(I1)'] if star_name == bin_data.iloc[comp_index]['NAME(I1)'] else bin_data.iloc[comp_index]['NAME(I2)']
							self.mcomp[i, j] = bin_data.iloc[comp_index]['M1[M*]'] if star_name == bin_data.iloc[comp_index]['NAME(I1)'] else bin_data.iloc[comp_index]['M2[M*]']
							self.ecc[i, j] = bin_data.iloc[comp_index]['ECC']
							self.semi[i, j] = bin_data.iloc[comp_index]['SEMI[AU]']


				darr = np.array([self.bflag[i], self.icomp[i], self.mcomp[i], self.ecc[i], self.semi[i]])
				self.save_snap(self.home_dir, darr, i)
				print('Complete for {0}/{1} times'.format(i+1, len(self.times_wide_binary)))
				self.compiled_flag[i] =True
		self.save(self.home_dir)
				
			
		return {
		'bflag': self.bflag,
		'icomp': self.icomp,
		'mcomp': self.mcomp,
		'ecc': self.ecc,
		'semi': self.semi
		}
	
	def get_history(self, istar):
		if np.any(istar==self.star_names):
			istar_loc = np.where(self.star_names==istar)[0]
			return self.times_wide_binary, self.bflag[:, istar_loc], self.icomp[:, istar_loc], self.semi[:, istar_loc], self.ecc[:, istar_loc], self.mcomp[:, istar_loc]
		else:
			raise Exception('No star "{0}" found in database'.format(istar))
			

if __name__=='__main__':
	import nbody6_interface as nbi
	sim = nbi.nbody6_cluster(np.array([]), np.array([]), np.array([]),  outname='clustersim', load=True, init=False)

	wbin_snap = WideBinarySnapshot()
	wbin_snap.create_database()

	binary_snapshot = BinarySnapshot()
	binary_snapshot.create_database() 
	
	munit, runit, tunit, vunit = sim.units_astro
	istars = np.arange(1100)
	allbin = AllBinaries(istars)
	allbin.create_binary_arrays()
	plt.rc('text', usetex=True)
	irand = np.random.choice(istars, size=200, replace=False)
	#irand  = np.arange(10)
	fig, ax = plt.subplots(figsize=(5.,4.))
	for istar in irand:
		t,bf, ic, a, e, m2 = allbin.get_history(istar)
		plt.plot(t*tunit, a, linewidth=1)

	ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
	plt.yscale('log')
	plt.xlabel('Time [Myr]')
	plt.ylabel('Semi-major axis: $a$ [au]')
	plt.ylim([1., 2e4])
	plt.xlim([0.,3.])
	plt.savefig('sma_200bin.pdf', format='pdf', bbox_inches='tight')
	plt.show()
	
	fig, ax = plt.subplots(figsize=(5.,4.))
	for istar in irand:
		t,bf, ic, a, e, m2 = allbin.get_history(istar)
		plt.plot(t*tunit, a*(1-e), linewidth=1)

	ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
	plt.yscale('log')
	plt.xlabel('Time [Myr]')
	plt.ylabel('Pericentre distance: $a(1-e)$ [au]')
	plt.ylim([1., 2e4])
	plt.xlim([0.,3.])
	plt.savefig('rperi_100bin.pdf', format='pdf', bbox_inches='tight')
	plt.show()
	exit()

	wbin_snap = WideBinarySnapshot()
	wbin_snap.create_database()

	binary_snapshot = BinarySnapshot()
	binary_snapshot.create_database()  # Create a database with file information


	for istar in [39, 41, 43, 45, 47]:
		star1_dict = wbin_snap.gen_history(istar)

		print(star1_dict['t'])

		plt.plot(star1_dict['t'], star1_dict['semi']*(1.-star1_dict['ecc']))

	for istar in [17, 99, 139, 175, 197]:
		star1_dict = binary_snapshot.gen_history(istar)

		print(star1_dict['t'])

		plt.plot(star1_dict['t'], star1_dict['semi']*(1.-star1_dict['ecc']), linestyle='dashed')
	plt.yscale('log')
	plt.show()

	bsnap = binary_snapshot.search_snapshot('Time[NB]', 0.0)

	print(bsnap)
	import matplotlib.pyplot as plt
	plt.scatter(np.log10(bsnap['P[Days]']), bsnap['ECC'])
	plt.show()
	plt.hist(bsnap['ECC'])
	plt.show()
	plt.hist(np.log10(bsnap['P[Days]']))
	plt.show()
	plt.hist(np.log10(bsnap['SEMI[AU]']))
	plt.show()

	# Display the pandas DataFrame with header-1 information
	print(binary_snapshot.snapshots_info)
