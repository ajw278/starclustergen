import os
import pandas as pd
import numpy as np
import glob
import sys

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

    def get_flist(self, run_dir):
        os.chdir(run_dir)
        snapshot_files = []
        for iconf in range(1000):
            files_tmp = glob.glob('bdat.9_'+str(iconf)+'.*')
            
            if len(files_tmp)>0:
                fle_nums = np.zeros(len(files_tmp))
                for ifle in range(len(files_tmp)):
                    fle_nums[ifle] = float(files_tmp[ifle].split('.')[-1])

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
        saveload.save_obj(self, self.out+'.binaries')
        return None

    def load(self, home_dir):
        os.chdir(home_dir)

        if os.path.exists('obj/'+self.out+'.binaries.pkl'):

            oldclass = saveload.load_obj(self.out+'.binaries')
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
        self.compiled = True
        self.save(self.home_dir)

    def extract_header_1_info(self, snapshot_path):
        with open(snapshot_path, 'r') as file:
            header_1 = file.readline().split()
            if len(header_1)<13:
                fpart = header_1[0][:4]
                lpart = header_1[0][4:]
                header_1[0] = fpart
                header_1.insert(1, lpart)
            
            
            header_1_dict = {self.header_cols[i+1]: self.dtypes[i+1](header_1[i]) for i in range(len(header_1))}
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
                print(self.snapshots_info[attribute])
                closest_index = np.argmin(np.abs(self.snapshots_info[attribute] - value))
                print(np.abs(self.snapshots_info[attribute] - value))
                print(closest_index)
                result_df = self.snapshots_info.loc[closest_index]
                print(result_df)
            
            else:
                # If not a float column, perform an exact match
                result_df = self.snapshots_info[self.snapshots_info[attribute] == value].iloc[0]

            if not result_df.empty:
                snapshot_file = result_df['File']
                print(snapshot_file)
                snapshot_data = self.read_snapshot(snapshot_file)
                return snapshot_data
            else:
                print("Snapshot not found for attribute {} = {}".format(attribute, value))
                return None
        except Exception as e:
            print("Error searching for snapshot:", str(e))
            return None

    def read_snapshot(self, snapshot_path):

        with open(self.run_dir+'/'+snapshot_path, 'r') as file:
            # Skip the header lines
            for _ in range(3):
                file.readline()
            header = file.readline().split()
            # Read the data into a DataFrame with explicit delimiter
            data = pd.read_csv(file, delim_whitespace=True, comment='#', names=header)
        
        return data

# Example usage:
binary_snapshot = BinarySnapshot()
binary_snapshot.create_database()  # Create a database with file information

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