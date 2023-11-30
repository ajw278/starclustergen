import numpy as np
import sys


#Generate lines from the position data
def GenLines(x,y,z=[],Nsteps=0):
	if len(z) == 0:
		linedata = np.empty((2, Nsteps))
		for index in range(0,Nsteps):
			linedata[:,index] = x[index],y[index]
	else:
		linedata = np.empty((3, Nsteps))
		for index in range(0,Nsteps):
			linedata[:,index] = x[index],y[index],z[index]

	return linedata

def update_lines(num, dataLines, lines, times, text, Nsteps, dim=3):
	if dim==3:
		istar =0
		for line, data in zip(lines,dataLines):
			line.set_data(data[0:2, :num])
			line.set_3d_properties(data[2,:num])
			text.set_text('$t = $ %5.3f' %times[num])
			"""
			startxt[istar].set_position((data[0, num],data[1,num],data[2, num]))
			startxt[istar].update_bbox_position_size(startxt[istar])
			istar+=1"""
	elif dim==2:
		for line, data in zip(lines,dataLines):
			line.set_data(data[0:2, :num])
			text.set_text('$t = $ %5.3f' %times[num])
	else:
		print("Dimensions not correctly defined during line update.")
		sys.exit()
	return lines
