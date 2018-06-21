import os
import glob
import re
import f2n
import matplotlib
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileStructure(object):
	
	def __init__(self, dirpath=None, runids=None):
		self.dirpath = dirpath
		self.runids = runids
	
	
	@classmethod
	def explore(cls, dirpath):
		
		dirpaths = glob.glob(os.path.join(dirpath, "*"))
		
		logger.info("Located {} directories".format(len(dirpaths)))
		
		pattern = r"/.*/run_(.*)"
		matches = [re.match(pattern, dirpath) for dirpath in dirpaths]
		runids = [match.group(1) for match in matches if match != None]
		
		logger.info("Matched {} run-ids".format(len(runids)))
		
		return cls(dirpath, runids)
		
	

class CheckImg(object):
	
	def __init__(self, fitspath, pngpath):
		self.fitspath = fitspath
		self.pngpath = pngpath
		
	def makepng(self):
		
		logger.info("Making png of '{}'...".format(self.fitspath))
		
		image_array = f2n.read_fits(self.fitspath)
		sf = f2n.SimpleFigure(image_array, z1=0.0, z2=1.0, scale=0.1, withframe=False)
		sf.draw()
		sf.save_to_file(self.pngpath)
		

class CheckMos(object):
	def __init__(self, dirpath, pngpath):
		self.dirpath = dirpath
		self.pngpath = pngpath
		
	
	def makepng(self):
		
		logger.info("Making png of '{}'...".format(self.dirpath))
		
		chipids = list(range(1, 33))
		chipposs = [25, 26, 27, 28, 17, 18, 19, 20, 9, 10, 11, 12]
		
		chipids = [1, 7, 10]
		chipposs = [25, 19, 10]
		
		chipw = 2040
		chiph = 4050
		scale = 3000.0
		figw = chipw * 8 / scale
		figh = chiph * 4 / scale
		
		
		subplotpars = matplotlib.figure.SubplotParams(
			left = 0.0,
			right = 1.0,
			bottom = 0.0,
			top = 1.0,
			wspace = 0.01,
			hspace = 0.01
		)
		fig = plt.figure(figsize=(figw, figh), subplotpars=subplotpars)
		
		for (i, chippos) in enumerate(chipposs):
			chipid = chipids[i]
			ax = fig.add_subplot(4, 8, chippos)
			ax.set_axis_off()
			
			ia = f2n.read_fits(os.path.join(self.dirpath, "BIAS_{}.fits".format(chipid)))
			si = f2n.SkyImage(ia, 0.0, 1.0)
			si.rebin(10, method="max")
			f2n.draw_sky_image(ax, si)
		
			ax.text(0.5, 0.1, "{}".format(chipid),
				horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
				color="red", fontsize=18)
		
		
		#fig.savefig(self.pngpath, bbox_inches='tight')
		fig.savefig(self.pngpath)
		
		
		"""
		for 
		
		
		image_array = f2n.read_fits(self.fitspath)
		sf = f2n.SimpleFigure(image_array, z1=0.0, z2=1.0, scale=0.1, withframe=False)
		sf.draw()
		sf.save_to_file(self.pngpath)
		"""


"""
biasdir = "/vol/kraid2/kraid2/terben/KIDS_V1.0.0/BIAS"
rs = FileStructure.explore(biasdir)

fitspath = "/vol/kraid2/kraid2/terben/KIDS_V1.0.0/BIAS/run_16_06_f/BIAS/BIAS_7.fits"
pngpath = "test.png"
img = CheckImg(fitspath, pngpath)
img.makepng()
"""

biasdir = "/vol/kraid2/kraid2/terben/KIDS_V1.0.0/BIAS/run_16_06_f/BIAS/"

cm = CheckMos(biasdir, "test.png")
cm.makepng()




