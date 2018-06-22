"""
Module to generate quality-control (QC) checkplots for the prereduction of KiDS, on a run-by-run basis

In case of questions, contact the author, Malte Tewes, at mtewes@astro.uni-bonn.de 
"""

import os
import glob
import shutil
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
	def explore(cls, dirpath, lastn=None):
		"""Returns a FileStructure object containing the runids found in the dirpath.
		
		If lastn is not None, I'll consider only the last lastn runids (useful for tests)
		"""
		
		dirpaths = sorted(glob.glob(os.path.join(dirpath, "*")))
		
		logger.info("Located {} directories".format(len(dirpaths)))
		
		pattern = r"/.*/run_(.*)"
		matches = [re.match(pattern, dirpath) for dirpath in dirpaths]
		runids = sorted([match.group(1) for match in matches if match != None])
		
		logger.info("Matched {} run-ids".format(len(runids)))
		
		if lastn:
			runids = runids[-lastn:]
			logger.info("Keeping only the last {} run-ids: {}".format(lastn, runids))
		
		return cls(dirpath, runids)
		
	

class CheckMos(object):
	def __init__(self, dirpath, pngpath, kind):
		self.dirpath = dirpath
		self.pngpath = pngpath
		self.kind = kind # BIAS or DARK or SKYFLAT
		
	
	def make_png(self):
		
		logger.info("Making png of '{}'...".format(self.dirpath))
		
		# Arrangement of the 32 OmegaCAM chips
		chipids = list(range(1, 33))
		chipposs = [25, 26, 27, 28, 17, 18, 19, 20, 9, 10, 11, 12, 1, 2, 3, 4, 29, 30, 31, 32, 21, 22, 23, 24, 13, 14, 15, 16, 5, 6, 7, 8]
		
		#chipids = [1, 7, 10]
		#chipposs = [25, 19, 10]
		
		assert len(chipposs) == len(chipids)
		
		if self.kind == "SKYFLAT":
			fitsprefix = "SKYFLAT_r_SDSS"
		else:
			fitsprefix = self.kind
		
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
			wspace = 0.0,
			hspace = 0.0
		)
		fig = plt.figure(figsize=(figw, figh), subplotpars=subplotpars)
		
		for (i, chippos) in enumerate(chipposs):
			chipid = chipids[i]
			ax = fig.add_subplot(4, 8, chippos)
			ax.set_axis_off()
			
			ia = f2n.read_fits(os.path.join(self.dirpath, "{}_{}.fits".format(fitsprefix, chipid)))
			si = f2n.SkyImage(ia)
			if self.kind == "BIAS":
				si.rebin(10, method="max")
				si.set_z(0.0, 4.0)
			elif self.kind == "DARK":
				si.rebin(10, method="mean")
				si.set_z(-1.0, 1.0)
			else:
				si.rebin(10, method="mean")
				si.set_z(10000, 25000)
				
			f2n.draw_sky_image(ax, si)
		
			ax.text(0.5, 0.1, "{}".format(chipid),
				horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
				color="red", fontsize=16)
		
		fig.text(0.5, 0.95, self.kind,
				horizontalalignment='center', verticalalignment='center', color="red", fontsize=18)
		
		
		#fig.savefig(self.pngpath, bbox_inches='tight')
		fig.savefig(self.pngpath)
		
		
		"""
		for 
		
		
		image_array = f2n.read_fits(self.fitspath)
		sf = f2n.SimpleFigure(image_array, z1=0.0, z2=1.0, scale=0.1, withframe=False)
		sf.draw()
		sf.save_to_file(self.pngpath)
		"""


def update_checkmos(kidsdir, workdir, kind="BIAS", lastn=None, redo=False):
	"""Update the mosaic-checkplots for the specified kind of files.
	
	kind can be "BIAS", "DARK", "SKYFLAT"
	
	lastn: if specified, I only run on the lastn last runids (for tests).
	"""
	
	logger.info("Updating mosaic-checkplots for {}...".format(kind))
	
	
	if kind == "SKYFLAT":
		firstdir = "r_SDSS"
		seconddir = "SKYFLAT_r_SDSS"
	else:
		firstdir = kind
		seconddir = kind
	
	fs = FileStructure.explore(os.path.join(kidsdir, firstdir), lastn=lastn)
	logger.info("Starting to loop over {} run-ids...".format(len(fs.runids)))
	
	if not os.path.exists(os.path.join(workdir, kind)):
		os.makedirs(os.path.join(workdir, kind))
	
	for runid in fs.runids:
		
		fitsdir = os.path.join(kidsdir, firstdir, "run_{}".format(runid), seconddir)
		outpath = os.path.join(workdir, kind, "{}.png".format(runid))
		if not redo:
			if os.path.exists(outpath):
				continue
		
		checkmos = CheckMos(fitsdir, outpath, kind)
		checkmos.make_png()


def update_illum_correction(kidsdir, workdir, lastn=None, redo=False):
	"""This just copies the existing png"""
	
	logger.info("Updating illumination-correction plots...")

	fs = FileStructure.explore(os.path.join(kidsdir, "r_SDSS"), lastn=lastn)
	
	subworkdir = os.path.join(workdir, "ILLUMCOR")
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)
	
	for runid in fs.runids:
		infilepath = os.path.join(kidsdir, "r_SDSS", "run_{}".format(runid), "STANDARD_r_SDSS", "illum_correction_0", "residuals.png")
		outfilepath = os.path.join(subworkdir, "{}.png".format(runid))
		if not redo:
			if os.path.exists(outfilepath):
				continue
		logger.info("Copying '{}'...".format(infilepath))
		shutil.copy(infilepath, outfilepath)
		

def update_zeropoint_calib(kidsdir, workdir, lastn=None, redo=False):
	"""This crops the existing png"""
	
	logger.info("Updating zeropoint-calib plots...")

	fs = FileStructure.explore(os.path.join(kidsdir, "r_SDSS"), lastn=lastn)
	
	subworkdir = os.path.join(workdir, "ZPCALIB")
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)
	
	for runid in fs.runids:
		infilepath = os.path.join(kidsdir, "r_SDSS", "run_{}".format(runid), "STANDARD_r_SDSS", "calib", "night_0_r_SDSS_result.png")	
		outfilepath = os.path.join(subworkdir, "{}.png".format(runid))
		if not redo:
			if os.path.exists(outfilepath):
				continue
		
		logger.info("Croping '{}'...".format(infilepath))
		#shutil.copy(infilepath, outfilepath)
		
		# Using os.system to ensure compatibility with old pythons...
		cmd = "convert {} -crop '1386x381+110+393' -resize '70%' {}".format(infilepath, outfilepath)
		os.system(cmd)



def update_composite(workdir, lastn=None, redo=False):
	
	logger.info("Updating composite images...")
	
	subworkdir = os.path.join(workdir, "COMPOSITE")
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)

	# Determining run-ids to be processed	
	runids = sorted(list(set([os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(workdir, "BIAS", "*.png"))])))
	if not redo:
		done_runids = sorted(list(set([os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(workdir, "COMPOSITE", "*.png"))])))
		runids = sorted(list(set(runids) - set(done_runids)))
	if len(runids) == 0:
		logger.info("No new files to process")
		return

	frames = ["BIAS", "DARK", "ILLUMCOR", "SKYFLAT", "ZPCALIB"]
	#frames = ["BIAS", "DARK", "ILLUMCOR", "SKYFLAT"]
	
	
	for runid in runids:
		logger.info("Assembling composite for '{}'...".format(runid))
		for frame in frames:
			framepath = os.path.join(workdir, frame, "{}.png".format(runid))
			if not os.path.exists(framepath):
				logger.warning("Cannot assemble composite for '{}', as file '{}' does not exist.".format(runid, framepath))
				
		outpath = os.path.join(workdir, "COMPOSITE", "{}.png".format(runid))
		inpaths = [os.path.join(workdir, frame, "{}.png".format(runid)) for frame in frames]
		inpaths_txt = " ".join(inpaths)
		
		cmd = "montage -background '#DDDDDD' -geometry +2+2 -tile x2 {} {}".format(inpaths_txt, outpath)
		os.system(cmd)
		cmd = "convert -crop '1744x1086+0+0' {} {}".format(outpath, outpath)
		os.system(cmd)
		
		logger.info("Wrote '{}'".format(outpath))

	

def update_all(kidsdir, workdir, lastn=None, redo=False):
	"""Generates QC images for all available runs
	
	"""
	logger.info("Updating QC images in '{}' based on the content of '{}'...".format(workdir, kidsdir))
	
	# Masterbias
	update_checkmos(kidsdir, workdir, kind="BIAS", lastn=lastn, redo=redo)
	
	# Masterdark
	update_checkmos(kidsdir, workdir, kind="DARK", lastn=lastn, redo=redo)

	# Skyflat
	update_checkmos(kidsdir, workdir, kind="SKYFLAT", lastn=lastn, redo=redo)

	# Illumination correction
	update_illum_correction(kidsdir, workdir, lastn=lastn, redo=redo)

	# Zp calib
	update_zeropoint_calib(kidsdir, workdir, lastn=lastn, redo=redo)

	# And the composite
	update_composite(workdir, lastn=lastn, redo=redo)
	
	logger.info("Done with all updates")


def main():
	
	import argparse

	parser = argparse.ArgumentParser(description='Make QC checkplots of the pre-reduction')
	parser.add_argument("-w", "--workdir", default=None, help="Path to a directory in which the QC stuff can be kept")
	parser.add_argument("--kidsdir", default=None, help="Path to directory containing the KIDS pre-reduction files")
	parser.add_argument("-n", "--lastn", default=None, type=int, help="Process only the last LASTN runs (good for tests)")
	parser.add_argument("-r", "--redo", action="store_true", help="Reprocess checkplots even if they already exist")
               
	args = parser.parse_args()
	
	# Some default values
	if not args.workdir:
		args.workdir = "/vol/fohlen11/fohlen11_1/mtewes/KiDS_prered_QC"
	if not args.kidsdir:
		args.kidsdir = "/vol/kraid2/kraid2/terben/KIDS_V1.0.0"
	
	#print(args)
	update_all(kidsdir=args.kidsdir, workdir=args.workdir, lastn=args.lastn, redo=args.redo)
	

if __name__ == "__main__":
    # execute only if run as a script
    main()


