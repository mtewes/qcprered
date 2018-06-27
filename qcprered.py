"""
Module to generate quality-control (QC) checkplots for the prereduction of KiDS, on a run-by-run basis.

Run `python qcpredred.py -h` for help.

In case of questions, contact me, Malte Tewes, at mtewes@astro.uni-bonn.de 
"""

import os
import glob
import shutil
import re
import sky_image_plot as f2n
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FileStructure(object):
	"""Class to represent the filestructure of the different runs.
	
	OK, the author realized afterwards that the motivation for such a class was a total overkill.
	So we're just getting a list of runids here.
	"""
	
	def __init__(self, dirpath=None, runids=None):
		self.dirpath = dirpath
		self.runids = runids
	
	def keep_only(self, lastn):
		"""Keep only the lastn last runids
		
		"""
		if lastn is not None:
			self.runids = self.runids[-lastn:]
			logger.info("Keeping only the last {} run-ids: {}".format(lastn, self.runids))
			

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
			
		newfs = cls(dirpath, runids)
		newfs.keep_only(lastn)
		
		return newfs	
	
	

class CheckMos(object):
	def __init__(self, dirpath, filename_template, chip_width, chip_height, n_chip_horizontal, n_chip_vertical):
		"""
		Class to produce mosaic checkplots for frames taken by multi-chip cameras.
		
		- dirpath: path to the directory containing the FITS files from all the chips.
		- filename_template: string like "BIAS_{}.fits" that will be used with a glob to get the FITS files. The "{}" gets replaced by the chip number.
			
		"""
		self.dirpath = dirpath
		self.filename_template = filename_template
	
		self.chip_width = chip_width
		self.chip_height = chip_height
		self.n_chip_horizontal = n_chip_horizontal
		self.n_chip_vertical = n_chip_vertical
	
		self.set_theli_layout()
	
	
	def set_theli_layout(self):
		"""
		Sets 2 lists: one with the chip numbers, and one that contains the "third arguments" to matplotlib subplot
		"""
		logger.debug("Setting chip layout for {} x {} (horizontal x vertical) chips...".format(self.n_chip_horizontal, self.n_chip_vertical))
		
		self.chip_ids = list(range(1, self.n_chip_horizontal * self.n_chip_vertical + 1))
		chiplines = []	
		for i in range(self.n_chip_vertical):
			chiplines.append(list(range(i*self.n_chip_horizontal+1, (i+1)*self.n_chip_horizontal+1)))
		self.chip_positions = [p for chipline in chiplines[::-1] for p in chipline]
		
		logger.debug("chip_positions: {}".format(self.chip_positions))
			
		assert len(self.chip_positions) == len(self.chip_ids)
		
		# This is how Omegacam is documented:
		#self.chip_ids = list(range(1, 33))
		#self.chip_positions = [25, 26, 27, 28, 17, 18, 19, 20, 9, 10, 11, 12, 1, 2, 3, 4, 29, 30, 31, 32, 21, 22, 23, 24, 13, 14, 15, 16, 5, 6, 7, 8]
	
	
	def make_png(self, pngpath, kind="FULL", scale=1000, pixelbin=10, title=None, subtitle=None):
		"""	
		
		- pngpath: path to the png to be saved
		- kind: "BIAS" or "DARK" or "FLAT". Selects different methods to define the grayscale.
		- scale is in pixel per inch
		- pixelbin is in pixel
		- title: Is written on top of the png.
		
		"""
		
		logger.info("Making png of '{}'...".format(self.dirpath))
			
		figw = self.n_chip_horizontal * self.chip_width / scale
		figh = self.n_chip_vertical * self.chip_height / scale
		
		subplotpars = matplotlib.figure.SubplotParams(
			left = 0.0,
			right = 1.0,
			bottom = 0.0,
			top = 1.0,
			wspace = 0.0,
			hspace = 0.0
		)
		fig = plt.figure(figsize=(figw, figh), subplotpars=subplotpars)
		
		for (i, chippos) in enumerate(self.chip_positions):
			chipid = self.chip_ids[i]
			ax = fig.add_subplot(self.n_chip_vertical, self.n_chip_horizontal, chippos)
			ax.set_axis_off()
			
			ia = f2n.read_fits(os.path.join(self.dirpath, self.filename_template.format(chipid)))
			si = f2n.SkyImage(ia)
			if kind == "BIAS":
				si.rebin(pixelbin, method="max")
				si.set_z(0.0, 4.0)
				if not subtitle: subtitle = "linear [0, 4]"
			elif kind == "DARK":
				si.rebin(pixelbin)
				si.set_z(-1.0, 1.0)
				if not subtitle: subtitle = "linear [-1, 1]"
			elif kind == "FLAT":
				si.rebin(pixelbin)
				si.data /= np.median(si.data)
				si.set_z(0.98, 1.02)
				if not subtitle: subtitle = "linear [0.98, 1.02]*med(chip)"
			else:
				si.rebin(pixelbin)
				si.set_z(0, 65536)
				
			f2n.draw_sky_image(ax, si)
		
			ax.text(0.5, 0.1, "{}".format(chipid),
				horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
				color="yellow", fontsize=12)

		if title != None:
			fig.text(0.5, 0.95, title,
				horizontalalignment='center', verticalalignment='center', color="yellow", fontsize=16)
		
		if subtitle != None:
			fig.text(0.5, 0.9, subtitle,
				horizontalalignment='center', verticalalignment='center', color="yellow", fontsize=12)
		
		
		
		#fig.savefig(self.pngpath, bbox_inches='tight')
		fig.savefig(pngpath)
		


def update_checkmos(kidsdir, workdir, kind="BIAS", lastn=None, redo=False):
	"""Update the mosaic-checkplots for the specified kind of files.
	
	kind can be "BIAS", "DARK", "SKYFLAT", "SUPERFLAT"
	
	lastn: if specified, I only run on the lastn last runids (for tests).
	"""
	
	logger.info("Updating mosaic-checkplots for {}...".format(kind))
	
	if kind == "BIAS":
		firstdir = "BIAS"
		seconddir = "BIAS"
		pngkind = "BIAS"
		filename_template = "BIAS_{}.fits"
		title = "BIAS"
	elif kind == "DARK":
		firstdir = "DARK"
		seconddir = "DARK"
		pngkind = "DARK"
		filename_template = "DARK_{}.fits"
		title = "DARK"
	elif kind == "SKYFLAT":
		firstdir = "r_SDSS"
		seconddir = "SKYFLAT_r_SDSS"
		pngkind = "FLAT"
		title = "SKYFLAT_r_SDSS"
		filename_template = "SKYFLAT_r_SDSS_{}.fits"
	elif kind == "SUPERFLAT":
		firstdir = "r_SDSS"
		seconddir = "SCIENCE_r_SDSS"
		pngkind = "FLAT"
		title = "SUPERFLAT SCIENCE_r_SDSS"
		filename_template = "SCIENCE_r_SDSS_{}.fits"
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
		
		checkmos = CheckMos(fitsdir, filename_template,
			chip_width=2040, chip_height=4050, n_chip_horizontal=8, n_chip_vertical=4)
		checkmos.make_png(outpath, kind=pngkind, scale=3000, pixelbin=10, title=title)


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
	
	fs = FileStructure(".", runids)
	fs.keep_only(lastn)

	frames = ["BIAS", "DARK", "ILLUMCOR", "SKYFLAT", "SUPERFLAT", "ZPCALIB"]
	
	
	for runid in fs.runids:
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
		cmd = "convert -crop '2010x1086+0+0' {} {}".format(outpath, outpath)
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

	# Superflat
	update_checkmos(kidsdir, workdir, kind="SUPERFLAT", lastn=lastn, redo=redo)

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


