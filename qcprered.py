"""
Module to generate quality-control (QC) checkplots for the prereduction of KiDS, on a run-by-run basis.

Run `python qcpredred.py -h` for help.

In case of questions, contact me, Malte Tewes, mtewes at astro.uni-bonn.de 
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
	
	def keep_last(self, lastn):
		"""Keep only the lastn last runids
		
		"""
		if lastn is not None:
			self.runids = self.runids[-lastn:]
			logger.info("Keeping only the last {} run-ids: {}".format(lastn, self.runids))			

	def keep_only(self, singlerunid=None):
		"""Keep only one specific runid
		
		"""
		if singlerunid is not None:
			if singlerunid in self.runids:
				self.runids = [singlerunid]
			else:
				logger.info("Can't keep specific run-id '{}', it was not found!".format(singlerunid))
		

	@classmethod
	def explore(cls, dirpath, lastn=None, singlerunid=None):
		"""Returns a FileStructure object containing the runids found in the dirpath.
		
		If lastn is not None, I'll consider only the last lastn runids (useful for tests)
		
		If runid is specified, I'll keep only this specific runid (if found)
		"""
		
		dirpaths = sorted(glob.glob(os.path.join(dirpath, "*")))
		
		logger.info("Located {} directories".format(len(dirpaths)))
		
		pattern = r"/.*/run_(.*)"
		matches = [re.match(pattern, dirpath) for dirpath in dirpaths]
		runids = sorted([match.group(1) for match in matches if match != None])
		
		logger.info("Matched {} run-ids".format(len(runids)))
			
		newfs = cls(dirpath, runids)
		newfs.keep_last(lastn)
		newfs.keep_only(singlerunid)
		
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
	
	
	def make_png(self, pngpath, kind="FULL", scale=1000, pixelbin=10, binmethod="mean", z1=None, z2=None, title=None, subtitle=None):
		"""	
		
		- pngpath: path to the png to be saved
		- kind: "BIAS" or "DARK" or "FLAT". Selects different methods to define the grayscale.
		- scale is in pixel per inch
		- pixelbin is in pixel
		- binmethod is either "mean" (default) or "max" or "min"
		- z1 and z2 give the cuts
		- title: Is written on top of the png.
		- subtitle: idem
		
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
			
			try:
				chippath = os.path.join(self.dirpath, self.filename_template.format(chipid))
				ia = f2n.read_fits(chippath)
				is_real_data = True
			except IOError:
				logger.info("Could not find '{}', usign zero image...".format(chippath))
				ia = np.zeros((self.chip_width, self.chip_height))
				is_real_data = False
			
			si = f2n.SkyImage(ia)
			si.rebin(pixelbin, method=binmethod)
			
			if kind == "FLAT" and is_real_data:
				si.data /= np.median(si.data)
			
			if is_real_data:
				si.set_z(z1, z2)
			else:
				si.set_z(0.0, 1.0)
				
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
		


def update_checkmos(kidsdir, workdir, filtername=None, kind="BIAS", lastn=None, redo=False, singlerunid=None):
	"""Update the mosaic-checkplots for the specified kind of files.
	
	kind can be "BIAS", "DARK", "SKYFLAT", "SUPERFLAT"
	
	lastn: if specified, I only run on the lastn last runids (for tests).
	"""
	
	logger.info("Updating mosaic-checkplots for {}...".format(kind))
	
	if (filtername is None) and (not kind in ["BIAS", "DARK"]):
		raise ValueError("Specify a filtername!")
	
	if kind == "BIAS":
		firstdir = "BIAS"
		seconddir = "BIAS"
		pngkind = "BIAS"
		filename_template = "BIAS_{}.fits"
		title = "BIAS"
		binmethod="max"
		z1 = 0.0
		z2 = 4.0
		subtitle = "linear [0, 4]"
		
	elif kind == "DARK":
		firstdir = "DARK"
		seconddir = "DARK"
		pngkind = "DARK"
		filename_template = "DARK_{}.fits"
		title = "DARK"
		subtitle = "linear [-1, 1]"
		binmethod="mean"
		z1 = -1.0
		z2 = 1.0
		
	elif kind == "SKYFLAT":
		firstdir = filtername
		seconddir = "SKYFLAT_{}".format(filtername)
		pngkind = "FLAT"
		filename_template = "SKYFLAT_{}_{{}}.fits".format(filtername)
		title = "SKYFLAT_{}".format(filtername)
		binmethod="mean"
		if filtername in ["u_SDSS", "i_SDSS", "z_SDSS"]:
			z1 = 0.9
			z2 = 1.1	
		else:
			z1 = 0.97
			z2 = 1.03	
		subtitle = "linear [{}, {}]*med(chip)".format(z1, z2)
		
		
	elif kind == "SUPERFLAT":
		firstdir = filtername
		seconddir = "SCIENCE_{}".format(filtername)
		pngkind = "FLAT"
		filename_template = "SCIENCE_{}_{{}}.fits".format(filtername)
		title = "SUPERFLAT SCIENCE_{}".format(filtername)
		binmethod="mean"
		if filtername in ["z_SDSS"]:
			z1 = 0.8
			z2 = 1.2	
		else:
			z1 = 0.97
			z2 = 1.03	
		subtitle = "linear [{}, {}]*med(chip)".format(z1, z2)


	else:
		raise ValueError("Not implemented.")
	
	fs = FileStructure.explore(os.path.join(kidsdir, firstdir), lastn=lastn, singlerunid=singlerunid)
	logger.info("Starting to loop over {} run-ids...".format(len(fs.runids)))
	
	if filtername is None:
		outdir = os.path.join(workdir, kind)
	else:
		outdir = os.path.join(workdir, "{}_{}".format(kind, filtername))
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	for runid in fs.runids:
		
		fitsdir = os.path.join(kidsdir, firstdir, "run_{}".format(runid), seconddir)
		outpath = os.path.join(outdir, "{}.png".format(runid))
	
		if not redo:
			if os.path.exists(outpath):
				continue
		
		checkmos = CheckMos(fitsdir, filename_template,
			chip_width=2040, chip_height=4050, n_chip_horizontal=8, n_chip_vertical=4)
		checkmos.make_png(outpath, kind=pngkind, scale=3000, pixelbin=10, binmethod=binmethod,  z1=z1, z2=z2, title=title+" {}".format(runid), subtitle=subtitle)


def update_illum_correction(kidsdir, workdir, filtername, lastn=None, redo=False, singlerunid=None):
	"""This just copies the existing png"""
	
	logger.info("Updating illumination-correction plots for filter '{}'...".format(filtername))

	fs = FileStructure.explore(os.path.join(kidsdir, filtername), lastn=lastn, singlerunid=singlerunid)
	
	subworkdir = os.path.join(workdir, "ILLUMCOR_"+filtername)
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)
	
	for runid in fs.runids:
		infilepath = os.path.join(kidsdir, filtername, "run_{}".format(runid), "STANDARD_"+filtername, "illum_correction_0", "residuals.png")
		outfilepath = os.path.join(subworkdir, "{}.png".format(runid))
		if not redo:
			if os.path.exists(outfilepath):
				continue
		logger.info("Copying '{}'...".format(infilepath))
		try:
			shutil.copy(infilepath, outfilepath)
		except IOError:
			logger.info("File '{}' could not be read, using dummy png instead...".format(infilepath))
			shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "300px-No_image_available.svg.png"), outfilepath)

def update_zeropoint_calib(kidsdir, workdir, filtername, lastn=None, redo=False, singlerunid=None):
	"""This crops the existing png"""
	
	logger.info("Updating zeropoint-calib plots for filter '{}'...".format(filtername))

	fs = FileStructure.explore(os.path.join(kidsdir, filtername), lastn=lastn, singlerunid=singlerunid)
	
	subworkdir = os.path.join(workdir, "ZPCALIB_"+filtername)
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)
	
	for runid in fs.runids:
		infilepath = os.path.join(kidsdir, filtername, "run_{}".format(runid), "STANDARD_"+filtername, "calib", "night_0_{}_result.png".format(filtername))	
		outfilepath = os.path.join(subworkdir, "{}.png".format(runid))
		if not redo:
			if os.path.exists(outfilepath):
				continue
		
		logger.info("Croping '{}'...".format(infilepath))
		#shutil.copy(infilepath, outfilepath)
		
		# Using os.system to ensure compatibility with old pythons...
		if os.path.exists(infilepath):
			cmd = "convert {} -crop '1386x381+110+393' -resize '70%' {}".format(infilepath, outfilepath)
			os.system(cmd)
		else:
			logger.info("File '{}' could not be found, using dummy png instead...".format(infilepath))
			shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "300px-No_image_available.svg.png"), outfilepath)




def update_composite(workdir, filtername, lastn=None, redo=False, singlerunid=None):
	
	logger.info("Updating composite images...")
	
	subworkdir = os.path.join(workdir, "COMPOSITE_{}".format(filtername))
	if not os.path.exists(subworkdir):
		os.makedirs(subworkdir)

	# Determining run-ids to be processed	
	runids = sorted(list(set([os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(workdir, "BIAS", "*.png"))])))
	if not redo:
		done_runids = sorted(list(set([os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(workdir, "COMPOSITE", "*.png"))])))
		runids = sorted(list(set(runids) - set(done_runids)))
	
	fs = FileStructure(".", runids)
	fs.keep_last(lastn)
	fs.keep_only(singlerunid)

	frames = ["BIAS", "DARK", "ILLUMCOR_"+filtername, "SKYFLAT_"+filtername, "SUPERFLAT_"+filtername, "ZPCALIB_"+filtername]
	
	
	for runid in fs.runids:
		logger.info("Assembling composite for '{}'...".format(runid))
		all_frames_available = True
		for frame in frames:
			framepath = os.path.join(workdir, frame, "{}.png".format(runid))
			if not os.path.exists(framepath):
				all_frames_available = False
				logger.warning("Cannot assemble composite for '{}', as file '{}' does not exist.".format(runid, framepath))
		
		if not all_frames_available:
			continue
				
		outpath = os.path.join(subworkdir, "{}.png".format(runid))
		inpaths = [os.path.join(workdir, frame, "{}.png".format(runid)) for frame in frames]
		inpaths_txt = " ".join(inpaths)
		
		cmd = "montage -background '#DDDDDD' -geometry +2+2 -tile x2 {} {}".format(inpaths_txt, outpath)
		os.system(cmd)
		cmd = "convert -crop '2010x1086+0+0' {} {}".format(outpath, outpath)
		os.system(cmd)
		
		logger.info("Wrote '{}'".format(outpath))

	

def update_all(kidsdir, workdir, filternames=None, lastn=None, redo=False, singlerunid=None):
	"""Generates QC images for all available runs
	
	"""
	logger.info("Updating QC images in '{}' based on the content of '{}'...".format(workdir, kidsdir))
	
	if not filternames:
		filternames = []
		logger.warning("No filters specified, will not update all plots!")

	
	# Masterbias
	update_checkmos(kidsdir, workdir, kind="BIAS", lastn=lastn, redo=redo, singlerunid=singlerunid)
	
	# Masterdark
	update_checkmos(kidsdir, workdir, kind="DARK", lastn=lastn, redo=redo, singlerunid=singlerunid)
	
	for filtername in filternames:
	
		logger.info("Starting updates of filter '{}'...".format(filtername))
		
		# Skyflat
		update_checkmos(kidsdir, workdir, kind="SKYFLAT", filtername=filtername, lastn=lastn, redo=redo, singlerunid=singlerunid)

		# Superflat
		update_checkmos(kidsdir, workdir, kind="SUPERFLAT", filtername=filtername, lastn=lastn, redo=redo, singlerunid=singlerunid)

		# Illumination correction
		update_illum_correction(kidsdir, workdir, filtername=filtername, lastn=lastn, redo=redo, singlerunid=singlerunid)

		# Zp calib
		update_zeropoint_calib(kidsdir, workdir, filtername=filtername, lastn=lastn, redo=redo, singlerunid=singlerunid)

		# And the composite
		update_composite(workdir, filtername=filtername, lastn=lastn, redo=redo, singlerunid=singlerunid)
	
	logger.info("Done with all updates")
	


def main():
	
	import argparse
	
	allfilters = ["u", "g", "r", "i", "z"]
	

	parser = argparse.ArgumentParser(description='Make QC checkplots of the pre-reduction')
	parser.add_argument("-w", "--workdir", default=None, help="Path to a directory in which to write the QC images (output)")
	parser.add_argument("--kidsdir", default=None, help="Path to directory containing the KIDS pre-reduction (input)")
	parser.add_argument("-n", "--lastn", default=None, type=int, help="Process only the last LASTN runs (good for tests)")
	parser.add_argument("-r", "--redo", action="store_true", help="Reprocess and overwrite checkplots even if they already exist")
	parser.add_argument("-f", "--filter", default=allfilters, choices=allfilters, help="Which filter among {u, g, r, i, z} to process (default: all)")
 	parser.add_argument("-s", "--singlerunid", default=None, help="Run only on this particular run-id (e.g.,  '17_07_f')")
              
	args = parser.parse_args()
	args.filter = list(args.filter) # To have ["r"] instead of "r".
	args.filter = ["{}_SDSS".format(f) for f in args.filter] # Adding "_SDSS" to all filternames.
	
	# Some default values
	if not args.workdir:
		args.workdir = "/vol/fohlen11/fohlen11_1/mtewes/KiDS_prered_QC"
	if not args.kidsdir:
		args.kidsdir = "/vol/kraid2/kraid2/terben/KIDS_V1.0.0"
	
		
	#print(args)
	update_all(kidsdir=args.kidsdir, workdir=args.workdir, lastn=args.lastn, redo=args.redo, filternames=args.filter, singlerunid=args.singlerunid)
	

if __name__ == "__main__":
    # execute only if run as a script
    main()


