"""
	Complete Workflow for generating ATS inputs, compatible for ELM domain-soil_columns unstructured-mesh
	
	This workflow provides a complete working example to develop a simulation campaign for integrated hydrology within ATS.
	
	It uses the following datasets:
	
		- NHD Plus for hydrography.
		- 3DEP for elevation
		- NLCD for land cover/transpiration/rooting depths
		- MODIS for LAI
		- GLYHMPS geology data for structural formations
		- Pelletier for depth to bedrock and soil texture information
		- SSURGO for soil data, where available, in the top 2m.
	
	Given some basic inputs (in the next cell) including a {myNAME}, 
	this workflow creates the following files, all of which will reside in output_data:
	
		- Mesh file: {myNAME}.exo, includes all labeled sets
		- Forcing: DayMet data -- daily raster of precip, RH, incoming radiation, etc.
		- {myNAME}_daymet_2010_2011.h5, the DayMet data on this watershed
		- {myNAME}_daymet_CyclicSteadystate.h5, a "typical year" of DayMet, smoothed for spinup purposes, then looped certain number of years
		- Forcing: LAI data -- every 4 days, time series by land cover type of LAI.
		- {myNAME}_LAI_MODIS_transient.h5, the LAI, interpolated and smoothed from the raw MODIS data
		- {myNAME}_LAI_MODIS_CyclicSteadystate.h5, a "typical year" of LAI, smoothed for spinup purposes then looped 10 years
		
	ATS Input files for three runs, intended to be run sequentially:
		- {myNAME}_steadystate.xml the steady-state solution based on uniform application of mean rainfall rate
		- {myNAME}_cyclic_steadystate.xml the cyclic steady state based on typical years
		- {myNAME}_transient.xml the forward model, run from 2010 -- 2011
	
	ELM Input files:
		- domain.lnd.Nx1pt.{myNAME}.nc
		- surfdata_Nx1pt_simyr1850-{myNAME}.nc
		- landuse_timeseries_Nx1pt_hist_simyr1850-2015-{myNAME}.nc
		- ./cpl_bypass_full/ATS-subdaily_{TBOT/QBOT/PRECTmms/PSRF/FSDS/FLDS/WIND}_z01.nc
"""

#########################################################################################################

#--- #----- I. Pacakges Needed -----#

# these can be turned on for development work
#%load_ext autoreload
#%autoreload 2

import netCDF4

# setting up logging first or else it gets preempted by another package
import watershed_workflow.ui
watershed_workflow.ui.setup_logging(1)

import os,sys
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import pandas as pd
import geopandas as gpd
import cftime, datetime
pd.options.display.max_columns = None

import shutil

#--- provide paths to relevant packages for mesh and input file generation 

# Set paths to relevant packages for mesh and input file generation
# Update these paths to match your local installation
#
sys.path.append('/Users/f9y/micromamba/amanzi-ats-tools/seacas-exodus/lib')
sys.path.append('/Users/f9y/micromamba/amanzi-ats-tools/amanzi_xml')
os.environ['AMANZI_SRC_DIR']='/Users/f9y/mygithub/ATS_REPOS/amanzi'
os.environ['ATS_SRC_DIR']='/Users/f9y/mygithub/ATS_REPOS/amanzi/src/physics/ats'

import watershed_workflow 
import watershed_workflow.config
import watershed_workflow.sources
import watershed_workflow.utils
import watershed_workflow.plot
import watershed_workflow.mesh
import watershed_workflow.regions
import watershed_workflow.meteorology
import watershed_workflow.land_cover_properties
import watershed_workflow.resampling
import watershed_workflow.condition
import watershed_workflow.io
import watershed_workflow.sources.standard_names as names

import ats_input_spec
import ats_input_spec.public
import ats_input_spec.io

import amanzi_xml.utils.io as aio
import amanzi_xml.utils.search as asearch
import amanzi_xml.utils.errors as aerrors

#
#-------------------------------------------------------------------------------------
#--- options for using ELM soil column layer structure
ELM_SOILCOLUMN=True
# by default, ELM soil layer number is 15. If  option true, it's 30
# and ELM namelist flag: more_vertlayers = .true.
MORE_VERTLAYERS=False
if ELM_SOILCOLUMN:
	import watershed_workflow.elm_domain as elm_domain
	import watershed_workflow.elm_mksrfdata as elm_mksrfdata
	from types import SimpleNamespace
	import watershed_workflow.elm_metdata as elm_metdata
	# e3sm has a xml file, in which by machine-name, DIN_LOC_ROOT is pre-defined.
	# (TODO this locally or from data server)
	elm_domain.set_e3sm_input('/Users/f9y/e3sm_inputdata')
	DIN_LOC_ROOT='/Users/f9y/e3sm_inputdata'
#-------------------------------------------------------------------------------------



# set the default figure size for notebooks
plt.rcParams["figure.figsize"] = (8, 6)


print(watershed_workflow.__file__)

#########################################################################################################

#--- #----- II. Input: Parameters and other source data -----#

"""
Note, this section will need to be modified for other runs of this workflow in other regions.
"""

# Force Watershed Workflow to pull data from this directory rather than a shared data directory.
# This picks up the myname-specific datasets set up here to avoid large file downloads for 
# demonstration purposes.
#
def splitPathFull(path):
    """
    Splits an absolute path into a list of components such that
    os.path.join(*splitPathFull(path)) == path
    """
    parts = []
    while True:
        head, tail = os.path.split(path)
        if head == path:  # root on Unix or drive letter with backslash on Windows (e.g., C:\)
            parts.insert(0, head)
            break
        elif tail == path:  # just a single file or directory
            parts.insert(0, tail)
            break
        else:
            parts.insert(0, tail)
            path = head
    return parts

cwd = splitPathFull(os.getcwd())

#--- REMOVE THIS PORTION OF THE CELL for general use outside of myname -- this is just locating 
#--- the working directory within the WW directory structure
wkdir_name = 'GCREW'
if cwd[-1] == wkdir_name:
    pass
elif cwd[-1] == 'examples':
    cwd.append(wkdir_name)
else:
    cwd.extend(['examples',wkdir_name])
#--- END REMOVE THIS PORTION

#--- A few user defined options
# name of watershed or whatever for your cases
myname = wkdir_name.lower()+'-atselm'

# Note, this directory is where downloaded data will be put as well
data_dir = os.path.join(*(cwd + ['input_data',]))
def toInput(filename):
    return os.path.join(data_dir, filename)

output_dir = os.path.join(*(cwd + ['output_data',]))
def toOutput(filename):
    return os.path.join(output_dir, filename)

work_dir = os.path.join(*cwd)
def toWorkingDir(filename):
    return os.path.join(work_dir, filename)
   
   
#--- Set the data directory to the local space to get the locally downloaded files
# REMOVE THIS CELL for general use outside of my case area
watershed_workflow.config.setDataDirectory(data_dir)


## Parameters cell -- this provides all parameters that can be changed via pipelining to generate a new watershed. 
mycase_shapefile = os.path.join(data_dir, 'gcrew_basin31.shp')

# Geometric parameters
# -- parameters to clean and reduce the river network prior to meshing
simplify = 10                   # length scale to target average edge 
ignore_small_rivers = 2         # remove rivers with fewer than this number of reaches -- important for NHDPlus HR 
prune_by_area_fraction = 0.01   # prune any reaches whose contributing area is less than this fraction of the domain

# -- mesh triangle refinement control
refine_d0 = 200
refine_d1 = 600

refine_L0 = 50   # this is very sensative to cell number
refine_L1 = 200

refine_A0 = refine_L0**2 / 2
refine_A1 = refine_L1**2 / 2

# Simulation control
# - note that we use the NoLeap calendar, same as DayMet.  Simulations are typically run over the "water year"
#   which starts August 1.
start = cftime.DatetimeNoLeap(2003,1,1)  # modis LAI starts from 2002-07-04
end = cftime.DatetimeNoLeap(2022,1,1)  # ROAC v.1.1 met. data ends by 2021-12-31  

nyears_cyclic_steadystate = 4   # how many years to run spinup

# Global Soil Properties
min_porosity = 0.05 # minimum porosity considered "too small"
max_permeability = 1.e-10 # max value considered "too permeable"
max_vg_alpha = 1.e-3 # max value of van Genuchten's alpha -- our correlation is not valid for some soils

# a dictionary of output_filenames -- will include all filenames generated
output_filenames = {}

# Note that, by default, we tend to work in the DayMet CRS because this allows us to avoid
# reprojecting meteorological forcing datasets.
crs = watershed_workflow.crs.default_crs

# get the shape and crs of the shape
mycase_source = watershed_workflow.sources.ManagerShapefile(mycase_shapefile)
mycase = mycase_source.getShapes(out_crs=crs)
mycase.rename(columns={'AREA' : names.AREA}, inplace=True)

#--- set up a dictionary of source objects
#
# Data sources, also called managers, deal with downloading and parsing data files from a variety of online APIs.
sources = watershed_workflow.sources.getDefaultSources()
sources['hydrography'] = watershed_workflow.sources.hydrography_sources['NHDPlus HR']

#
# This demo uses a few datasets that have been clipped out of larger, national
# datasets and are distributed with the code.  This is simply to save download
# time for this simple problem and to lower the barrier for trying out
# Watershed Workflow.  A more typical workflow would delete these lines (as 
# these files would not exist for other watersheds).
#
# The default versions of these download large raster and shapefile files that
# are defined over a very large region (globally or the entire US).
#
# DELETE THIS SECTION for non-mycase runs
dtb_file = os.path.join(data_dir, 'soil_structure', 'DTB', 'DTB.tif')
if ELM_SOILCOLUMN:
	# a modified DTB datasets, in which including less than 2m or so soils
	dtb_file = os.path.join(DIN_LOC_ROOT, 'lnd/clm2/surfdata_map','high_res','soildtb_30x30sec_nwh_c220613.tif')

# The Pelletier DTB map is not particularly accurate at mycase -- the SoilGrids map seems to be better.
# Here we will use a clipped version of that map.
sources['depth to bedrock'] = watershed_workflow.sources.ManagerRaster(dtb_file)

# END DELETE THIS SECTION

# log the sources that will be used here
watershed_workflow.sources.logSources(sources)


#########################################################################################################

#--- #----- III. Basin Geometry -----#

"""
In this section, we choose the basin, the streams to be included in the stream-aligned mesh, and make sure that all are resolved discretely at appropriate length scales for this work.

"""

#--- III-1. the Watershed

# Construct and plot the WW object used for storing watersheds
watershed = watershed_workflow.split_hucs.SplitHUCs(mycase)

#--- III-2. the Rivers
# download/collect the river network within that shape's bounds
reaches = sources['hydrography'].getShapesByGeometry(watershed.exterior, crs, out_crs=crs)
rivers = watershed_workflow.river_tree.createRivers(reaches, method='hydroseq')
watershed_orig, rivers_orig = watershed, rivers

# keeping the originals for plotting comparisons
def createCopy(watershed, rivers):
    """To compare before/after, we often want to create copies.  Note in real workflows most things are done in-place without copies."""
    return watershed.deepcopy(), [r.deepcopy() for r in rivers]
    

watershed, rivers = createCopy(watershed_orig, rivers_orig)

# simplifying -- this sets the discrete length scale of both the watershed boundary and the rivers
watershed_workflow.simplify(watershed, rivers, refine_L0, refine_L1, refine_d0, refine_d1)

# simplify may remove reaches from the rivers object
# -- this call removes any reaches from the dataframe as well, signaling we are all done removing reaches
#
# ETC: NOTE -- can this be moved into the simplify call?
for river in rivers:
    river.resetDataFrame()

# Now that the river network is set, find the watershed boundary outlets
for river in rivers:
    watershed_workflow.hydrography.findOutletsByCrossings(watershed, river)

#########################################################################################################

#--- #----- IV.  Mesh Geometry -----#

"""
Discretely create the stream-aligned mesh. Download elevation data, and condition the mesh discretely to make for better topography.

"""

# Refine triangles if they get too acute
min_angle = 32 # degrees

# width of reach by stream order (order:width)
widths = dict({1:8,2:12,3:16,4:20})

#--- create the mesh (surface m2)
m2, areas, dists = watershed_workflow.tessalateRiverAligned(watershed, rivers, 
                                                            river_width=widths,
                                                            refine_min_angle=min_angle,
                                                            refine_distance=[refine_d0, refine_A0, refine_d1, refine_A1],
                                                            diagnostics=True)

# prepartition to maintain ordering
m2 = m2.partition(8, True)

# get a raster for the elevation map, based on 3DEP
#dem = sources['DEM'].getDataset(watershed.exterior.buffer(10), watershed.crs)['dem']

# locally available raster DEM
dem_raster = os.path.join(data_dir,'topography','ned19_n39x00_w076x75_md_dnr_lidar2004_gcrew.tif') 
sources['DEM'] = watershed_workflow.sources.ManagerRaster(dem_raster)
dem = sources['DEM'].getDataset(watershed.exterior.buffer(10), watershed.crs)['band_1']
m2.cell_data['DEM'] = watershed_workflow.getDatasetOnMesh(m2, dem, method='linear')

#--- provide surface mesh elevations
watershed_workflow.elevate(m2, dem)

# In the pit-filling algorithm, we want to make sure that river corridor is not filled up. Hence we exclude river corridor cells from the pit-filling algorithm.
# hydrologically condition the mesh, removing pits
river_mask=np.zeros((len(m2.conn)))
for i, elem in enumerate(m2.conn):
    if not len(elem)==3:
        river_mask[i]=1
watershed_workflow.condition.fillPitsDual(m2, is_waterbody=river_mask)

# There are a range of options to condition river corridor mesh. We hydrologically condition the river mesh, ensuring unimpeded water flow in river corridors by globally adjusting flowlines to rectify artificial obstructions from inconsistent DEM elevations or misalignments. Please read the documentation for more information

# conditioning river mesh
#
# adding elevations to the river tree for stream bed conditioning
watershed_workflow.condition.setProfileByDEM(rivers, dem)

# conditioning the river mesh using NHD elevations
watershed_workflow.condition.conditionRiverMesh(m2, rivers[0])

# plotting surface mesh with elevations
fig, ax = plt.subplots()
ax2 = ax.inset_axes([0.65,0.05,0.3,0.5])
cbax = fig.add_axes([0.05,0.02,0.9,0.04])

mp = m2.plot(facecolors='elevation', edgecolors=None, ax=ax, linewidth=0.5, colorbar=False)
cbar = fig.colorbar(mp, orientation="horizontal", cax=cbax)
ax.set_title('surface mesh with elevations')
ax.set_aspect('equal', 'datalim')

mp2 = m2.plot(facecolors='elevation', edgecolors='white', ax=ax2, colorbar=False)
ax2.set_aspect('equal', 'datalim')

xlim = (np.min(m2.coords[:,0]), np.max(m2.coords[:,0]))
ylim = (np.min(m2.coords[:,1]), np.max(m2.coords[:,1]))

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xticks([])
ax2.set_yticks([])

ax.indicate_inset_zoom(ax2, edgecolor='k')

cbar.ax.set_title('elevation [m]')

plt.show()


# add labeled sets for subcatchments and outlets
watershed_workflow.regions.addWatershedAndOutletRegions(m2, watershed, outlet_width=250, exterior_outlet=True)

# add labeled sets for river corridor cells
watershed_workflow.regions.addRiverCorridorRegions(m2, rivers)

# add labeled sets for river corridor cells by order
watershed_workflow.regions.addStreamOrderRegions(m2, rivers)

print('2D labeled sets')
print('---------------')
for ls in m2.labeled_sets:
	print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

#########################################################################################################

#--- #----- VII. Extrude the 2D Mesh to make a 3D mesh -----#

#--- VII-1. Generate a dz structure for the top 2m of soil
#

if ELM_SOILCOLUMN:	
#---	# VII-1A. dz structure from ELM soil column
	zi_soil, dzs_soil, z_soil = elm_domain.soilcolumn(more_vertlayers=MORE_VERTLAYERS, nlevgrnd=15)   
	dzs_soil = dzs_soil[1:] # in ELM, layer indexing from 1. So need to do something here.
	z_soil = z_soil[1:]
	total_thickness = sum(dzs_soil)
	print('ELM soil column total thickness: ', sum(dzs_soil))
		
#---	# VII-1A(1). generate ELM domain.nc, unstructureed, from m2 surface mesh
	import xarray
	
	ncf_elmoutput = 'GCREW31-elm_ICB20TRCNPRDCTCBC.elm.h0.2021-01-01-00000.nc'
	#ncf = toOutput(f'{ncf_elmoutput}')
	ncf = os.path.join(work_dir,'gcrew-atselm-transient',ncf_elmoutput)
	ncfid = xarray.open_dataset(ncf)
    # visualizing ELM data, as needed
	elmvar = 'SOILPSI'
	doy = 364
	layer = 0
	if elmvar in ncfid.variables:
		m2.cell_data[elmvar] = ncfid.variables[elmvar].to_numpy()[doy,layer,:]*(-1.e6)
		elmvar_unit = 'Pa'#ncfid.variables[elmvar].attrs['units'].split('(')[0]

		# simply plotting
		#elmvar_gons = m2.plot(facecolors=m2.cell_data[elmvar], cmap='RdBu', edgecolors=None)
		fig, ax = plt.subplots()

		mp = m2.plot(facecolors=m2.cell_data[elmvar], edgecolors=None, cmap='rainbow', ax=ax, linewidth=0.01, colorbar=False)
		cbax = fig.add_axes([0.85,0.60,0.04,0.35])
		cbar = fig.colorbar(mp, orientation="vertical", cax=cbax)
		cbar.ax.set_title(elmvar+ ' ('+elmvar_unit+')')

		ax.set_title('grided ELM output: '+elmvar+ '@DOY - '+str(doy))
		ax.set_aspect('equal', 'datalim')
		xlim = (np.min(m2.coords[:,0]), np.max(m2.coords[:,0]))
		ylim = (np.min(m2.coords[:,1]), np.max(m2.coords[:,1]))
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		plt.show()



#########################################################################################################

#--- #----- X. summary -----# 
#
logging.info('this workflow is a total success!')





