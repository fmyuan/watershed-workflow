"""
	Complete Workflow for generating ELM tri-domain and surface data
		
	It uses the following datasets:
	
		- 3DEP for elevation
		- NLCD for land cover/transpiration/rooting depths
		- MODIS for LAI
		- GLYHMPS geology data for structural formations
		- Pelletier for depth to bedrock and soil texture information
		- SSURGO for soil data, where available, in the top 2m.
	
	Given some basic inputs (in the next cell) including a {myNAME}, 
	this workflow creates the following files, all of which will reside in output_data:
	
		- Mesh file: {myNAME}.exo, includes all labeled sets
		- Forcing: LAI data -- every 4 days, time series by land cover type of LAI.
			
	ELM Input files:
		- domain.lnd.Nx1pt.{myNAME}.nc
		- surfdata_Nx1pt_simyr1850-{myNAME}.nc
		- landuse_timeseries_Nx1pt_hist_simyr1850-2015-{myNAME}.nc
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
from rvt import vis as rvtvis
from shapely.geometry.geo import box

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
MORE_VERTLAYERS=True
if ELM_SOILCOLUMN:
	import watershed_workflow.elm_domain as elm_domain
	import watershed_workflow.elm_mksrfdata as elm_mksrfdata
	from types import SimpleNamespace
	import watershed_workflow.elm_metdata as elm_metdata
	# e3sm has a xml file, in which by machine-name, DIN_LOC_ROOT is pre-defined.
	# (TODO this locally or from data server)
	elm_domain.set_e3sm_input('/Users/f9y/project_e3sm/e3sm_inputdata')
	DIN_LOC_ROOT='/Users/f9y/project_e3sm/e3sm_inputdata'
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
wkdir_name = 'NGEE-TFS_AnakuvukRiverBurns'
if cwd[-1] == wkdir_name:
    pass
elif cwd[-1] == 'examples':
    cwd.append(wkdir_name)
else:
    cwd.extend(['examples',wkdir_name])
#--- END REMOVE THIS PORTION

#--- A few user defined options
# name of watershed or whatever for your cases
myname = wkdir_name.lower()+'-elm'

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
mycase_shapefile = os.path.join(data_dir, 'Anak_Burn_Perim_Rocha.shp')

# Geometric parameters
# -- parameters to clean and reduce the river network prior to meshing
# 
# Simulation control
start = cftime.DatetimeNoLeap(2003,1,1)  # modis LAI starts from 2002-07-04
end = cftime.DatetimeNoLeap(2022,1,1)           
nyears_cyclic_steadystate = 4            # how many years to run spinup

# Global Soil Properties
min_porosity = 0.05       # minimum porosity considered "too small"
max_permeability = 1.e-10 # max value considered "too permeable"
max_vg_alpha = 1.e-3      # max value of van Genuchten's alpha -- our correlation is not valid for some soils

# a dictionary of output_filenames -- will include all filenames generated
output_filenames = {}

# Note that, by default, we tend to work in projected CRS
if True:
	crs = watershed_workflow.crs.default_crs
	refine_max_edge_length = 1000 # m
	refine_max_area = 750*750  # m2
	refine_tol = 1.e-5
	exterior_buff = 100
else:
	crs = watershed_workflow.crs.latlon_crs
	refine_max_edge_length = 0.01 # latlon deg
	refine_max_area = 0.01*0.01  # degxdeg
	refine_tol = 1.e-5
	exterior_buff = 1.e-4

# get the shape and crs of the watershed in .shp file

mycase_source = watershed_workflow.sources.ManagerShapefile(mycase_shapefile)
watershed_shape = mycase_source.getShapes(out_crs=crs)
watershed_shape.rename(columns={'AREA' : names.AREA}, inplace=True)

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

geo_file = os.path.join(data_dir, 'soil_structure', 'GLHYMPS', 'GLHYMPS.shp')

# GLHYMPs is a several-GB download, so we have sliced it and included the slice here
sources['geologic structure'] = watershed_workflow.sources.ManagerGLHYMPS(geo_file)

# The Pelletier DTB map is not particularly accurate at mycase -- the SoilGrids map seems to be better.
# Here we will use a clipped version of that map.
sources['depth to bedrock'] = watershed_workflow.sources.ManagerRaster(dtb_file)



# END DELETE THIS SECTION

# log the sources that will be used here
watershed_workflow.sources.logSources(sources)


#########################################################################################################

#--- #----- III. Basin Geometry -----#

"""
In this section, we choose the basin, the streams to be included in the stream-aligned mesh, 
and make sure that all are resolved discretely at appropriate length scales for this work.

"""

#--- III-1. the Watershed

# Construct and plot the WW object used for storing watersheds
watershed = watershed_workflow.split_hucs.SplitHUCs(watershed_shape)
watershed.plot()

#########################################################################################################

#--- #----- IV.  Mesh Geometry -----#

"""
Discretely create the stream-aligned mesh. Download elevation data, and condition the mesh discretely to make for better topography.

"""

#--- create the mesh (surface m2)

# watershed bounding box to m2 mesh

bbox = np.asarray(watershed.exterior.buffer(exterior_buff).bounds)
bbox = box(bbox[0],bbox[1],bbox[2],bbox[3])
bbox_gpd = gpd.GeoDataFrame({'id': [1]}, geometry=[bbox], crs=watershed.crs)
bbox_poly= bbox_gpd.segmentize(max_segment_length=refine_max_edge_length)

#clip to watershed polygon
cliper = watershed.exterior.buffer(exterior_buff).simplify(50.0)
bbox_poly = bbox_poly.clip(cliper)

bbox_shapefile = os.path.join(data_dir,'topography','watershed_bbox.shp')
bbox_poly.to_file(bbox_shapefile)

bbox_source = watershed_workflow.sources.ManagerShapefile(bbox_shapefile)
bbox_shapes = bbox_source.getShapes(out_crs=crs)
bbox_shapes.rename(columns={'AREA' : names.AREA}, inplace=True)
watershed_bbox = watershed_workflow.split_hucs.SplitHUCs(bbox_shapes)

 
# directly use watershed_workflow.triagulation() for watershed_bbox only
# so that we can get a more regular tri-mesh than raw watershed
m2, areas, dists = watershed_workflow.triangulate(watershed_bbox,
										#refine_min_angle=30.0,
										#refine_max_edge_length=refine_max_edge_length,
										refine_max_area=refine_max_area,
										enforce_delaunay=False,
										tol = refine_tol, 
										diagnostics=True)

# prepartition to maintain ordering
#m2 = m2.partition(8, True)

# get a raster for the elevation map, based on 3DEP, or locally
#dem = sources['DEM'].getDataset(watershed.exterior.buffer(100), watershed.crs)['dem']
# locally available raster DEM
dem_raster = os.path.join(data_dir,'topography','AnakRiverBurns_dem_25m.tif') 
sources['DEM'] = watershed_workflow.sources.ManagerRaster(dem_raster)
dem = sources['DEM'].getDataset(watershed_bbox.exterior.buffer(exterior_buff), watershed_bbox.crs)['band_1']
m2.cell_data['DEM'] = watershed_workflow.getDatasetOnMesh(m2, dem, method='linear')
if os.path.exists('./dem_raw.tif'):
	os.makedirs(os.path.join(data_dir, 'topography'), exist_ok=True)
	if os.path.exists(os.path.join(data_dir, 'topography')+'/dem_raw.tif'): 
		os.system('rm -f '+os.path.join(data_dir, 'topography')+'/dem_raw.tif')		
	shutil.move('./dem_raw.tif', os.path.join(data_dir, 'topography'),)
	

#--- provide surface mesh elevations
watershed_workflow.elevate(m2, dem)

# Plot the DEM raster
fig, ax = plt.subplots()

# Plot the DEM data
im = dem.plot(ax=ax, cmap='terrain', add_colorbar=False)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Elevation (m)', rotation=270, labelpad=15)

# Add title and labels
ax.set_title('Digital Elevation Model (DEM)', fontsize=14, fontweight='bold')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

# Set equal aspect ratio
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# plotting surface mesh with elevations
fig, ax = plt.subplots()
ax2 = ax.inset_axes([0.85,0.03,0.25,0.40])
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

for ls in m2.labeled_sets:
	print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')



#########################################################################################################

#--- #----- V. Surface properties -----#

"""
Meshes interact with data to provide forcing, parameters, and more in the actual simulation. Specifically, we need vegetation type on the surface to provide information about transpiration and subsurface structure to provide information about water retention curves, etc.
"""

#--- V-1. NLCD for LULC ---

"""
We'll start by downloading and collecting land cover from the NLCD dataset, and generate sets for each land cover type that cover the surface. Likely these will be some combination of grass, deciduous forest, coniferous forest, and mixed.
"""

# download the NLCD raster
sources['land cover'] = watershed_workflow.sources.land_cover_sources['NLCD (AK)']
nlcd = sources['land cover'].getDataset(watershed.exterior.buffer(10), watershed.crs)['cover']

# what land cover types did we get?
logging.info('Found land cover dtypes: {}'.format(nlcd.dtype))
logging.info('Found land cover types: {}'.format(set(list(nlcd.values.ravel()))))

# create a colormap for the data
nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
      watershed_workflow.colors.createNLCDColormap(np.unique(nlcd))
nlcd_cmap

fig, ax = plt.subplots(1,1)
nlcd.plot.imshow(ax=ax, cmap=nlcd_cmap, norm=nlcd_norm, add_colorbar=False)
watershed_workflow.colors.createIndexedColorbar(ncolors=len(nlcd_indices), 
                               cmap=nlcd_cmap, labels=nlcd_labels, ax=ax) 
ax.set_title('Land Cover')
plt.show()

#--- map nlcd onto the mesh
m2_nlcd = watershed_workflow.getDatasetOnMesh(m2, nlcd, method='nearest')
# double-check that nan not in the values
#assert 127 not in m2_nlcd
m2_nlcd[np.where(m2_nlcd==127)] = 31 # as barren
m2.cell_data['land_cover'] = m2_nlcd

# create a new set of labels and indices with only those that actually appear on the mesh
nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
      watershed_workflow.colors.createNLCDColormap(np.unique(m2_nlcd))


mp = m2.plot(facecolors=m2_nlcd, cmap=nlcd_cmap, norm=nlcd_norm, edgecolors=None, colorbar=False)
watershed_workflow.colors.createIndexedColorbar(ncolors=len(nlcd_indices), 
                               cmap=nlcd_cmap, labels=nlcd_labels, ax=plt.gca()) 
plt.show()

# add labeled sets to the mesh for NLCD
nlcd_labels_dict = dict(zip(nlcd_indices, nlcd_labels))
watershed_workflow.regions.addSurfaceRegions(m2, names=nlcd_labels_dict)

nlcd_labels_dict

for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')


#--- V-2. MODIS LAI ---

"""
Leaf area index is needed on each land cover type -- this is used in the Evapotranspiration calculation.
"""
# download LAI and corresponding LULC datasets -- these are actually already downloaded, 
# as the MODIS AppEEARS API is quite slow
#
# Note that MODIS does NOT work with the noleap calendar, so we have to convert to actual dates first
start_leap = cftime.DatetimeGregorian(start.year, start.month, start.day)
end_leap = cftime.DatetimeGregorian(end.year, end.month, end.day)

res = sources['LAI'].getDataset(watershed.exterior, crs, start_leap, end_leap)

modis_data = res
assert modis_data['LAI'].rio.crs is not None
print(modis_data['LULC'].rio.crs, modis_data['LULC'].dtype)


# MODIS data comes with time-dependent LAI AND time-dependent LULC -- just take the mode to find the most common LULC
modis_data['LULC'] = watershed_workflow.data.computeMode(modis_data['LULC'], 'time_LULC')

# now it is safe to have only one time
modis_data = modis_data.rename({'time_LAI':'time'})

# remove leap day (366th day of any leap year) to match our Noleap Calendar
modis_data = watershed_workflow.data.filterLeapDay(modis_data)

# plot the MODIS data -- note the entire domain is covered with one type for mycase (it is small!)
modis_data['LULC'].plot.imshow()

# compute the transient time series
modis_lai = watershed_workflow.land_cover_properties.computeTimeSeries(modis_data['LAI'], modis_data['LULC'], 
                                                        polygon=watershed.exterior.buffer(200), polygon_crs=watershed.crs)

modis_lai

# smooth the data in time
modis_lai_smoothed = watershed_workflow.data.smoothTimeSeries(modis_lai, 'time')

# save the MODIS time series to disk
output_filenames['modis_lai_transient'] = toOutput(f'{myname}_LAI_MODIS_transient.h5')
watershed_workflow.io.writeTimeseriesToHDF5(output_filenames['modis_lai_transient'], modis_lai_smoothed)
watershed_workflow.land_cover_properties.plotLAI(modis_lai_smoothed, indices='MODIS')

# compute a typical year
steadystate_td = datetime.timedelta(days=nyears_cyclic_steadystate*365)
modis_lai_typical = watershed_workflow.data.computeAverageYear(modis_lai_smoothed, start-steadystate_td, output_nyears=10)

output_filenames['modis_lai_cyclic_steadystate'] = toOutput(f'{myname}_LAI_MODIS_CyclicSteadystate.h5')
watershed_workflow.io.writeTimeseriesToHDF5(output_filenames['modis_lai_cyclic_steadystate'], modis_lai_typical)
watershed_workflow.land_cover_properties.plotLAI(modis_lai_typical, indices='MODIS')


#--- V-3. Crosswalk of LAI to NLCD LC ---

crosswalk = watershed_workflow.land_cover_properties.computeCrosswalk(modis_data['LULC'], nlcd, method='fractional area')

# Compute the NLCD-based time series
nlcd_lai_cyclic_steadystate = watershed_workflow.land_cover_properties.applyCrosswalk(crosswalk, modis_lai_typical)
nlcd_lai_transient = watershed_workflow.land_cover_properties.applyCrosswalk(crosswalk, modis_lai_smoothed)

watershed_workflow.land_cover_properties.removeNullLAI(nlcd_lai_cyclic_steadystate)
watershed_workflow.land_cover_properties.removeNullLAI(nlcd_lai_transient)
nlcd_lai_transient

# write the NLCD-based time series to disk
output_filenames['nlcd_lai_cyclic_steadystate'] = toOutput(f'{myname}_LAI_NLCD_CyclicSteadystate.h5')
watershed_workflow.io.writeTimeseriesToHDF5(output_filenames['nlcd_lai_cyclic_steadystate'], nlcd_lai_cyclic_steadystate)

output_filenames['nlcd_lai_transient'] = toOutput(f'{myname}_LAI_NLCD_{start.year}_{end.year}.h5')
watershed_workflow.io.writeTimeseriesToHDF5(output_filenames['nlcd_lai_transient'], nlcd_lai_transient)


#########################################################################################################

#--- #----- VI. Subsurface Soil, Geologic Structure -----#


#--- VI-1. get NRCS shapes, on a reasonable crs
nrcs = sources['soil structure'].getShapesByGeometry(watershed.exterior, watershed.crs, out_crs=crs)
nrcs

# create a clean dataframe with just the data we will need for ATS
def replace_column_nans(df, col_nan, col_replacement):
    """In a df, replace col_nan entries by col_replacement if is nan.  In Place!"""
    row_indexer = df[col_nan].isna()
    df.loc[row_indexer, col_nan] = df.loc[row_indexer, col_replacement]
    return

# where poro or perm is nan, put Rosetta poro
replace_column_nans(nrcs, 'porosity [-]', 'Rosetta porosity [-]')
replace_column_nans(nrcs, 'permeability [m^2]', 'Rosetta permeability [m^2]')

# drop unnecessary columns
for col in ['Rosetta porosity [-]', 'Rosetta permeability [m^2]', 'bulk density [g/cm^3]', 'total sand pct [%]',
            'total silt pct [%]', 'total clay pct [%]']:
	# ELM requires sand and clay (and OM) for calculating thermal-hydraulic properties
	if ELM_SOILCOLUMN and (col in ['total sand pct [%]', 'total clay pct [%]']): continue
	nrcs.pop(col)
    
# drop nans
nan_mask = nrcs.isna().any(axis=1)
dropped_mukeys = nrcs.index[nan_mask]

# Drop those rows
nrcs = nrcs[~nan_mask]

assert nrcs['porosity [-]'][:].min() >= min_porosity
assert nrcs['permeability [m^2]'][:].max() <= max_permeability
nrcs

# check for nans
nrcs.isna().any()

# Compute the soil color of each cell of the mesh
# Note, we use mukey here because it is an int, while ID is a string
soil_color_mukey = watershed_workflow.getShapePropertiesOnMesh(m2, nrcs, 'mukey', 
                                                         resolution=50, nodata=-999)

nrcs.set_index('mukey', drop=False, inplace=True)

unique_soil_colors = list(np.unique(soil_color_mukey))
if -999 in unique_soil_colors:
    unique_soil_colors.remove(-999)

# retain only the unique values of soil_color
nrcs = nrcs.loc[unique_soil_colors]

# renumber the ones we know will appear with an ATS ID using ATS conventions
nrcs['ATS ID'] = range(1000, 1000+len(unique_soil_colors))
nrcs.set_index('ATS ID', drop=True, inplace=True)

# create a new soil color and a soil thickness map using the ATS IDs
soil_color = -np.ones_like(soil_color_mukey)
soil_value = np.nan * np.ones(soil_color.shape, 'd')

for v in ['thickness [m]','total sand pct [%]', 'total clay pct [%]']:
	if v not in nrcs.keys(): continue
	for ats_ID, ID, value in zip(nrcs.index, nrcs.mukey, nrcs[v]):
	    mask = np.where(soil_color_mukey == ID)
	    soil_value[mask] = value
	    if 'soil_color' not in m2.cell_data.keys(): soil_color[mask] = ats_ID
	
	if 'soil_color' not in m2.cell_data.keys(): m2.cell_data['soil_color'] = soil_color
	# Nan unfortunately is not good to ELM soil profile
	idx = np.where(np.isnan(soil_value))
	if len(idx[0])>0: soil_value[idx]=np.nanmean(soil_value)  # need a better way to do nearest interp (TODO)
	m2.cell_data[v] = soil_value


# plot the soil color
# -- get a cmap for soil color
sc_indices, sc_cmap, sc_norm, sc_ticks, sc_labels = \
      watershed_workflow.colors.createIndexedColormap(nrcs.index)

mp = m2.plot(facecolors=m2.cell_data['soil_color'], cmap=sc_cmap, norm=sc_norm, edgecolors=None, colorbar=False)
watershed_workflow.colors.createIndexedColorbar(ncolors=len(nrcs), 
                               cmap=sc_cmap, labels=sc_labels, ax=plt.gca()) 
plt.show()

#
#--- VI-2. Depth to Bedrock from SoilGrids ---

dtb = sources['depth to bedrock'].getDataset(watershed.exterior, watershed.crs)['band_1']

# the SoilGrids dataset is in cm --> convert to meters
if not ELM_SOILCOLUMN: dtb.values = dtb.values/100.

# map to the mesh
m2.cell_data['dtb'] = watershed_workflow.getDatasetOnMesh(m2, dtb, method='linear')


gons = m2.plot(facecolors=m2.cell_data['dtb'], cmap='RdBu', edgecolors=None)
plt.show()


#--- VI-3. GLHYMPs Geology ---

glhymps = sources['geologic structure'].getShapesByGeometry(watershed.exterior.buffer(1000), watershed.crs, out_crs=crs)
glhymps = watershed_workflow.soil_properties.mangleGLHYMPSProperties(glhymps,
                                              min_porosity=min_porosity, 
                                              max_permeability=max_permeability, 
                                              max_vg_alpha=max_vg_alpha)

# intersect with the buffered geometry -- don't keep extras
glhymps = glhymps[glhymps.intersects(watershed.exterior.buffer(10))]
glhymps

# quality check -- make sure glymps shapes cover the watershed
print(glhymps.union_all().contains(watershed.exterior))
glhymps

# clean the data
glhymps.pop('logk_stdev [-]')

assert glhymps['porosity [-]'][:].min() >= min_porosity
assert glhymps['permeability [m^2]'][:].max() <= max_permeability
assert glhymps['van Genuchten alpha [Pa^-1]'][:].max() <= max_vg_alpha

if any(glhymps.isna().any()):
	# column-wised, simply replaced with means in the extent
	# since it's for bed rock  - who really knows whatever in deep
	for v, anytf in glhymps.isna().any().items():
		if anytf:
			vtf = glhymps[v].isna()
			glhymps[v][vtf] = np.nanmean(glhymps[v])
# row-wised still has Nan, then drop it - could be problemetic (TODO checking)
nan_rows = glhymps.isna().any(axis=1)
glhymps = glhymps[~nan_rows]

# note that for larger areas there are often common regions -- two labels with the same properties -- no need to duplicate those with identical values.
def reindex_remove_duplicates(df, index):
    """Removes duplicates, creating a new index and saving the old index as tuples of duplicate values. In place!"""
    if index is not None:
        if index in df:
            df.set_index(index, drop=True, inplace=True)
    
    index_name = df.index.name

    # identify duplicate rows
    duplicates = list(df.groupby(list(df)).apply(lambda x: tuple(x.index)))

    # order is preserved
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df[index_name] = duplicates
    return

reindex_remove_duplicates(glhymps, 'ID')
glhymps

# Compute the geo color of each cell of the mesh
geology_color_glhymps = watershed_workflow.getShapePropertiesOnMesh(m2, glhymps, 'index', 
                                                         resolution=50, nodata=-999)

# retain only the unique values of geology that actually appear in our cell mesh
unique_geology_colors = list(np.unique(geology_color_glhymps))
if -999 in unique_geology_colors:
    unique_geology_colors.remove(-999)

# retain only the unique values of geology_color
glhymps = glhymps.loc[unique_geology_colors]

# renumber the ones we know will appear with an ATS ID using ATS conventions
glhymps['ATS ID'] = range(100, 100+len(unique_geology_colors))
glhymps['TMP_ID'] = glhymps.index
glhymps.reset_index(drop=True, inplace=True)
glhymps.set_index('ATS ID', drop=True, inplace=True)

# create a new geology color using the ATS IDs
# note: if -999 (nodata) is in, 'geology_color' will be -1, which cause mat-id exception somewhere below
geology_color = -np.ones_like(geology_color_glhymps)
for ats_ID, tmp_ID in zip(glhymps.index, glhymps.TMP_ID):
    geology_color[np.where(geology_color_glhymps == tmp_ID)] = ats_ID

glhymps.pop('TMP_ID')

m2.cell_data['geology_color'] = geology_color
                            
geology_color_glhymps.min()

#--- VI-4. Combine to form a complete subsurface dataset -----#


bedrock = watershed_workflow.soil_properties.getDefaultBedrockProperties()

# merge the properties databases
subsurface_props = pd.concat([glhymps, nrcs, bedrock])

# save the properties to disk for use in generating input file
output_filenames['subsurface_properties'] = toOutput(f'{myname}_subsurface_properties.csv')
subsurface_props.to_csv(output_filenames['subsurface_properties'])
subsurface_props


#########################################################################################################

#--- #----- VII. Extrude the 2D Mesh to make a 3D mesh -----#

# set the floor of the domain as max DTB
dtb_max = np.nanmax(m2.cell_data['dtb'].values)
m2.cell_data['dtb'] = m2.cell_data['dtb'].fillna(dtb_max)

print(f'total thickness: {dtb_max} m')


#--- VII-1. Generate a dz structure for the top 2m of soil
#

if ELM_SOILCOLUMN:
#---	# VII-1A. dz structure from ELM soil column
	zi_soil, dzs_soil, z_soil = elm_domain.soilcolumn(more_vertlayers=MORE_VERTLAYERS, nlevgrnd=15)   
	dzs_soil = dzs_soil[1:] # in ELM, layer indexing from 1. So need to do something here.
	z_soil = z_soil[1:]
	total_thickness = sum(dzs_soil)
	print('ELM soil column total thickness: ', sum(dzs_soil))

	# no geolayer needed because ELM soil column thickness of ~42 m
	dzs_geo = np.empty(0)
	
	
#---	# VII-1A(1). generate ELM domain.nc, unstructureed, from m2 surface mesh
	elmdomain = {}
	
	ngrid = m2.num_cells
	print('ELM surf domain grid number: ', ngrid)
	m2_crs = str(m2.crs)
	if m2.crs.is_projected:
		# need to transform proj to lat/lon
		elm_crs = watershed_workflow.crs.latlon_crs
		x = m2.centroids[:, 0]
		y = m2.centroids[:, 1]
		elmdomain['xc'], elmdomain['yc'] = \
			watershed_workflow.warp.xy(x, y, m2.crs, elm_crs) 
		
	else:
		elmdomain['xc'] = m2.centroids[:,0]
		elmdomain['yc'] = m2.centroids[:,1]
	elmdomain['zc'] = m2.centroids[:,2]
	
	nv = len(m2.conn[0])
	elmdomain['xv'] = np.empty((ngrid,nv))
	elmdomain['yv'] = np.empty((ngrid,nv))
	elmdomain['zv'] = np.empty((ngrid,nv))
	#elmdomain['area'] = np.empty(ngrid)  # need to be in arc-radian^2
	elmdomain['area_km2'] = np.empty(ngrid)
	for j in range(ngrid):
		#if river_mask[j]==1.0: #skip river grid, which has 4 vertices
		#	elmdomain['xv'][j,:] = np.nan
		#	elmdomain['yv'][j,:] = np.nan
		#	continue
		
		if m2.crs.is_projected:
			elm_crs = watershed_workflow.crs.latlon_crs
			
			x = m2.coords[m2.conn[j][:]][:,0]
			y = m2.coords[m2.conn[j][:]][:,1]
			lon, lat = watershed_workflow.warp.xy(x, y, m2.crs, elm_crs)
			elmdomain['xv'][j,:] = lon
			elmdomain['yv'][j,:] = lat
		else:
			elmdomain['xv'][j,:] = m2.coords[m2.conn[j][:]][:,0]
			elmdomain['yv'][j,:] = m2.coords[m2.conn[j][:]][:,1]
		vert3 = m2.coords[m2.conn[j][:]][:,0:2]
		area_xy = watershed_workflow.utils.computeTriangleArea(*vert3)
		elmdomain['area_km2'][j] = area_xy*1.0e-6
		
		elmdomain['zv'][j,:] = m2.coords[m2.conn[j][:]][:,2]
	 
	elmdomain['mask'] = np.ones(ngrid) # land mask always 1 now, but cautious near coastal region
	#elmdomain['mask'][np.where(river_mask==1.0)] = 0 # mask out river or water-body temporarily
	elmdomain['frac'] = np.ones(ngrid)
	#elmdomain['frac'][np.where(river_mask==1.0)] = 0
	
	elmdomain['crs']=m2.crs
	
	ncf_domain = 'domain.lnd.'+str(ngrid)+'x1pt_'+myname+'.nc'
	output_filenames['elmdomain'] = toOutput(f'{ncf_domain}')
	try:
		os.remove(output_filenames['elmdomain'])
	except FileNotFoundError:
		pass

	elm_domain.domain_ncwrite(elmdomain, WRITE2D=False, ncfile=output_filenames['elmdomain'], coord_system=False)

#---	# VII-1A(2). extracting and re-distributing ELM surfdata*.nc, according to new domain.nc
	# (CAN do offline)
	ncsrf0 = 'surfdata_0.5x0.5_simyr1850_c240308_TOP.nc' # this will be from E3SM inputdata server or local
	ncflu0 = 'landuse.timeseries_0.5x0.5_hist_simyr1850-2015_c240308.nc'
	
	allsurf = elm_domain.refine_surfdata(outdir=work_dir, \
                    lnd_domain_file='../domain.lnd.r05_RRSwISC6to18E3r5.240328.nc', \
                    fsurdat=ncsrf0, \
                    flanduse_timeseries=ncflu0, \
                    userdomain=output_filenames['elmdomain'])
	
#---	# VII-1A(3). updating ELM surfdata*.nc, unstructureed, from m2 surface cell_data, if any
	ncfin = allsurf[0]
	ncf_surf = 'surfdata_'+str(ngrid)+'x1pt_simyr1850_'+myname+'.nc'
	output_filenames['elmsurfdata'] = toOutput(f'{ncf_surf}')
	
	if ncflu0 != None:
		ncfin1 = allsurf[1]
		ncf_surf1 = 'landuse.timeseries_'+str(ngrid)+'x1pt_hist_simyr1850-2015_'+myname+'.nc'
		output_filenames['elmlutimeseries'] = toOutput(f'{ncf_surf1}')
		
	
	surf_from_atsm2 = {}
	surf_from_atsm2['LATIXY'] = elmdomain['yc']
	surf_from_atsm2['LONGXY'] = elmdomain['xc']
	
	# surfdata to be put into
	surf_vars = ''
	nlevsoi = 10
	
	# from ATS
	surf_vars+='TOPO'
	surf_from_atsm2['TOPO'] = m2.centroids[:,2]  # elevation already alligned with m2

	topo_raster = os.path.join(data_dir,'topography','basin_slope.tif')
	if os.path.exists(topo_raster):
		topo = watershed_workflow.sources.ManagerRaster(topo_raster). \
				getDataset(watershed.exterior.buffer(10), watershed.crs)['band_1']
	else:
		#'rvtvis' tools to cal. slope/aspect raw DEM (not yet alligned with m2)
		slp = rvtvis.slope_aspect(dem, output_units="degree")['slope']
		topo = dem.copy(deep=True)        # make a deep copy of geopanda xr so that have coords as dem
		topo.data = slp	                  # data re-assign 	
	m2.cell_data['SLOPE'] = watershed_workflow.getDatasetOnMesh(m2, topo, method='linear')
	surf_vars+=',SLOPE'
	surf_from_atsm2['SLOPE'] = m2.cell_data['SLOPE'].to_numpy()		

	topo_raster = os.path.join(data_dir,'topography','basin_aspect.tif')
	if os.path.exists(topo_raster):
		topo = watershed_workflow.sources.ManagerRaster(topo_raster). \
				getDataset(watershed.exterior.buffer(10), watershed.crs)['band_1']
	else:
		#'rvtvis' tools to cal. slope/aspect from raw DEM (not yet alligned with m2)
		asp = rvtvis.slope_aspect(dem, output_units="degree")['aspect']
		topo = dem.copy(deep=True)
		topo.data = asp		
	m2.cell_data['ASPECT'] = watershed_workflow.getDatasetOnMesh(m2, topo, method='linear')
	surf_vars+=',ASPECT'
	surf_from_atsm2['ASPECT'] = m2.cell_data['ASPECT'].to_numpy()		
		
	if 'SLOPE' in m2.cell_data.keys() and 'ASPECT' in m2.cell_data.keys():
		surf_vars+=',SINSL_SINAS' #sin(SLOPE)*sin(ASPECT) 
		surf_from_atsm2['SINSL_SINAS'] = np.sin(m2.cell_data['SLOPE'].to_numpy())* \
										np.sin(m2.cell_data['ASPECT'].to_numpy())
		surf_vars+=',SINSL_COSAS' #sin(SLOPE)*cos(ASPECT) 
		surf_from_atsm2['SINSL_COSAS'] = np.sin(m2.cell_data['SLOPE'].to_numpy())* \
										np.cos(m2.cell_data['ASPECT'].to_numpy())


	topo_raster = os.path.join(data_dir,'topography','basin_skyview.tif') 
	if os.path.exists(topo_raster):
		topo = watershed_workflow.sources.ManagerRaster(topo_raster). \
				getDataset(watershed.exterior.buffer(10), watershed.crs)['band_1']
	else:
		#'rvtvis' tools to cal. skyview factor from raw DEM (not yet alligned with m2)
		resx = np.nanmean(np.diff(dem.coords['x']))
		resy = np.nanmean(np.diff(dem.coords['y']))
		svf = rvtvis.sky_view_factor(dem, resolution=min(abs(resx),abs(resy)))['svf']
		topo = dem.copy(deep=True)
		topo.data = svf
	m2.cell_data['SKY_VIEW'] = watershed_workflow.getDatasetOnMesh(m2, topo, method='linear')
	surf_vars+=',SKY_VIEW'
	surf_from_atsm2['SKY_VIEW'] = m2.cell_data['SKY_VIEW'].to_numpy()

	#
	#TERRAIN_CONFIG: may be estimated, according to Lee et al. (2011) (Eq. (4)), as following: 
	# (1+cos(slope))/2-SKYVIEW
	# Wei-Liang Lee, K.N. Liou, and Alex Hall, 2011. Parameterization of solar fluxes over mountain surfaces for application to climate models. JGR,116, D01101
	if 'SLOPE' in m2.cell_data.keys() and \
		'SKY_VIEW' in m2.cell_data.keys():
		m2.cell_data['TERRAIN_CONFIG'] = \
			(1.0+np.cos(m2.cell_data['SLOPE'].to_numpy()))/2.0 \
			- m2.cell_data['SKY_VIEW'].to_numpy()
		surf_vars+=',TERRAIN_CONFIG'
		surf_from_atsm2['TERRAIN_CONFIG'] = m2.cell_data['TERRAIN_CONFIG'].to_numpy()
 
	# looks like ATS not really have layered properties of soil
	# to be consistent, ELM should be in a similar way
	znodes = m2.cell_data['thickness [m]'].to_numpy()
	zdata = m2.cell_data['total sand pct [%]'].to_numpy()
	temp_data = np.zeros((nlevsoi, ngrid))
	for ig in range(ngrid):
		temp_data[:,ig] = \
			elm_mksrfdata.mksrfdata_soilcolumn_interp( \
						srf_soildata=zdata[ig], \
						srf_soilnode=znodes[ig], \
						nlevsoi = nlevsoi, fill_method="extrapolate")
		
	surf_vars+=',PCT_SAND'
	surf_from_atsm2['PCT_SAND'] = temp_data 
	
	zdata = m2.cell_data['total clay pct [%]'].to_numpy()
	temp_data = np.zeros((nlevsoi, ngrid))
	for ig in range(ngrid):
		temp_data[:,ig] = \
			elm_mksrfdata.mksrfdata_soilcolumn_interp( \
						srf_soildata=zdata[ig], \
						srf_soilnode=znodes[ig], \
						nlevsoi = nlevsoi, fill_method="extrapolate")
	surf_vars+=',PCT_CLAY'
	surf_from_atsm2['PCT_CLAY'] = temp_data

	
	# from SoilGrids v2.0.1
	soilgrids_dir = os.path.join(data_dir,'soil_structure','soilgrids')
	os.makedirs(soilgrids_dir, exist_ok=True)
	xlmt = elmdomain['xv'].flatten()
	xrange = [np.nanmin(xlmt), np.nanmax(xlmt)]
	ylmt = elmdomain['yv'].flatten()
	yrange = [np.nanmin(ylmt), np.nanmax(ylmt)]
	
	soilvars=['ocd','bdod','sand','silt','clay'] #unit: hg/m3-->0.1kg/m3, cg/cm3 -->0.01kg/dm3, %, %, %
	#soilvars=['ocd'] #unit: hg/m3 --> 0.1kg/m3 ORGANIC in ELM, but not sure if in kgC or kgSOM ???
	horizons = ['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm']   
	elm_mksrfdata.download_geotiff_soilgrids( \
			Range_XLONG=xrange, Range_YLATI=yrange, \
			outputpath=soilgrids_dir, \
            soilvars=soilvars, \
            value='mean')
	#
	vardata = {}
	for ivar in soilvars:
		temp_data = np.empty((len(horizons), ngrid))
		for iz in range(len(horizons)):
			zstr=horizons[iz]
			svar_horizon_id = ivar+'_'+zstr+'_mean'
			svar_horizon_value_tif = soilgrids_dir+'/'+svar_horizon_id+'.tif'
			svar = watershed_workflow.sources.ManagerRaster(svar_horizon_value_tif)
			svar2 = svar.getDataset(watershed.exterior, watershed.crs)['band_1']
			# map to the mesh
			m2.cell_data[svar_horizon_id] = watershed_workflow.getDatasetOnMesh(m2, svar2, method='linear')
			temp_data[iz,:] = m2.cell_data[svar_horizon_id].to_numpy()

		znodes = np.array([0.0, 0.025, 0.10, 0.225, 0.45, 0.80, 1.50, 2.0])  #mid-horizon + 2-ends (from top-bottom range of data interpolation)
			
		# ELM soil column data has 10 layers down to about 4.2 m
		vardata[ivar] = np.empty((nlevsoi, ngrid))
		for ig in range(ngrid):
			zdata = np.concatenate(([temp_data[0,ig]],temp_data[:,ig],[temp_data[-1,ig]]))
			if ivar in ['bdod','sand','silt','clay']:
				vardata[ivar][:,ig] = \
					elm_mksrfdata.mksrfdata_soilcolumn_interp( \
						srf_soildata=zdata, \
						srf_soilnode=znodes, \
						nlevsoi=nlevsoi, fill_method="extrapolate")
			if ivar in ['ocd']:
				vardata[ivar][:,ig] = \
					elm_mksrfdata.mksrfdata_soilcolumn_interp( \
						srf_soildata=zdata, \
						srf_soilnode=znodes, \
						nlevsoi=nlevsoi)
							
	if 'ocd' in vardata.keys():
		surf_vars+=',ORGANIC'
		surf_from_atsm2['ORGANIC'] = vardata['ocd']*0.1 #but not sure if 'ocd' in kgC or kgSOM ???
	
	# ELM LandUnit and PFT crosswalk from NLCD (already aligned with m2 mesh)
	if 'land_cover'in m2.cell_data.keys():
		nlcd_xr = m2.cell_data['land_cover']
		surf_lupft = elm_mksrfdata.mksrfdata_lupft_fromNLCD(ncfin, nlcd_xr, 
							natvegLUonly=True, defaultPFT=True, grid_aggregated=False)
		for v_lupft in surf_lupft.keys():
			surf_vars+=','+v_lupft
			surf_from_atsm2[v_lupft] = surf_lupft[v_lupft]
	
	#
	# write all in 'surf_vars' to surfdata.nc
	print('usr-provided varables: ', surf_vars)
	elm_mksrfdata.mksrfdata_updatevals(ncfin, \
						fsurfnc_out=output_filenames['elmsurfdata'], \
						user_srf_data=surf_from_atsm2, \
						user_srf_vars=surf_vars)
	
	# TODO - updating landuse_timeseries data 
	

if ELM_SOILCOLUMN:
    # visualizing ELM data, as needed
	elmvar = 'ORGANIC'
	if True:
		m2.cell_data[elmvar] = surf_from_atsm2[elmvar][0,:]

		# simply plotting
		elmvar_gons = m2.plot(facecolors=m2.cell_data[elmvar], cmap='rainbow', edgecolors=None, linewidth=0.01)
		plt.show()

    #
else:
#---	# VII-1B. dz manually constructed
	# this looks like it would work out, with rounder numbers:
	dzs_soil = [0.05, 0.05, 0.05, 0.12, 0.23, 0.5, 0.5, 0.5]
	print(sum(dzs_soil))

	# 50m total thickness, minus 2m soil thickness, leaves us with 48 meters to make up.
	dzs_geo = [1.0, 2.0, 4.0, 8.0, 11, 11, 11]
	print(dzs_geo)
	print(sum(dzs_geo))


# layer bottom(s)
DTB = m2.cell_data['dtb'].values
soil_color = m2.cell_data['soil_color'].values
geo_color = m2.cell_data['geology_color'].values
soil_thickness = m2.cell_data['thickness [m]'].values


# data structures needed for extrusion
layer_types = []
layer_data = []
layer_ncells = []
layer_mat_ids = []

#--- VII-2. soil layer
depth = 0
for dz in dzs_soil:
    depth += 0.5 * dz
    layer_types.append('constant')
    layer_data.append(dz)
    layer_ncells.append(1)

    # use glhymps params
    br_or_geo = np.where(depth < DTB, geo_color, 999)

    soil = np.bitwise_and(soil_color > 0, depth < soil_thickness)
	
    soil_or_br_or_geo = np.where(np.bitwise_or(soil, geo_color < 0),  # if no geo, goes with soil
                                 soil_color,
                                 br_or_geo)

    layer_mat_ids.append(soil_or_br_or_geo)
    depth += 0.5 * dz

#--- VII-3. geologic layer
for dz in dzs_geo:
    depth += 0.5 * dz
    layer_types.append('constant')
    layer_data.append(dz)
    layer_ncells.append(1)

    geo_or_br = np.where(depth < DTB, geo_color, 999)

    layer_mat_ids.append(geo_or_br)
    depth += 0.5 * dz

# print the summary
watershed_workflow.mesh.Mesh3D.summarizeExtrusion(layer_types, layer_data, 
                                            layer_ncells, layer_mat_ids)

# downselect subsurface properties to only those that are used
layer_mat_id_used = list(np.unique(np.array(layer_mat_ids)))
subsurface_props_used = subsurface_props.loc[layer_mat_id_used]
subsurface_props_used


#--- VII-4. extrude to obtain m3 mesh3D
m3 = watershed_workflow.mesh.Mesh3D.extruded_Mesh2D(m2, layer_types, layer_data, 
                                             layer_ncells, layer_mat_ids)

print('2D labeled sets')
print('---------------')
for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('')
print('Extruded 3D labeled sets')
print('------------------------')
for ls in m3.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('')
print('Extruded 3D side sets')
print('---------------------')
for ls in m3.side_sets:
    print(f'{ls.setid} : FACE : {len(ls.cell_list)} : "{ls.name}"')


# save the mesh to disk
output_filenames['mesh'] = toOutput(f'{myname}.exo')
try:
    os.remove(output_filenames['mesh'])
except FileNotFoundError:
    pass
m3.writeExodus(output_filenames['mesh'], 'material id')


#########################################################################################################

#--- #------ VIII. Meteorological forcing dataset -----#
try:

#--- #------ VIII-1. ATS-ready formats

	# download the data -- note it is hourly!
	met_data_raw = sources['meteorology'].getDataset(watershed.exterior.buffer(500), crs, start_leap, end_leap)
		
	#--- #------ VIII-2. hourly dataset generation as ELM-ready format
	if ELM_SOILCOLUMN:
		# standard ELM met. variables
		elmvnames=['LONGXY','LATIXY','time', 'start_year', 'end_year', \
				'ZBOT','TBOT', 'PRECTmms', 'QBOT', 'FSDS', 'FLDS', 'PSRF', 'WIND']
	
		elmmet={}
		met_noleap = watershed_workflow.data.filterLeapDay(met_data_raw)
		
		y = met_noleap['latitude'].to_numpy()
		x = met_noleap['longitude'].to_numpy()
		elmmet['LONGXY'] = np.meshgrid(x,y)[0].flatten() # 2D -> 1D
		elmmet['LATIXY'] = np.meshgrid(x,y)[1].flatten() # 2D -> 1D
		
		#
		t = met_noleap['time'].to_numpy()
		t_yr0 = t[0].year
		t_unit = "days since "+str(t_yr0)+"-01-01 00:00:00" 
		t_dsyr0 = cftime.date2num(t, t_unit, calendar='noleap')
		elmmet['time'] = t_dsyr0
		elmmet['tunit'] = t_unit
		
		elmmet['TBOT'] = np.reshape(met_noleap['TMP_2maboveground'].to_numpy(),(t.size,x.size*y.size))
		elmmet['QBOT'] = np.reshape(met_noleap['SPFH_2maboveground'].to_numpy(),(t.size,x.size*y.size))
		elmmet['PRECTmms'] = np.reshape(met_noleap['APCP_surface'].to_numpy()/3600.0,(t.size,x.size*y.size))  # kg/m2/hour --> mm/s
		elmmet['FSDS'] = np.reshape(met_noleap['DSWRF_surface'].to_numpy(),(t.size,x.size*y.size))
		elmmet['FLDS'] = np.reshape(met_noleap['DLWRF_surface'].to_numpy(),(t.size,x.size*y.size))
		elmmet['PSRF'] = np.reshape(met_noleap['PRES_surface'].to_numpy(),(t.size,x.size*y.size))
		uw = np.reshape(met_noleap['UGRD_10maboveground'].to_numpy(),(t.size,x.size*y.size))
		uv = np.reshape(met_noleap['VGRD_10maboveground'].to_numpy(),(t.size,x.size*y.size)) 
		elmmet['WIND'] = np.sqrt(uw*uw + uv*uv)
		
		# save in ELM forcing data format
		#met_odir = data_dir+'/meteorology/cpl_bypass_full/'
		met_odir = toOutput(f'cpl_bypass_full/')
		output_filenames['meteorology_for_elm_dir'] = met_odir
		os.makedirs(met_odir, exist_ok=True)
		options_wrt = SimpleNamespace( \
	            met_idir = data_dir+'/cpl_bypass_template/', \
	            met_odir = met_odir, \
	            nc_create = True, \
	            nc_write = False, \
	            nc_write_mettype = 'ATS-subdaily_cplbypass' )
		elm_metdata.elm_metdata_write(options_wrt, elmmet)
	
	#

except:
	print('NO met. data generated! ')	


#########################################################################################################

#--- #----- X. summary -----# 
#
# the following files were generated during this run:
print(f'{"role":<35}: filename')
print('-'*34, ': ', '-'*50)
for k,v in output_filenames.items():
    vs = list(splitPathFull(v))
    if vs[-2] == myname:
        v2 = vs[-1]
    else:
        v2 = os.path.join(vs[-2], vs[-1])
    
    print(f'{k:<35}: {v2}')


# double-checking correct centroid info
np.save('m3_bary_centroids.npy', m3.barycentric_centroids)

logging.info('this workflow is a total success!')





