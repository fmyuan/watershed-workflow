# Coweeta ELM+ATS Coupled Workflow
'''
This workflow generates all input files needed for a coupled ELM+ATS simulation at the
Coweeta watershed. ELM drives all meteorological forcing and provides porosity (watsat)
to ATS at runtime via the ELM-ATS coupling layer.

Datasets used:
* `NHD Plus HR` for hydrography
* `3DEP` for elevation
* `NLCD` for land cover (PFT assignment for ELM)
* `NRCS/SSURGO` for soil texture (sand/clay → ELM pedotransfer → watsat; Rosetta → ATS WRM)

Outputs written to `elm_output_data/`:
* `coweeta_np<NTASKS>.exo` — 3D mesh (15 layers, ~42 m deep, Oak Harbor dz
  structure), partitioned into NTASKS pieces; the filename is tagged with
  NTASKS since the mesh partition is fixed at build time and must match the
  MPI rank count used to build and run the ELM+ATS cases
* `coweeta_np<NTASKS>.h5` — lat/lon cell data for ELM column matching
* `coweeta_domain.nc` — ELM domain file
* `coweeta_surfdata.nc` — ELM surface data (PFT, texture, geometry)
* `coweeta_stage1a.xml` — ATS-only steady-state spinup input (stage 1a of
  `examples/coweeta/build_example.sh`); self-contained, no ELM coupling
* `coweeta_stage2.xml`, `coweeta_stage3.xml` — ATS input files for the coupled
  ELM+ATS runs (stages 2 and 3), based on `elm_ats_template.xml`; identical except
  for which previous stage's checkpoint each restarts ATS pressure from
* `coweeta_subsurface_properties.csv` — soil properties lookup table
* `user_nl_elm_stage1b`, `user_nl_elm_stage2`, `user_nl_elm_stage3` — ELM's own
  input files for each ELM-running stage, written alongside the ATS XMLs since
  they are just as much domain-specific input generation

After running this notebook, copy the outputs into a campaign directory's shared
`inputdata/` (e.g. `Coweeta_Campaign0/inputdata/`) as described in the completion cell.
'''

#%load_ext autoreload
#%autoreload 2


## FIX ME -- why is this broken without importing netcdf first?
import netCDF4

import watershed_workflow.io
watershed_workflow.io.setupLogging(1)

import exodus3

import os, sys
import logging
import numpy as np
from matplotlib import pyplot as plt
import shapely
import pandas as pd
import geopandas as gpd
pd.options.display.max_columns = None

# ats_input_spec is not pip-installable; add its source tree to the path
#sys.path.insert(0, os.path.expanduser('~/code/ats/ats_input_spec/repos/master'))

# Set paths to relevant packages for mesh and input file generation
# Update these paths to match your local installation
#
sys.path.append('/Users/f9y/micromamba/amanzi-ats-tools/seacas-exodus/lib')
sys.path.append('/Users/f9y/micromamba/amanzi-ats-tools/amanzi_xml')
os.environ['AMANZI_SRC_DIR']='/Users/f9y/mygithub/ATS_REPOS/amanzi'
os.environ['ATS_SRC_DIR']='/Users/f9y/mygithub/ATS_REPOS/amanzi/src/physics/ats'


import watershed_workflow
import watershed_workflow.utils
import watershed_workflow.utils.warp
import watershed_workflow.sources
import watershed_workflow.plot
import watershed_workflow.mesh
import watershed_workflow.crs
import watershed_workflow.properties.land_cover
import watershed_workflow.hydro
import watershed_workflow.io
import watershed_workflow.io.elm
import watershed_workflow.sources.standard_names as names

import ats_input_spec
import ats_input_spec.public
import ats_input_spec.io

import amanzi_xml.utils.io as aio
import amanzi_xml.utils.search as asearch
import amanzi_xml.utils.errors as aerrors

import h5py

plt.rcParams['figure.figsize'] = (8, 6)

## Parameters

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
def splitPathFull(path):
    parts = []
    while True:
        head, tail = os.path.split(path)
        if head == path:
            parts.insert(0, head); break
        elif tail == path:
            parts.insert(0, tail); break
        else:
            parts.insert(0, tail); path = head
    return parts

cwd = splitPathFull(os.getcwd())
if cwd[-1] == 'Coweeta':
    pass
elif cwd[-1] == 'examples':
    cwd.append('Coweeta')
else:
    cwd.extend(['examples', 'Coweeta'])

data_dir = os.path.join(*(cwd + ['input_data']))
def toInput(f): return os.path.join(data_dir, f)

output_dir = os.path.join(*(cwd + ['elm_output_data']))
os.makedirs(output_dir, exist_ok=True)
def toOutput(f): return os.path.join(output_dir, f)

work_dir = os.path.join(*cwd)
def toWorkingDir(f): return os.path.join(work_dir, f)

# Point at local data cache so this notebook works offline / without large downloads
watershed_workflow.utils.setDataDirectory(data_dir)

# -----------------------------------------------------------------------
# Mesh / watershed parameters
# -----------------------------------------------------------------------
name = 'coweeta'
coweeta_shapefile = toInput('coweeta_basin.shp')

simplify          = 60
ignore_small_rivers = 2
prune_by_area_fraction = 0.01

refine_L0 = 125;  refine_L1 = 300
refine_d0 = 200;  refine_d1 = 600
refine_A0 = refine_L0**2 / 2
refine_A1 = refine_L1**2 / 2

# -----------------------------------------------------------------------
# ELM / ATS template paths
# -----------------------------------------------------------------------
# Oak Harbor 1-column surfdata used as template for SurfdataBuilder
template_surfdata = toInput('elm_ats_surfdata_template.nc')

# ATS XML template (site-independent ELM-ATS skeleton with MESH_FILENAME placeholders)
ats_xml_template = toInput('elm_ats_template.xml')


print(f'template_surfdata: {template_surfdata}')
print(f'ats_xml_template : {ats_xml_template}')
print(f'output_dir       : {output_dir}')

output_filenames = {}

crs = watershed_workflow.crs.default_crs

## Section 1: Basin Geometry

coweeta_source = watershed_workflow.sources.ManagerShapefile(coweeta_shapefile)
coweeta = coweeta_source.getShapes(out_crs=crs)
coweeta.rename(columns={'AREA': names.AREA}, inplace=True)
watershed = watershed_workflow.Watershed(coweeta)
watershed.plot()

sources = watershed_workflow.sources.getDefaultSources()
sources['hydrography'] = watershed_workflow.sources.sources['geometry']['NHDPlusHR-pynhd_nhdplushr']
watershed_workflow.sources.logSources(sources)

reaches = sources['hydrography'].getShapesByGeometry(watershed.exterior, crs, out_crs=crs)
rivers  = watershed_workflow.hydro.createRivers(reaches, method='hydroseq')
watershed_orig, rivers_orig = watershed, rivers

def createCopy(ws, rivs):
    return ws.deepcopy(), [r.deepcopy() for r in rivs]

watershed, rivers = createCopy(watershed_orig, rivers_orig)

watershed_workflow.simplify(watershed, rivers, refine_L0, refine_L1, refine_d0, refine_d1)
for river in rivers:
    river.resetDataFrame()
for river in rivers:
    watershed_workflow.hydro.findOutletsByCrossings(watershed, river)

min_angle = 32

def widths(reach):
    mapping = {1: 8, 2: 12, 3: 16}
    return mapping.get(reach.properties['stream_order'], 8)

m2, areas, dists = watershed_workflow.tessalateRiverAligned(
    watershed, rivers,
    river_width=widths,
    refine_min_angle=min_angle,
    refine_distance=[refine_d0, refine_A0, refine_d1, refine_A1],
    diagnostics=True)

# elevate to a dem
dem = sources['DEM'].getDataset(watershed.exterior.buffer(100), watershed.crs)['dem']
watershed_workflow.elevate(m2, dem)

# now deal with the river...
# adding elevations to the river tree for stream bed conditioning
watershed_workflow.mesh.setProfileByDEM(rivers, dem)

# now condition the river to fix places where the DEM does not intersect the river
def computeBurnInDepthFromData(reach):
    return reach['bankfull_depth'] 

def computeBurnInDepth(da_sq_miles):
    """burn-in depth as a function of drainage area"""
    depth_in_feet = 1.22 * da_sq_miles**0.317
    return 0.3048 * depth_in_feet # ft --> meters

def computeBurnInDepthFromDA(reach):
    depth = computeBurnInDepth(reach['drainage_area_sqkm'] * 0.386102)
    logging.debug(f"reach of DA {reach['drainage_area_sqkm']} has depth {depth}")
    return depth

watershed_workflow.mesh.conditionRiverMeshes(m2,
                       rivers,
                       network_burn_in_depth=computeBurnInDepthFromDA)

# hydrologically condition the mesh, removing pits
outlet_edge = watershed_workflow.mesh.Edge(rivers[0]['elems'][-1][0],rivers[0]['elems'][-1][-1])
print(outlet_edge)
river_cells = [i for (i,elem) in enumerate(m2.conn) if len(elem) > 3]    
m2, res = watershed_workflow.mesh.conditionMesh(m2, preserved_pits=river_cells, forced_outlet_edges=[outlet_edge,])

print(m2.cell_areas[0])

# pre-partition
# NTASKS must match the MPI rank count build_example.sh will use to build and
# run the ELM+ATS cases -- ATS's mesh decomposition (and any checkpoint
# restart across spinup stages) is fixed at this partition count. The mesh
# filename itself is tagged with NTASKS (mesh_name, used below for
# coweeta_np<NTASKS>.exo / coweeta_np<NTASKS>.h5) so a stale mesh built for a
# different task count is obviously
# named wrong rather than silently mismatched -- build_example.sh's own
# NTASKS-vs-partition-count check is a second, independent guard on the same
# invariant.
try:
    ntasks = int(os.environ['NTASKS'])
except KeyError:
    ntasks = 4 # default for Coweeta
mesh_name = f'{name}_np{ntasks}'
m2 = m2.partition(ntasks, True)

print(m2.cell_areas[0])
print(m2.centroids[0])

fig, ax = plt.subplots()
cbax = fig.add_axes([0.05, 0.02, 0.9, 0.04])
mp = m2.plot(facecolors='elevation', edgecolors=None, ax=ax, colorbar=False)
fig.colorbar(mp, orientation='horizontal', cax=cbax).ax.set_title('elevation [m]')
ax.set_title('surface mesh with elevations')
ax.set_aspect('equal', 'datalim')
plt.show()

watershed_workflow.mesh.addWatershedAndOutletRegions(m2, watershed, outlet_width=250, exterior_outlet=True)
watershed_workflow.mesh.addRiverCorridorRegions(m2, rivers)
watershed_workflow.mesh.addStreamOrderRegions(m2, rivers)

for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print(m2.num_cells)

## Section 2: NLCD Land Cover and ELM PFT Assignment

nlcd = sources['land cover'].getDataset(watershed.exterior.buffer(100), watershed.crs)['cover']
logging.info('Found land cover types: {}'.format(set(list(nlcd.values.ravel()))))

nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
    watershed_workflow.plot.createNLCDColormap(np.unique(nlcd))

fig, ax = plt.subplots()
nlcd.plot.imshow(ax=ax, cmap=nlcd_cmap, norm=nlcd_norm, add_colorbar=False)
watershed_workflow.plot.createIndexedColorbar(
    ncolors=len(nlcd_indices), cmap=nlcd_cmap, labels=nlcd_labels, ax=ax)
ax.set_title('NLCD Land Cover')
plt.show()

m2_nlcd = watershed_workflow.getDatasetOnMesh(m2, nlcd, method='nearest')
m2.cell_data['land_cover'] = m2_nlcd

assert 127 not in m2_nlcd, 'nodata NLCD value found on mesh'

nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
    watershed_workflow.plot.createNLCDColormap(np.unique(m2_nlcd))

# Map NLCD codes to ELM natural PFT indices (0-16)
pft_index = watershed_workflow.properties.land_cover.mapNLCDToPFT(
    m2_nlcd, climate_zone='temperate')
m2.cell_data['pft index'] = pft_index

print('PFT indices on mesh:', np.unique(pft_index))

fig, ax = plt.subplots(1,1, figsize=(4,2))
pft_index_list, pft_cmap, pft_norm, pft_ticks, pft_labels = \
    watershed_workflow.plot.createPFTColormap(np.unique(pft_index))

mp = m2.plot('pft index', cmap=pft_cmap, norm=pft_norm,
             edgecolors=None, colorbar=False, ax=ax)
watershed_workflow.plot.createIndexedColorbar(
    ncolors=len(pft_index_list), cmap=pft_cmap, labels=pft_labels, ax=plt.gca())
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()

fig.savefig('./Coweeta_PFTs.png', dpi=300)

plt.show()

nlcd_labels_dict = dict(zip(nlcd_indices, nlcd_labels))
watershed_workflow.mesh.addSurfaceRegions(m2, names=nlcd_labels_dict)

for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

## Section 3: NRCS Soil Texture and ATS WRM Parameters
'''
NRCS SSURGO provides sand/clay texture used for:
- ELM: pedotransfer function `watsat = 0.489 - 0.00126*sand` (all 15 nlevgrnd layers)
- ATS: Rosetta van Genuchten parameters (alpha, n, residual saturation) and permeability

Each column is vertically homogeneous — the same NRCS MUKEY texture applies to all 15
layers.  ELM internally duplicates layer-10 texture for layers 11-15 (SoilStateType.F90
lines 708-713), consistent with this approach.
'''
nrcs = sources['soil structure'].getShapesByGeometry(
    watershed.exterior, watershed.crs, out_crs=crs)
nrcs

def replaceColumnNans(df, col_nan, col_replacement):
    mask = df[col_nan].isna()
    df.loc[mask, col_nan] = df.loc[mask, col_replacement]

# Fill any missing Rosetta-derived values with Rosetta fallback columns
replaceColumnNans(nrcs, 'porosity [-]', 'Rosetta porosity [-]')
replaceColumnNans(nrcs, 'permeability [m^2]', 'Rosetta permeability [m^2]')

# Drop Rosetta backup columns (we keep the originals, now gap-filled)
for col in ['Rosetta porosity [-]', 'Rosetta permeability [m^2]',
            'bulk density [g/cm^3]']:
    if col in nrcs.columns:
        nrcs.pop(col)

# Drop any rows still containing NaN
nan_mask = nrcs.isna().any(axis=1)
dropped = nrcs.index[nan_mask]
if len(dropped):
    logging.warning(f'Dropping {len(dropped)} NRCS rows with NaN after gap-fill: {list(dropped)}')
nrcs = nrcs[~nan_mask]

nrcs

print(watershed_workflow.properties.soil.__file__)

# Map NRCS MUKEY onto the 2D mesh cells
soil_color_mukey = watershed_workflow.getShapePropertiesOnMesh(
    m2, nrcs, 'mukey', resolution=50, nodata=-999)

# Fail if any mesh cell has no NRCS coverage
assert -999 not in soil_color_mukey, \
    'Some mesh cells have no NRCS coverage (nodata=-999). Check watershed extent vs SSURGO.'

nrcs.set_index('mukey', drop=False, inplace=True)

unique_mukeys = list(np.unique(soil_color_mukey))
nrcs = nrcs.loc[unique_mukeys]  # keep only MUKEYs that appear on the mesh

# Assign sequential ATS IDs starting at 1000
nrcs['ATS ID'] = range(1000, 1000 + len(unique_mukeys))
nrcs.set_index('ATS ID', drop=True, inplace=True)

# Build the ATS-ID indexed soil color array
soil_color = -np.ones_like(soil_color_mukey)
for ats_id, mukey in zip(nrcs.index, nrcs['mukey']):
    soil_color[soil_color_mukey == mukey] = ats_id

m2.cell_data['soil_color'] = soil_color
print(f'{len(nrcs)} unique NRCS soil types on mesh, ATS IDs {nrcs.index.min()}–{nrcs.index.max()}')

sc_indices, sc_cmap, sc_norm, sc_ticks, sc_labels = \
    watershed_workflow.plot.createIndexedColormap(nrcs.index)

mp = m2.plot(facecolors=m2.cell_data['soil_color'], cmap=sc_cmap, norm=sc_norm,
             edgecolors=None, colorbar=False)
watershed_workflow.plot.createIndexedColorbar(
    ncolors=len(nrcs), cmap=sc_cmap, labels=sc_labels, ax=plt.gca())
plt.title('NRCS Soil Type (ATS ID)')
plt.show()

# Save subsurface properties CSV
output_filenames['subsurface_properties'] = toOutput(f'{name}_subsurface_properties.csv')
nrcs.to_csv(output_filenames['subsurface_properties'])
print('Saved:', output_filenames['subsurface_properties'])
nrcs

## Section 4: Extrude 2D Mesh to 3D
'''
Uses ELM's default exponential 15-layer vertical grid (`elm_default_dzsoi`, ~42 m total depth).
All 15 layers get the same NRCS soil type as the surface cell (vertically homogeneous).
Mat IDs 1000–(1000+N) for ELM nlevsoi layers; same IDs continue for layers 11-15 since
ELM duplicates layer-10 texture internally.
'''
# ELM vertical grid -- uses ELM's default exponential formula (initVerticalMod.F90)
dzs = watershed_workflow.io.elm.elm_default_dzsoi(nlevgrnd=15)

print(f'Number of layers: {len(dzs)}')
print(f'Layer thicknesses: {np.round(dzs, 4)}')
print(f'Total depth: {dzs.sum():.2f} m')




ncells = m2.num_cells
soil_color_arr = m2.cell_data['soil_color'].values  # shape (ncells,), ATS IDs

# All 15 layers get the surface cell's NRCS ATS ID (vertically homogeneous)
layer_mat_ids = [soil_color_arr.copy() for _ in range(15)]

watershed_workflow.Mesh3D.summarizeExtrusion(
    ['constant'] * 15, dzs.tolist(), [1] * 15, layer_mat_ids)

m3 = watershed_workflow.Mesh3D.extruded_Mesh2D(
    m2, 'constant', dzs.tolist(), [1] * 15, layer_mat_ids)

print('2D labeled sets')
for ls in m2.labeled_sets:
    print(f'  {ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('\n3D labeled sets')
for ls in m3.labeled_sets:
    print(f'  {ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('\n3D side sets')
for ss in m3.side_sets:
    print(f'  {ss.setid} : FACE : {len(ss.cell_list)} : "{ss.name}"')

# Partition the 2D mesh and write lat/lon cell data for ELM column matching
centroids = m2.centroids[:, 0:2]
lon, lat = watershed_workflow.utils.warp.warpXY(
    centroids[:, 0], centroids[:, 1], m2.crs, watershed_workflow.crs.latlon_crs)

m2.cell_data['longitude'] = lon
m2.cell_data['latitude']  = lat

output_filenames['mesh_h5'] = toOutput(f'{mesh_name}.h5')

# should put this capability somewhere in the library, but its specific to both ATS and HDF?
with h5py.File(output_filenames['mesh_h5'], 'w') as fid:
    grp1 = fid.create_group('longitude.cell.0')
    grp1.create_dataset('0', data=lon)

    grp2 = fid.create_group('latitude.cell.0')
    grp2.create_dataset('0', data=lat)

print('Wrote:', output_filenames['mesh_h5'])

# Write 3D mesh exodus file
output_filenames['mesh'] = toOutput(f'{mesh_name}.exo')
try:
    os.remove(output_filenames['mesh'])
except FileNotFoundError:
    pass
m3.writeExodus(output_filenames['mesh'], 'one block')
print('Wrote:', output_filenames['mesh'])

## Section 5: ELM Domain and Surface Data Files
'''
- `writeDomain()` writes the ELM domain NetCDF (lat/lon/area/mask/frac per cell)
- `SurfdataBuilder` clones the Oak Harbor template surfdata, replaces geometry and land
  cover, then fills soil texture from NRCS sand/clay for the 10 ELM `nlevsoi` layers.
  ELM internally duplicates layer-10 texture for layers 11-15 (SoilStateType.F90 708-713).

These files are shared across every ELM run in the spinup workflow (stages 1b, 2, and 3
in `examples/coweeta/build_example.sh`) -- they are built once here, from the same mesh
and NRCS data used throughout this notebook, so they always match the ATS mesh exactly.
Stage 1b in particular reuses `coweeta_domain.nc` / `coweeta_surfdata.nc`
as-is; nothing further needs to be generated for it.
'''
output_filenames['elm_domain'] = toOutput(f'{name}_domain.nc')
watershed_workflow.io.elm.writeDomain(m2, output_filenames['elm_domain'])
print('Wrote ELM domain:', output_filenames['elm_domain'])

# Build the SurfdataBuilder (requires 'pft index' in m2.cell_data)
builder = watershed_workflow.io.elm.SurfdataBuilder(m2, template_surfdata)
nlevsoi = builder._nlevsoi  # should be 10
print(f'Template nlevsoi: {nlevsoi}')

# Build soil_type array: shape (nlevsoi=10, ncells)
# All 10 layers get the surface cell's NRCS ATS ID
soil_type = np.tile(soil_color_arr, (nlevsoi, 1))  # (10, ncells)

# soil_properties DataFrame indexed by ATS ID
# setSoilProperties needs 'total sand pct [%]' and 'total clay pct [%]'
soil_props_for_elm = nrcs[['total sand pct [%]', 'total clay pct [%]']].copy()
if 'total gravel pct [%]' in nrcs.columns:
    soil_props_for_elm['total gravel pct [%]'] = nrcs['total gravel pct [%]']

builder.setSoilProperties(soil_type, soil_props_for_elm)
print('Set soil properties for', nlevsoi, 'layers x', ncells, 'cells')

output_filenames['elm_surfdata'] = toOutput(f'{name}_surfdata.nc')
builder.write(output_filenames['elm_surfdata'])
print('Wrote ELM surfdata:', output_filenames['elm_surfdata'])

## Stage 1a: ATS-only Steady-State Spinup
'''
This writes the ATS input file for **stage 1a** of the multi-stage spinup workflow
(see `examples/coweeta/build_example.sh`): a pure ATS run, uncoupled from ELM, that
drives the mesh built above with a spatially uniform, constant-in-time precipitation
rate until subsurface + surface flow reach steady state.

Unlike Stage 2's coupled XML (spliced from `elm_ats_template.xml`), this one is built
from `steadystate-template.xml` -- the same ATS-only template
`coweeta_ats.ipynb`'s `write_spinup_steadystate()` uses -- via `ats_input_spec`'s
`get_main()`/`populate_basic_properties()` pattern, rather than by pruning ELM-only
pieces out of the coupled template. An earlier version of this notebook built stage
1a's XML by loading `elm_ats_template.xml` and popping ELM-only evaluators
(`surface-evaporation`, transpiration, etc.) out of `state/evaluators`; that missed
the fact that `elm_ats_template.xml`'s `observations` block *also* references those
same evaluators (e.g. an `evaporation [m d^-1]` observable pointing at
`surface-evaporation`), so a pruned coupled template still asked State to create
evaluators that no longer existed
(`Evaluator "surface-evaporation" @ "" cannot be created in State`). Building from
`steadystate-template.xml` avoids this whole class of bug, since that template never
had ELM-only evaluators or observations to begin with.

Because this run is a precursor to the ELM-coupled stage 2 (which restarts ATS
pressure from this run's `checkpoint_final.h5`), the mesh partitioning here must match
ELM's domain decomposition -- so `partitioner = 'from exodus file'` is set explicitly
below. This is *not* set in `steadystate-template.xml` itself, since ATS-only uses of
that template (e.g. `coweeta_ats.ipynb`) don't need to match any external
decomposition and should keep using ATS's own default partitioner (Zoltan RCB).

- porosity comes from NRCS directly (`add_soil_type`'s `porosity` argument), since
  there is no ELM to provide `base_porosity` at runtime here
- forcing is a uniform, constant-in-time `surface-precipitation`. This notebook does
  not download meteorology itself (that's `coweeta_aorc_elm.ipynb`'s job, so AORC is
  only fetched once); instead the precip value is left as a placeholder token,
  `STEADYSTATE_PRECIP_MPS`, that `build_example.sh` substitutes at build time from
  `elm_output_data/coweeta_mean_precip_rain_mps.txt` (written by
  `coweeta_aorc_elm.ipynb`)
- no land cover / canopy physics (steadystate-template.xml has none -- this is a pure
  subsurface + overland flow problem, no ET)

Output: `coweeta_stage1a.xml`, to be copied into `examples/coweeta/1a_ats_spinup/`.
'''

# Mesh paths, written as a DIN_LOC_CAMPAIGN/... placeholder rather than a
# notebook-relative path -- build_example.sh sed's DIN_LOC_CAMPAIGN to the
# campaign's actual shared inputdata/ directory when it copies this XML into
# each stage's case dir, so the placeholder must appear literally (as its own
# path component) rather than as a relative climb that depends on case-dir
# nesting depth, which differs per stage.
rel_exo_1a = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.exo')
rel_h5_1a  = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.h5')

steadystate_template = toInput('steadystate-template.xml')

print('Mesh exo (DIN_LOC_CAMPAIGN placeholder):', rel_exo_1a)
print('Mesh h5  (DIN_LOC_CAMPAIGN placeholder):', rel_h5_1a)

# Build the ATS "main" input spec from scratch (not by pruning the coupled
# template), following the same get_main()/add_domains()/add_soil_type()
# pattern coweeta_ats.ipynb's write_spinup_steadystate() uses. steadystate=True
# switches add_observations_water_balance to a cycle-based (not time-based)
# observation schedule, appropriate for a run without a physical calendar.
main_props_1a = ats_input_spec.public.get_main()

# 3D subsurface domain + 2D surface domain (from the labeled sets built into
# the mesh); no snow/canopy domains, since steadystate-template.xml has no
# ET/canopy physics.
ats_input_spec.public.add_domain(
    main_props_1a, domain_name='domain', dimension=3,
    mesh_type='read mesh file', mesh_args={'file': rel_exo_1a})
main_props_1a['mesh']['domain']['build columns from set'] = 'surface'
ats_input_spec.public.add_domain(
    main_props_1a, domain_name='surface', dimension=2,
    mesh_type='surface', mesh_args={'surface sideset name': 'surface'})

# Labeled sets from the extruded 3D mesh (river corridor, stream order,
# outlet, etc. -- everything add_domain's own 'computational domain' /
# 'surface domain' / boundary regions don't already cover).
for ls in m3.labeled_sets:
    ats_input_spec.public.add_region_labeled_set(
        main_props_1a, ls.name, ls.setid, rel_exo_1a, ls.entity)
for ss in m3.side_sets:
    ats_input_spec.public.add_region_labeled_set(
        main_props_1a, ss.name, ss.setid, rel_exo_1a, 'FACE')

# Soil properties per NRCS/Rosetta soil type, porosity included (no ELM to
# provide base_porosity at runtime).
for ats_id in nrcs.index:
    props = nrcs.loc[ats_id]
    region_name = f'soil type {ats_id}'
    smoothing = 0.01 if props['van Genuchten n [-]'] < 1.5 else 0.0

    ats_input_spec.public.add_soil_type(
        main_props_1a, region_name, label=ats_id, filename=rel_exo_1a,
        porosity=float(props['porosity [-]']),
        permeability=float(props['permeability [m^2]']),
        van_genuchten_alpha=float(props['van Genuchten alpha [Pa^-1]']),
        van_genuchten_n=float(props['van Genuchten n [-]']),
        residual_sat=float(props['residual saturation [-]']),
        smoothing_interval=float(smoothing),
    )

# Uniform, constant-in-time surface-precipitation over the whole surface
# domain. A dummy numeric value is used here -- this notebook does not
# download meteorology itself (that's coweeta_aorc_elm.ipynb's job, so AORC
# is only fetched once). The dummy is overwritten with a placeholder token
# directly in the XML tree below (see the "write" cell), which
# build_example.sh later substitutes with 0.6x the domain/time-mean AORC rain
# rate written by coweeta_aorc_elm.ipynb to
# Coweeta_Campaign0/inputdata/coweeta_mean_precip_rain_mps.txt (the 0.6x
# factor roughly accounts for ET, the same rule of thumb
# coweeta_ats.ipynb's write_spinup_steadystate() uses).
precip_ev_1a = main_props_1a['state']['evaluators'].append_empty('surface-precipitation')
precip_ev_1a.set_type('independent variable constant',
                      ats_input_spec.public.known_specs['evaluator-independent-variable-constant-spec'])
precip_ev_1a['value'] = 0.0   # dummy; overwritten with a placeholder token before writing

# Water balance observations against Coweeta's actual region names (built by
# add_domain above / addWatershedAndOutletRegions earlier in this notebook) --
# add_observations_water_balance's own defaults assume different naming.
ats_input_spec.public.add_observations_water_balance(
    main_props_1a, 'computational domain',
    surface_region='surface domain',
    boundary_region='computational domain boundary',
    outlet_region='surface domain outlet',
    has_canopy=False, steadystate=True,
)

main_xml_1a = ats_input_spec.io.to_xml(main_props_1a)

# Mesh partitioning must match ELM's domain decomposition, since stage 2
# restarts ATS pressure from this run's checkpoint. Set explicitly here
# rather than in steadystate-template.xml itself, since other, non-ELM uses
# of that template (e.g. coweeta_ats.ipynb) don't need this and should keep
# ATS's own default partitioner.
mesh_list_1a = asearch.find_path(main_xml_1a, ['mesh'], no_skip=True)
mesh_list_1a.setParameter('partitioner', 'string', 'from exodus file')

print(f'Built domains + region + WRM + permeability + porosity entries for '
      f'{len(nrcs)} soil types, plus a uniform surface-precipitation evaluator '
      f'and water-balance observations.')

# Load steadystate-template.xml and splice in the mesh/regions/soil/precip/
# observations built above -- the same populate_basic_properties() pattern
# coweeta_ats.ipynb uses, reimplemented here since this notebook doesn't
# import that notebook's helpers directly.
xml_1a = aio.fromFile(steadystate_template)

# mesh + regions: full replace (built from scratch above, including the
# partitioner fix)
xml_1a.replace('mesh', asearch.child_by_name(main_xml_1a, 'mesh'))
xml_1a.replace('regions', asearch.child_by_name(main_xml_1a, 'regions'))

# observations: full replace
obs_idx = next(i for (i, el) in enumerate(xml_1a) if el.get('name') == 'observations')
xml_1a[obs_idx] = asearch.child_by_name(main_xml_1a, 'observations')

# model parameters (WRM parameters) and evaluators (permeability,
# base_porosity, surface-precipitation): merge, replacing by name where the
# template already has a placeholder and appending otherwise
xml_1a_mp = asearch.find_path(xml_1a, ['state', 'model parameters'], no_skip=True)
for parlist in asearch.find_path(main_xml_1a, ['state', 'model parameters'], no_skip=True):
    try:
        xml_1a_mp.replace(parlist.getName(), parlist)
    except aerrors.MissingXMLError:
        xml_1a_mp.append(parlist)

xml_1a_ev = asearch.find_path(xml_1a, ['state', 'evaluators'], no_skip=True)
for elist in asearch.find_path(main_xml_1a, ['state', 'evaluators'], no_skip=True):
    try:
        xml_1a_ev.replace(elist.getName(), elist)
    except aerrors.MissingXMLError:
        xml_1a_ev.append(elist)

print('Spliced mesh, regions, observations, WRM parameters, permeability, '
      'base_porosity, and surface-precipitation into steadystate-template.xml.')

# Overwrite the dummy surface-precipitation value with a placeholder token
# (bypassing type-checked setValue, since 'double' Parameters normally must
# hold a numeric value) that build_example.sh substitutes at build time.
precip_placeholder = 'STEADYSTATE_PRECIP_MPS'

precip_value_el = asearch.find_path(
    xml_1a, ['state', 'evaluators', 'surface-precipitation', 'value'], no_skip=True)
precip_value_el.set('value', precip_placeholder)

# Write the stage 1a ATS XML
output_filenames['ats_xml_stage1a'] = toOutput(f'{name}_stage1a.xml')
aio.toFile(xml_1a, output_filenames['ats_xml_stage1a'])
print('Wrote stage 1a ATS XML:', output_filenames['ats_xml_stage1a'])

## Stage 1b: ELM Carbon Spinup (ELM-only, no ATS)
'''
Stage 1b of the multi-stage spinup workflow (see `examples/coweeta/build_example.sh`)
runs ELM by itself -- native ELM hydrology, no ATS coupling -- for ~200 cyclic years
with accelerated decomposition, to spin up carbon/nitrogen pools before the coupled
ELM+ATS run. It reuses inputs already written by this notebook:

- **Mesh / domain decomposition**: ELM runs on the same domain decomposition as the
  ATS mesh (`domain_decomp_type = 'ats'` is *not* set for this stage -- 1b uses ELM's
  native hydrology -- but the case still runs on the per-cell layout defined by
  `coweeta_domain.nc`, so results are directly comparable/mergeable with the coupled
  runs). No new mesh work is needed here.
- **`coweeta_domain.nc`, `coweeta_surfdata.nc`** (Section 5 above): already built from the
  same mesh (`m2`/`m3`) and NRCS soil data used everywhere else in this notebook, so
  they already match the ATS mesh exactly. Nothing new needs to be generated for
  stage 1b -- `build_example.sh` reads them directly from this notebook's
  `elm_output_data/` output directory, no copy step needed.
- **Meteorology**: cyclic, "typical year" AORC forcing (`atm_forcing_spinup/`), built
  separately in `coweeta_aorc_elm.ipynb` (not this notebook, to avoid downloading
  AORC twice -- see the Stage 1a section's docstring for the same rationale).

### Writing `user_nl_elm`

`user_nl_elm` is ELM's own input file, exactly like the ATS XML written above and
below -- so it belongs here, in the notebook, rather than as a heredoc in the general
CIME-orchestration script `build_example.sh`. The cells below define a small helper,
`elmRestartPath()`, and three functions, `writeUserNlElmStage1b/2/3()`, modeled on
`coweeta_ats.ipynb`'s `write_transient()` / `write_spinup_steadystate()` pattern:
plain functions parameterized by case directory name (and, for stages 2/3, the
*previous* stage's case directory name and run length), computing every path as a
relative offset from the case directory -- confirmed safe, since the C++ ATS reader
does no `chdir` before resolving `ats_inputdir`/`ats_inputfile`.

Case directory names (`1a_ats_spinup`, `1b_elm_carbon_spinup`, `2_cyclic_steadystate`,
`3_transient`) and each ELM-running stage's `RUN_STARTDATE`/`STOP_N` (1b: start year 1,
200 years; 2: start year 1, 10 years) are therefore duplicated here and in
`build_example.sh`'s own `STAGE1B_START_YEAR`/`STAGE1B_STOP_N`/etc. -- both sides need
to agree on the same convention, but there is no runtime dependency between them:
`build_example.sh` never needs to have actually run a prior stage to build the next
one, since `finidat`'s restart-file name is fully determined by the *simulated*
model calendar (`RUN_STARTDATE year + STOP_N`), not by wall-clock/submission time.

`writeUserNlElmStage1b()` writes `user_nl_elm_stage1b` (cyclic forcing,
`nyears_ad_carbon_only = 200`, `spinup_mortality_factor = 10`, no
`use_ats`/`ats_inputfile`, no `finidat` -- cold start).
'''

def elmRestartPath(case_dir_name, start_year, stop_n):
    """Relative path (from a sibling case directory) to an ELM restart file.

    ELM restart files are named <case>.elm.r.<YYYY>-01-01-00000.nc, where YYYY
    is the *simulated* model calendar date at write time (RUN_STARTDATE year +
    STOP_N) -- not a wallclock or submission timestamp. This mirrors
    build_example.sh's own elmRestartFile() bash function; the two must agree
    on case directory names and RUN_STARTDATE/STOP_N per stage.
    """
    restart_year = start_year + stop_n
    return os.path.join('..', '..', case_dir_name, 'run',
                         f'{case_dir_name}.elm.r.{restart_year:04d}-01-01-00000.nc')


def writeUserNlElmStage1b(surfdata_filename):
    """Write user_nl_elm for stage 1b: ELM-only carbon spinup, cyclic forcing,
    accelerated decomposition, cold start (no finidat)."""
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! Cyclic spinup forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_spinup'
 const_climate_hist = .true.

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'
 CO2_file = '$DIN_LOC_ROOT/atm/datm7/CO2/fco2_datm_1765-2007_c100614.nc'

 ! Accelerated decomposition carbon spinup
 nyears_ad_carbon_only = 200
 spinup_mortality_factor = 10

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage1b'] = toOutput('user_nl_elm_stage1b')
    with open(output_filenames['user_nl_elm_stage1b'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage1b'])
    return content


writeUserNlElmStage1b(os.path.basename(output_filenames['elm_surfdata']))


## Stage 2: ELM+ATS Cyclic Steady-State Spinup
'''
Stage 2 of the multi-stage spinup workflow (see `examples/coweeta/build_example.sh`)
runs the fully coupled ELM+ATS system to cyclic steady state (~10 years of repeated
"typical year" forcing), initialized from the stage 1a ATS checkpoint and the stage 1b
ELM restart.

Builds the coupled ATS input by loading `elm_ats_template.xml` and injecting:
1. Mesh filename (coweeta_np<NTASKS>.exo / coweeta_np<NTASKS>.h5)
2. Labeled set regions from m3
3. Per-soil-type WRM parameters (van Genuchten alpha/n/Sr, permeability) from NRCS/Rosetta

Porosity is **not** set here — ELM computes watsat from texture and passes it to ATS at
runtime via ExternalModelATS.F90.  The `compressible porosity` evaluator in the template
XML uses ELM's `base_porosity` directly.

This mesh/region/WRM construction is shared by **stages 2 and 3** (both couple ELM to
ATS with the same PK tree and soil properties), so the cells below build one XML object
that Stage 3 (further down) reuses, making an independent second copy with a different
restart source. They differ only in **initial conditions**: each stage's `flow` PK must
restart ATS's pressure field from the *previous* stage's final checkpoint (following the
same pattern `coweeta_ats.ipynb`'s `write_transient()` uses) --

- **Stage 2** restarts ATS pressure from stage 1a's `checkpoint_final.h5` (the
  ATS-only steady-state spinup above). ELM's carbon/biomass state for stage 2 instead
  comes from stage 1b's ELM restart, via `finidat`, written into `user_nl_elm_stage2`
  below by `writeUserNlElmStage2()`.
- **Stage 3** (below) restarts ATS pressure from stage 2's `checkpoint_final.h5`, and
  ELM's `finidat` similarly comes from stage 2's ELM restart.

Contrast with the stage 1a XML above, which is a self-contained, ELM-free steady-state
spinup with its own cold-start (`hydrostatic head`) initial condition -- stage 1a has
no previous stage to restart from.

**ATS input**: writes `coweeta_stage2.xml`, with the `flow` PK's `initial conditions`
restarting ATS pressure from stage 1a's `checkpoint_final.h5`
(`../../1a_ats_spinup/run/checkpoint_final.h5`, relative to the stage 2 run directory).
`build_example.sh` copies this file into the stage 2 case directory and points
`ats_inputfile` at it.

**ELM namelist**: written below by `writeUserNlElmStage2()` -- cyclic, "typical year"
AORC forcing (`DIN_LOC_CAMPAIGN/atm_forcing_spinup`, the same directory used by stage 1b),
`use_ats = .true.`, `domain_decomp_type = 'ats'`, `ats_inputfile = 'coweeta_stage2.xml'`,
`nyears_ad_carbon_only = 0` (unlike stage 1b -- carbon pools are already spun up), and
`finidat` pointing at stage 1b's restart file (`../../1b_elm_carbon_spinup/run/....elm.r....nc`,
computed the same deterministic way `build_example.sh` computes CIME's `RUN_REFDATE`: the
restart file's date is always `RUN_STARTDATE year + STOP_N` of the *previous* stage, so
this does not require stage 1b to have actually finished running yet -- see the
"Stage 3" heading's Approach note below, and E3SM's own OLMT spinup driver, which uses
the same convention).
'''

# Mesh paths, written as a DIN_LOC_CAMPAIGN/... placeholder (see the stage 1a
# mesh-path cell above for why) -- build_example.sh sed's this to the
# campaign's actual shared inputdata/ directory for both stages 2 and 3,
# which reuse this same mesh/region/WRM xml object.
rel_exo = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.exo')
rel_h5  = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.h5')

print('Mesh exo (DIN_LOC_CAMPAIGN placeholder):', rel_exo)
print('Mesh h5  (DIN_LOC_CAMPAIGN placeholder):', rel_h5)

# Load the template XML and replace the MESH_FILENAME placeholders via string substitution
with open(ats_xml_template, 'r') as f:
    xml_str = f.read()

xml_str = xml_str.replace('MESH_FILENAME.exo', rel_exo)
xml_str = xml_str.replace('MESH_FILENAME.h5',  rel_h5)

xml = aio.fromString(xml_str)
print('Loaded template and updated mesh filename references.')

# Build new region / WRM / permeability entries via ats_input_spec,
# then convert to XML and splice into the loaded template.
#
# We build a scratch main_props list we never write directly — just use it to
# generate well-formed XML sublists that we splice into the template xml.

main_props = ats_input_spec.public.get_main()

# Regions: add all labeled sets from the extruded 3D mesh.
# add_soil_type (below) will also add per-soil-type regions, so between the two
# calls we cover everything.  The template's fixed-named regions ('computational
# domain', 'surface domain', 'bottom', 'surface', 'surface outlet', soil layers)
# will be replaced or kept by the splice logic in the next cell.
for ls in m3.labeled_sets:
    ats_input_spec.public.add_region_labeled_set(
        main_props, ls.name, ls.setid, rel_exo, ls.entity)
for ss in m3.side_sets:
    ats_input_spec.public.add_region_labeled_set(
        main_props, ss.name, ss.setid, rel_exo, 'FACE')

# WRM + permeability (no porosity — ELM provides base_porosity at runtime)
for ats_id in nrcs.index:
    props = nrcs.loc[ats_id]
    region_name = f'soil type {ats_id}'
    smoothing = 0.01 if props['van Genuchten n [-]'] < 1.5 else 0.0

    ats_input_spec.public.add_soil_type(
        main_props, region_name, label=ats_id, filename=rel_exo,
        porosity=None,                              # ELM provides base_porosity
        permeability=float(props['permeability [m^2]']),
        van_genuchten_alpha=float(props['van Genuchten alpha [Pa^-1]']),
        van_genuchten_n=float(props['van Genuchten n [-]']),
        residual_sat=float(props['residual saturation [-]']),
        smoothing_interval=float(smoothing),
    )

main_xml = ats_input_spec.io.to_xml(main_props)
print(f'Built region + WRM + permeability entries for {len(nrcs)} soil types.')

# Splice main_xml subtrees into the loaded template xml

# Regions: add/replace each new region into the template's region list
xml_regions = asearch.find_path(xml, ['regions'], no_skip=True)
for new_region in asearch.find_path(main_xml, ['regions'], no_skip=True):
    rname = new_region.getName()
    try:
        xml_regions.replace(rname, new_region)
    except aerrors.MissingXMLError:
        xml_regions.append(new_region)

# WRM parameters: replace single 'all layers' placeholder with per-soil entries
xml_mp = asearch.find_path(xml, ['state', 'model parameters'], no_skip=True)
new_wrm = asearch.find_path(main_xml, ['state', 'model parameters', 'WRM parameters'],
                             no_skip=True)
try:
    xml_mp.replace('WRM parameters', new_wrm)
except aerrors.MissingXMLError:
    xml_mp.append(new_wrm)

# Permeability evaluator: replace placeholder with per-soil-type tensor evaluator
xml_ev = asearch.find_path(xml, ['state', 'evaluators'], no_skip=True)
new_perm = asearch.find_path(main_xml, ['state', 'evaluators', 'permeability'],
                              no_skip=True)
try:
    xml_ev.replace('permeability', new_perm)
except aerrors.MissingXMLError:
    xml_ev.append(new_perm)

print('Spliced regions, WRM parameters, and permeability into template XML.')

# Write coweeta_stage2.xml: same mesh/region/WRM xml built above, with the
# flow PK's initial conditions restarting ATS pressure from stage 1a's
# checkpoint. Stage 2's case dir is a sibling of stage 1a's, both one level
# below the campaign dir, so the restart path climbs back up to the campaign
# dir and down into stage 1a's own run/ directory (matching the mesh path
# convention above and the pattern used by coweeta_ats.ipynb's
# write_transient()).
restart_path_stage2 = os.path.join('..', '..', '1a_ats_spinup', 'run', 'checkpoint_final.h5')

stage2_xml_str = aio.toString(xml)
stage2_xml = aio.fromString(stage2_xml_str)  # independent deep copy via round-trip

# Replace the subsurface flow PK's cold-start initial conditions
# (hydrostatic head) with a restart from stage 1a's checkpoint. Overland
# flow's initial conditions are left as-is ('initialize surface head from
# subsurface'): it always derives its state from the just-restarted
# subsurface pressure rather than getting its own restart file, matching the
# pattern in cyclic_steadystate-template.xml / transient-template.xml.
flow_ic = asearch.find_path(stage2_xml, ['PKs', 'flow', 'initial conditions'], no_skip=True)
for child in list(flow_ic):
    flow_ic.remove(child)
flow_ic.setParameter('restart file', 'string', restart_path_stage2)

output_filenames['ats_xml_stage2'] = toOutput(f'{name}_stage2.xml')
aio.toFile(stage2_xml, output_filenames['ats_xml_stage2'])
print('Wrote stage2 ATS XML (restart from', restart_path_stage2 + '):',
      output_filenames['ats_xml_stage2'])


def writeUserNlElmStage2(surfdata_filename, ats_xml_filename,
                          prev_case_dir_name, prev_start_year, prev_stop_n):
    """Write user_nl_elm for stage 2: coupled ELM+ATS cyclic steady-state.

    finidat restarts ELM state from the previous stage (1b); ATS pressure
    restart is instead baked into ats_xml_filename's own 'restart file'
    parameter (see the ATS XML write cell above), so it is not repeated here.
    """
    finidat_path = elmRestartPath(prev_case_dir_name, prev_start_year, prev_stop_n)
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! ELM IC from end of stage 1b
 finidat = '{finidat_path}'

 ! Cyclic spinup forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_spinup'
 const_climate_hist = .true.

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'
 CO2_file = '$DIN_LOC_ROOT/atm/datm7/CO2/fco2_datm_1765-2007_c100614.nc'

 ! No accelerated decomposition in coupled spinup
 nyears_ad_carbon_only = 0
 spinup_mortality_factor = 1

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 use_ats = .true.
 domain_decomp_type = 'ats'
 ats_inputdir = '.'
 ats_inputfile = '{ats_xml_filename}'

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage2'] = toOutput('user_nl_elm_stage2')
    with open(output_filenames['user_nl_elm_stage2'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage2'])
    return content


writeUserNlElmStage2(
    os.path.basename(output_filenames['elm_surfdata']),
    os.path.basename(output_filenames['ats_xml_stage2']),
    prev_case_dir_name='1b_elm_carbon_spinup', prev_start_year=1, prev_stop_n=200)


## Stage 3: ELM+ATS Transient Run
'''
Stage 3 of the multi-stage spinup workflow (see `examples/coweeta/build_example.sh`) is
the final, fully coupled ELM+ATS transient run, initialized from the end of stage 2 and
driven by real (non-cyclic) meteorology.

**ATS input**: reuses the same mesh/region/WRM `xml` object built in the Stage 2 section
above (identical PK tree and soil properties -- only the restart source differs), making
an independent second copy and writing `coweeta_stage3.xml`, with the `flow` PK's
`initial conditions` restarting ATS pressure from stage 2's `checkpoint_final.h5`
(`../../../2_cyclic_steadystate/run/checkpoint_final.h5`, relative to the stage 3 run
directory).

**ELM namelist**: written below by `writeUserNlElmStage3()` -- transient AORC cpl_bypass
forcing (`metdata_type = 'atssubdaily'`, `const_climate_hist = .false.`, pointing at
`DIN_LOC_CAMPAIGN/atm_forcing_transient`), `use_ats = .true.`, `domain_decomp_type = 'ats'`,
`ats_inputfile = 'coweeta_stage3.xml'`. The AORC zone files cover 2010-2022 at 1-hour
resolution; `zone_mappings.txt` maps every Coweeta cell to zone 19. ELM's carbon/biomass
state restarts from stage 2 via `finidat` (`../../2_cyclic_steadystate/run/....elm.r....nc`),
computed the same way as stage 2's `finidat` from stage 1b.

`user_nl_elm` for every ELM-running stage (1b, 2, 3) is written entirely by this
notebook (`writeUserNlElmStage1b/2/3()`, defined in the Stage 1b section above), not by
`build_example.sh` -- `user_nl_elm` is ELM's own input file, exactly like the ATS XML, so
it belongs alongside the other domain-specific input generation here rather than in a
general CIME-orchestration script. `build_example.sh` just copies the finished file into
each case directory before `case.setup`, the same way it already copies the ATS XMLs.
'''

# Write coweeta_stage3.xml: the same mesh/region/WRM xml built in the Stage 2
# section above, making an independent second copy with the flow PK's
# initial conditions instead restarting ATS pressure from stage 2's
# checkpoint.
restart_path_stage3 = os.path.join('..', '..', '2_cyclic_steadystate', 'run', 'checkpoint_final.h5')

stage3_xml_str = aio.toString(xml)
stage3_xml = aio.fromString(stage3_xml_str)  # independent deep copy via round-trip

flow_ic = asearch.find_path(stage3_xml, ['PKs', 'flow', 'initial conditions'], no_skip=True)
for child in list(flow_ic):
    flow_ic.remove(child)
flow_ic.setParameter('restart file', 'string', restart_path_stage3)

output_filenames['ats_xml_stage3'] = toOutput(f'{name}_stage3.xml')
aio.toFile(stage3_xml, output_filenames['ats_xml_stage3'])
print('Wrote stage3 ATS XML (restart from', restart_path_stage3 + '):',
      output_filenames['ats_xml_stage3'])


def writeUserNlElmStage3(surfdata_filename, ats_xml_filename,
                          prev_case_dir_name, prev_start_year, prev_stop_n):
    """Write user_nl_elm for stage 3: coupled ELM+ATS transient run.

    finidat restarts ELM state from stage 2; ATS pressure restart is baked
    into ats_xml_filename's own 'restart file' parameter (see the ATS XML
    write cell above).
    """
    finidat_path = elmRestartPath(prev_case_dir_name, prev_start_year, prev_stop_n)
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! ELM IC from end of stage 2
 finidat = '{finidat_path}'

 ! Transient forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_transient'
 const_climate_hist = .false.

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'
 CO2_file = '$DIN_LOC_ROOT/atm/datm7/CO2/fco2_datm_1765-2007_c100614.nc'

 nyears_ad_carbon_only = 0
 spinup_mortality_factor = 1

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 use_ats = .true.
 domain_decomp_type = 'ats'
 ats_inputdir = '.'
 ats_inputfile = '{ats_xml_filename}'

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage3'] = toOutput('user_nl_elm_stage3')
    with open(output_filenames['user_nl_elm_stage3'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage3'])
    return content


writeUserNlElmStage3(
    os.path.basename(output_filenames['elm_surfdata']),
    os.path.basename(output_filenames['ats_xml_stage3']),
    prev_case_dir_name='2_cyclic_steadystate', prev_start_year=1, prev_stop_n=10)


## Summary and Next Steps
'''
After this notebook completes, its `elm_output_data/` directory holds everything
needed for the multi-stage spinup workflow -- no copy step required.
`build_example.sh` (in `examples/coweeta/`) and its `stage_inputdata.sh` helper
read directly from `watershed_workflow/examples/Coweeta/elm_output_data/` and
stage what they need into the campaign's shared `inputdata/` automatically.

`user_nl_elm` -- ELM's own input file -- is written entirely by this notebook
(`writeUserNlElmStage1b/2/3()`, defined in the Stage 1b/2/3 sections above),
exactly like the ATS XMLs, rather than as a heredoc in `build_example.sh`:
`user_nl_elm` and the ATS XML are both domain-specific input generation that
belongs alongside the rest of this notebook's mesh/region/soil work, not in a
general CIME-orchestration script. `build_example.sh` only copies each
finished file into its case directory before `case.setup`.

`build_example.sh` also stages the handful of global ELM datasets (aerosol
deposition, CO2, population density, N deposition) into the campaign's
`inputdata/` automatically (via `stage_inputdata.sh`), reading them directly
from `ELM_ATS_SRC_DIR/inputdata` -- you do not need to clone the full
`inputdata` repo onto machines that only run an already-built campaign.

Also run `coweeta_aorc_elm.ipynb` (writes meteorology to the same
`elm_output_data/` directory), then build the full multi-stage spinup
workflow (stages 1a-3):

```bash
cd ../../../examples/coweeta
export NTASKS=${NTASKS}
./build_example.sh
```

See `examples/coweeta/README.md` for the full run order (stage 1a's manual
ATS invocation, then `case.submit` for stages 1b/2/3 in order).
'''

print(f'{"role":<35}: filename')
print('-' * 36 + ': ' + '-' * 50)
for k, v in output_filenames.items():
    print(f'{k:<35}: {v}')

