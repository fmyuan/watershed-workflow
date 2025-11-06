#!/usr/bin/env python
import os
import numpy as np
from netCDF4 import Dataset
from cmath import nan, inf
from pyproj import Transformer
from pyproj import CRS

import watershed_workflow.elm_gridlocater

try:
    from mpi4py import MPI
    HAS_MPI4PY=True
except ImportError:
    HAS_MPI4PY=False

#---
# 
def mksrfdata_updatevals(fsurfnc_in, user_srf_data={}, user_srfnc_file='', user_srf_vars='', OriginPFTclass=True):
    
    print('#--------------------------------------------------#')
    print("Replacing values in surface data by merging user-provided dataset")
    fsurfnc_out ='./'+fsurfnc_in.split('/')[-1]+'-merged'
    
    
    #--------------------------------------
    #
    # PFT classes in B. Sulman et al (2021) paper: 12 arctic PFTs + 2 additional tree PFTs
    
    user_pfts={'pftname':[
                    "non_vegetated",
                    "arctic_lichen",
                    "arctic_bryophyte",
                    "arctic_needleleaf_tree",
                    "arctic_broadleaf_tree",
                    "arctic_evergreen_shrub",
                    "arctic_evergreen_tall_shrub",
                    "arctic_deciduous_dwarf_shrub",
                    "arctic_deciduous_low_shrub",
                    "arctic_low_to_tall_willowbirch_shrub",
                    "arctic_low_to_tall_alder_shrub",
                    "arctic_forb",
                    "arctic_dry_graminoid",
                    "arctic_wet_graminoid"
                    ],
                'pftnum': [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
               };
    
    
    if OriginPFTclass:
        # lichen as not_vegetated (0), moss/forb/graminoids as c3 arctic grass (12),
        # evergreen shrub(9), deci. boreal_shrub(11),
        # evergreen boreal tree(2), deci boreal tree (3)
        user_pfts['pftnum'] = [0,0,12,2,3,9,9,11,11,11,11,12,12,12]
        natpft = np.asarray(range(17))
    else:
        natpft = np.asarray(range(max(user_pfts['pftnum'])+1)) # this is the real arcticpft order number 
 
    #--------------------------------------
    UNSTRUCTURED = False    
    if not user_srfnc_file !='':
        print('read data from: ', user_srfnc_file)
        f=Dataset(user_srfnc_file)
        if 'gridcell' in f.dimensions.items(): UNSTRUCTURED = True
    elif not len(user_srf_data)<=0:
        if len(np.squeeze(user_srf_data['LATIXY']).shape())==1:
            UNSTRUCTURED = True
            
        if user_srf_vars=='':
            user_vname = user_srf_vars.keys()
        else:
            user_vname = user_srf_vars.split(',')
        user_srf = user_srf_data
        
    #--------------------------------------
    #                    
    # write into nc file
    with Dataset(fsurfnc_in,'r') as src, Dataset(fsurfnc_out, "w") as dst:
            
        # new surfdata dimensions
        for dname, dimension in src.dimensions.items():
            if dname == 'natpft':
                len_dimension = len(natpft)            # dim length from new data
            elif dname == 'gridcell':
                len_dimension = user_srf['LATIXY'].flatten().size()
            elif dname in ['lsmlat','lat']:
                len_dimension = user_srf['LATIXY'].shape()[0]
            elif dname in ['lsmlon', 'lon']:
                len_dimension = user_srf['LONGXY'].shape()[1]                
            else:
                len_dimension = len(dimension)
            dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)
            #
            
        # create variables and write to dst
        for vname, variable in src.variables.items():

            if UNSTRUCTURED:
                vdim = variable.dimensions
                if 'gridcell' not in vdim and \
                    ('lsmlat' in vdim and 'lsmlon' in vdim):
                    vdim = vdim.replace('lsmlon', 'gridcell')
                    vdim = vdim.remove('lsmlat')
            else:
                vdim = variable.dimensions
    
            # create variables, but will update its values later 
            # NOTE: here the variable value size updated due to newly-created dimensions above
            dst.createVariable(vname, variable.datatype, vdim)
            # copy variable attributes all at once via dictionary after created
            dst[vname].setncatts(src[vname].__dict__)
                  
            # values
            src_vals = src[vname][...]

            # dimension length may change, so need to 
            if vname in user_vname:
                varvals = src[vname][...]
                #
            else:
                varvals = src[vname][...]
                #
            
            #                                
            dst[vname][...] = varvals
                    
        # end of variable-loop        
                
        
        print('user surfdata merged and nc file created and written successfully!')
        
    #
#

#--------------------------------------------------------------------
def test():

    """
    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]
    """  
    input_path  = './'
    output_path = './'
    
    mksrfdata_updatevals('./surfdata_0.5x0.5_simyr1850_c240308_TOP.nc', \
                         user_srf_data=surf_from_atsm2, user_srf_vars='', OriginPFTclass=False)
    
      
#if __name__ == '__main__':
#    test()




