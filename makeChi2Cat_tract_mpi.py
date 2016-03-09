# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:53:21 2016

@author: anneya

                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |
                
                
For a single HSC tract, create a chi^2 - selected multiband catalog 
"""
from __future__ import print_function, division
import sys, os
import glob
import argparse
import numpy as np
from astropy import units as u
from astropy.io import fits
import gc
from mpi4py import MPI
comm=MPI.COMM_WORLD

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------

def GoodPatches(drPath,tract,bands):
    '''
    Returns a list of patches where a calexp images exists for all bands 
    '''
    patches = []
    for band in bands:
        os.chdir(drPath+'deepCoadd/'+band+'/'+tract)
        bandPatches = np.array(glob.glob('*'))
        goodBandPatches = []
        for bp in bandPatches:
            if (os.path.exists(drPath+'deepCoadd/'+band+'/'+tract+'/'+bp+'/calexp-'+band+'-'+tract+'-'+bp[0]+','+bp[-1]+'.fits') or \
                os.path.exists(drPath+'deepCoadd/'+band+'/'+tract+'/'+bp+'/calexp-'+band+'-'+tract+'-'+bp[0]+bp[-1]+'.fits')):
                goodBandPatches.append(bp[0]+bp[-1])
        if len(patches)==0:
            patches = goodBandPatches
        else:
            patches = np.intersect1d(np.array(patches),goodBandPatches)
    return(patches)

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def KillCommas(drPath,tract,bands,patches):
    '''
    Remove commas from paths and image names so SExtractor can parse
    '''
    for band in bands:
        for patch in patches:
            try:
                os.chdir(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch[0]+patch[-1])
            except:
                os.chdir(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch[0]+','+patch[-1])
            if os.path.isfile('calexp-'+band+'-'+tract+'-'+patch[0]+','+patch[-1]+'.fits'):
                os.system('mv calexp-'+band+'-'+tract+'-'+patch[0]+','+patch[-1]+'.fits calexp-'+band+'-'+tract+'-'+patch[0]+patch[-1]+'.fits')
            if not os.path.isfile('calexp-'+band+'-'+tract+'-'+patch[0]+patch[-1]+'.fits'):
                print('Broken link?')
                print(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch)
        for patch in patches:
            if os.path.exists(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch[0]+','+patch[-1]):
                os.system('mv '+drPath+'deepCoadd/'+band+'/'+tract+'/'+patch[0]+','+patch[-1]+'/ '+\
                            drPath+'deepCoadd/'+band+'/'+tract+'/'+patch[0]+patch[-1]+'/')    
    
#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def MakeChi2Image(fluxData,varMaps):
    '''
    Calculates chi2 image from arrays of flux data and corresponding variance maps
    Inputs:
        fluxData = NumPy array of shape(nImages,nXpixels,nYpixels) containing fluxes
        varMaps = NumPy array of shape(nImages,nXpixels,nYpixels) containing variances
    Outputs:
        Numpy array of shape(nXpixels,nYpixels), chi2
    '''
    medians = np.median(fluxData.reshape(fluxData.shape[0],np.product(fluxData.shape[1:])),axis=1)
    chi2 = np.zeros(shape=fluxData[0].shape)   
    for i in range(len(fluxData)):
        chi2 += ((fluxData[i]-medians[i])/np.sqrt(varMaps[i]))**2
    return(np.sqrt(chi2))

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def MakeChi2HSC(images,baseIm,outputImage,clobber):
    '''
    Calls MakeChi2Image() to create a chi2 image from a list of HSC images.
    Inputs:
        images = list of filenames of HSC images with extensions [IMAGE] and [VARIANCE]
        baseIm = filename of an image whose [IMAGE] header will be copied for WCS information
        outputImage = filename of output image
    Outputs:
        None
    '''
    fluxData,varMaps = [],[]
    for image in images:
        f = fits.open(image)
        fluxData.append(f['IMAGE'].data)
        varMaps.append(f['VARIANCE'].data)
    header = fits.getheader(baseIm,'IMAGE')
    for i in range(len(images)):
        header['chi2in_'+str(i)] = images[i]
    fluxData = np.array(fluxData)
    varMaps = np.array(varMaps)
    chi2 = MakeChi2Image(fluxData,varMaps)
    gc.collect()
    fits.writeto(outputImage, chi2, header,clobber=clobber)
    
#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def MakeChi2Images(drPath,tract,bands,patches,clobber,prefix):
    '''
    Make chi2 image and variance map for each patch in patches
    '''
    # Create chi2 'filter' directory with all patches
    if not os.path.exists(drPath+'deepCoadd/'+prefix+'chi2/'):
        os.system('mkdir '+drPath+'deepCoadd/'+prefix+'chi2/')
    if not os.path.exists(drPath+'deepCoadd/'+prefix+'chi2/'+tract):
        os.system('mkdir '+drPath+'deepCoadd/'+prefix+'chi2/'+tract)
    for patch in patches:
        if not os.path.exists(drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch):
            os.system('mkdir '+drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch)
    
    # Make images
    if not os.path.exists(drPath+'deepCoadd/'+prefix+'chi2/'):
        os.system('mkdir '+drPath+'deepCoadd/'+prefix+'chi2/')
    for patch in patches:
        outImage = drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch+'/'+prefix+'chi2-'+tract+'-'+patch+'.fits'       
        if not (clobber==False and os.path.isfile(outImage)):
            images = []
            for band in bands:
                images.append(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch+'/calexp-'+band+'-'+tract+'-'+patch[0]+patch[-1]+'.fits')
            MakeChi2HSC(images,images[0],outImage,clobber)
            gc.collect()

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def SExtractChi2(drPath,tract,patches,prefix,sexdir,dotsex):
    '''
    Run SExtractor on chi2 images so photometry from detection image can be used for corrections
    '''
    os.chdir(sexdir)
    for patch in patches:
        inImage = drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch+'/'+prefix+'chi2-'+tract+'-'+patch+'.fits'
        outCat = drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch+'/'+prefix+'chi2-'+tract+'-'+patch+'.cat'
        os.system('sex '+inImage+' -c '+dotsex+' -CATALOG_NAME '+outCat)
    
#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def SExtractorDualImage(drPath,tract,bands,patches,prefix,sexdir,dotsex,zps):
    '''
    Run SExtractor in dual image mode for each patch,band using chi2 image for detection
    '''
    os.chdir(sexdir)
    for band in bands:
        for patch in patches:
            # Move Variance extension to new file for SExtractor
            imname = drPath+'deepCoadd/'+band+'/'+tract+'/'+patch+'/'+\
                        'calexp-'+band+'-'+tract+'-'+patch[0]+patch[-1]
            varHead = fits.getheader(imname+'.fits','VARIANCE')
            maskHead = fits.getheader(imname+'.fits','MASK')
            im = fits.open(imname+'.fits')
            fits.writeto(imname+'_var.fits',im['VARIANCE'].data,varHead,clobber=True)
            fits.writeto(imname+'_mask.fits',im['MASK'].data.astype(float),maskHead,clobber=True)
            inImage = imname+'.fits[1]'
            outCat = drPath+'deepCoadd/'+band+'/'+tract+'/'+patch+'/'+band+'-'+tract+'-'+patch[0]+patch[-1]+'-chi2.cat'
            chi2Image = drPath+'deepCoadd/'+prefix+'chi2/'+tract+'/'+patch+'/'+prefix+'chi2-'+tract+'-'+patch+'.fits'
            # Run SExtractor
            os.system('sex '+chi2Image+','+inImage+' -c '+dotsex+' -CATALOG_NAME '+outCat+' -WEIGHT_IMAGE None,'+\
                        imname+'_var.fits -WEIGHT_TYPE NONE,MAP_VAR -MAG_ZEROPOINT '+str(zps[bands.index(band)]))
            # Add flags to catalog
            os.system(drPath+'./venice -m '+imname+'_mask.fits -cat '+outCat+ ' -f all -xcol 2 -ycol 3 -o '+\
                    drPath+'deepCoadd/'+band+'/'+tract+'/'+patch+'/'+band+'-'+tract+'-'+patch[0]+patch[-1]+'-chi2-flags.cat')
            
            os.system('rm '+imname+'_var.fits')
            os.system('rm '+imname+'_mask.fits')
            

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def CatPatch(drPath,tract,bands,patch,onceCols,apertures,chi2prefix,zps):
    '''
    Generate multiband catalog for a given patch
    '''
    # Make Header for combined catalog
    columns,bigColumns = [],[]
    fluxCols,aperFluxCols = [],[]
    f = open(drPath+'deepCoadd/'+bands[0]+'/'+tract+'/'+patch+'/'+bands[0]+'-'+tract+'-'+patch[0]+patch[-1]+'-chi2-flags.cat','r')
    i = 0    
    for line in f.readlines():
        if line[0]=='#':
            if 'APER' in line:
                for aperture in apertures:
                    columns.append(aperture+'_'+line[6:-1])
            else:
                columns.append(line[6:-1])  
            if 'FLUX_' in line and not 'RADIUS' in line and not 'APER' in line:
                fluxCols.append(i)
            if 'FLUX_APER' in line:
                aperFluxCols.append(i)
            i+=1
    f.close()
    columns.append('flags')
    for i in onceCols:
        bigColumns.append(columns[i])
    for band in bands:
        keepCols = []
        for i in range(len(columns)):
            if not i in onceCols:
                keepCols.append(i)
                bigColumns.append(band+'_'+columns[i])
    header = ''
    i = 0
    for h in bigColumns:
        header+='('+str(i)+') '+h+'\n'
        i+=1
    print(fluxCols)
    # combine Catalogs from each band
    bigCat = None
    for band in bands:      
        cat = np.genfromtxt(drPath+'deepCoadd/'+band+'/'+tract+'/'+patch+'/'+band+'-'+tract+'-'+patch[0]+patch[-1]+'-chi2-flags.cat')
        # convert fluxes to Jy
        for col in fluxCols:
            diffrac = cat[:,col]/cat[:,col+1]
            newFlux = cat[:,col] * 10**((8.9-zps[bands.index(band)])/2.5)
            newFluxErr = diffrac * newFlux
            cat[:,col+1] = newFluxErr
            cat[:,col] = newFlux
        for col in aperFluxCols:
            diffrac = cat[:,col]/cat[:,col+len(apertures)]
            newFlux = cat[:,col] * 10**((8.9-zps[bands.index(band)])/2.5)
            newFluxErr = diffrac * newFlux
            cat[:,col] = newFlux
            cat[:,col+len(apertures)] = newFluxErr
            
        if bigCat == None:
            bigCat = np.c_[cat]
        else:
            bigCat = np.c_[bigCat,cat[:,keepCols]]
            
    # add unique identifiers
    ids = []
    for j in bigCat[:,0]:
        ids.append(tract+patch+str(j))
    ids = np.array(ids)
    bigCat[:,0] = ids
    
    print('LEN(HEAD)=',len(bigColumns),'nColumns=',len(bigCat[0]))
    # identify objects in regions that overlap with other patches
    # get patch dimension
    imname = drPath+'deepCoadd/'+bands[0]+'/'+tract+'/'+patch+'/'+\
                        'calexp-'+bands[0]+'-'+tract+'-'+patch[0]+patch[-1]
    f = fits.open(imname+'.fits')
    
    # Add binary flags for objects in overlap region and those to be removed in tract catalog
    patchDims=[f[1].data.shape[0],f[1].data.shape[1]]
    xmax,ymax=patchDims[1]-100,patchDims[0]-100
    overlap = np.logical_not(np.sum(np.c_[bigCat[:,1]<100,bigCat[:,2]<100,bigCat[:,1]>xmax,bigCat[:,2]>ymax].astype(int),axis=1))
    xmax,ymax=patchDims[1]-50,patchDims[0]-50
    remove = np.logical_not(np.sum(np.c_[bigCat[:,1]<50,bigCat[:,2]<50,bigCat[:,1]>xmax,bigCat[:,2]>ymax].astype(int),axis=1))
    header+='('+str(i)+') in_overlap_region \n'
    i+=1
    header+='('+str(i)+') remove_for_unique'
    np.savetxt(drPath+'chicat/'+tract+'_'+patch+'_'+chi2prefix+'_chi2.cat',np.c_[bigCat,overlap,remove],header=header.replace('count','jansky'))

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def TractCat(drPath,tract,patches,chi2prefix):
    '''
    Combine multiband catalogs for each patch in tract
    '''
    flagCols,header = [],''
    f = open(drPath+'chicat/'+tract+'_'+patches[0]+'_'+chi2prefix+'_chi2.cat','r')
    i = 0
    for line in f.readlines():
        if line[0]=='#':
            header+=line[2:]
        if 'flags' in line:
            flagCols.append(i)
        i+=1
    f.close()
    
    bigCat = None
    for patch in patches:
        cat = np.genfromtxt(drPath+'chicat/'+tract+'_'+patch+'_'+chi2prefix+'_chi2.cat')
        flags = cat[:,flagCols].astype(int)
        bad = flags>=256
        bad = np.sum(np.c_[bad,np.logical_not(cat[:,-1].astype(int).astype(bool)).astype(int)],axis=1)
        goodN = np.where(bad==0)
        if bigCat == None:
            bigCat = cat[goodN]
        else:
            bigCat = np.r_[bigCat,cat[goodN]]
    np.savetxt(drPath+'chicat'+'/'+tract+'_'+chi2prefix+'_chi2.cat',bigCat,header=header[:-1])
#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
def CatToFits(inCat,outCat,deredCat,cs,bands,getDustLoc):
    '''
    Create .fits table from ascii catalog, deredden
    '''
    colNames,colUnits,colFormats = [],[],[]
    f = open(inCat,'r')
    for line in f.readlines():
        if line[0]=='#':
            lis = line[3:-1].split()
            name = lis[1]
            colNames.append(name.replace('-','_'))
            if name=='NUMBER' or 'flag' in name:
                colFormats.append('K')
            else:
                colFormats.append('D')
            if '[' in line and ']' in line:
                unit = line[line.index('[')+1:line.index(']')]
                if unit =='deg':
                    unit = 'degree'
                colUnits.append(unit)
            else:
                colUnits.append(None)
        else:
            break
    f.close()
    cat = np.genfromtxt(inCat)
    cols = []
    for k in range(len(cat[0])):
        cols.append(fits.Column(name=colNames[k],format=colFormats[k],unit=colUnits[k],array=cat[:,k]))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(outCat,clobber=True)
    
    # deredden
    print('Dereddening...')
    coeffs,bs = '',''
    for c in colNames:
        if 'FLUX' in c and not 'ERR' in c and not 'RADIUS' in c and not 'CHI2' in c:
            bs+=c+','
            ind = None
            for band in bands:
                band=band.replace('-','_')
                if band in c:
                    ind = c.index(band)
                    break
            coeffs+=str(cs[ind])+','
    os.system(getDustLoc+'./getDust.py '+outCat+' '+deredCat+' -band '+bs[:-1]+' -coef '+coeffs[:-1]+' -correct '+\
                '-s '+getDustLoc+'SFD_dust_4096_sgp.fits -n '+getDustLoc+'SFD_dust_4096_ngp.fits' )


#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
parser.add_argument("-b", "--bands",nargs='+', help="bands to include in chi2 images",\
                    default=['MegaCam-u','HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y'])
parser.add_argument("-p","--dataPath",help="full path to data release (where deepCoadds is a subfolder)",\
                    default = '/home/agolob/data/HSC/s15b_deep/')
parser.add_argument("--tract",help='HSC tract id', default = '8524')
parser.add_argument("--clobberChi2",help='if True, will overwrite existing chi2 images',\
                    default = False)
parser.add_argument("--chi2prefix",help="prefix to be added to chi2 images",default='ugrizy')
parser.add_argument("--sexdir",help="Path to directory where sextractor control files live, venice and getDust.py stuff assumed there",\
                    default = '/home/agolob/local/source/HSCchi2Cat/')

parser.add_argument("--dotsex",help="SExtractor .param file",default='default.sex')
parser.add_argument("--zps",help="ZeroPoints for each band in bands",nargs='+',\
                            default=[30.,27.,27.,27.,27.,27.])
parser.add_argument("--onceCols",help="Columns in dual image SExtractor catalogs that should only appear once in combined catalog",\
                    nargs='+',default=[0,1,2,3,4])
parser.add_argument("--apertures",help="List of apertures",nargs='+',\
                    default=['12pix','24pix'])
parser.add_argument("--deredCoeffs",help="Corresponding Albda/E(B-V) (in mags)",nargs='+',\
                    default=[4.732,3.711,2.626,1.916,1.469,1.242])
args = parser.parse_args()

# Locate patches where all bands have coadds
print('Locating patches where all bands have coadds...')
patches = GoodPatches(args.dataPath,args.tract,args.bands)
# MPI
rank = comm.Get_rank()
nProcs = comm.Get_size()
nEach = len(patches)//nProcs

# Remove commas
if rank==0:
    print('Removing commas...')
    KillCommas(args.dataPath,args.tract,args.bands,patches)
comm.Barrier()

# Get new patch names
allpatches = GoodPatches(args.dataPath,args.tract,args.bands)
if rank==nProcs-1:
    patches = allpatches[rank*nEach:]
else:
    patches = allpatches[rank*nEach:rank*nEach+nEach]
print(rank,patches)
# Make chi2 image for each good patch
if rank==0:
    print('Making chi^2 images...')
MakeChi2Images(args.dataPath,args.tract,args.bands,patches,args.clobberChi2,args.chi2prefix)
gc.collect()

# Run SExtractor on chi2 images
#print('SExtracting on chi2 images...')
#SExtractChi2(args.dataPath,args.tract,patches,args.chi2prefix,args.sexdir,args.dotsex)

# Run SExtractor in dual image mode on other bands
if rank==0:
    print('SExtracting in dual image mode...')
SExtractorDualImage(args.dataPath,args.tract,args.bands,patches,args.chi2prefix,args.sexdir,args.dotsex,args.zps)
gc.collect()

# Create a multiband catalog for each patch
if rank==0:
    print('Creating multiband catalogs for each patch...')
for patch in patches:
    CatPatch(args.dataPath,args.tract,args.bands,patch,args.onceCols,args.apertures,args.chi2prefix,args.zps)
gc.collect()
comm.Barrier()
# Make multiband catalog for tract from catalogs from each patch 
if rank==0:
    print('Creating catalog for tract '+args.tract+'...')
    TractCat(args.dataPath,args.tract,patches,args.chi2prefix)
    gc.collect()
# Convert to fits and deredden
if rank==0:
    print('Creating .fits catalog...')
    CatToFits(args.dataPath+'chicat'+'/'+args.tract+'_'+args.chi2prefix+'_chi2.cat',\
            args.dataPath+'chicat'+'/'+args.tract+'_'+args.chi2prefix+'_chi2.fits',\
            args.dataPath+'chicat'+'/'+args.tract+'_'+args.chi2prefix+'_chi2_dered.fits',\
            args.deredCoeffs,args.bands,args.sexdir)



    



