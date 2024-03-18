import ee
# Initialize the Earth Engine module.
ee.Initialize()

import re



# # Imagery
# 
# Gather and setup the imagery to use for inputs (predictors).  This is a three-year, cloud-free, Landsat 8 composite.  Display it in the notebook for a sanity check.

# In[301]:


class input_maker:
    def __init__(self, YEAR_SEL, BOUND):
        ##########################################################################
        # Use Landsat 7 surface reflectance data.
        #self.landsat7 = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
        self.YEAR_SEL = YEAR_SEL
        self.landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") #This is actually landsat 8
        self.landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        self.CDLCollection = ee.ImageCollection("USDA/NASS/CDL")
        self.Elevation = ee.Image('USGS/3DEP/10m').select('elevation')
        self.Slope = ee.Terrain.slope(self.Elevation).divide(90)
        #self.crop = ee.ImageCollection()

        self.CRS_ = 'EPSG:3857'
        self.SCALE_ = 30


        # Define a kernel.
        self.kernel = ee.Kernel.circle(radius=1)
        self.PI = ee.Number(-1).acos()
        self.STEPS = 100
        self.t = ee.List.sequence(0, self.STEPS-1, 1)
        #dts = t.map(get_dx_dy)
        self.rmin = 30 # 1 pixel in NASS = 30 m = 90 feet
        self.rmax = 630 # 2000 feet max for searching circles
        self.R = ee.List.sequence(self.rmin, self.rmax, 30)

        self.utm_projection = ee.Projection(self.CRS_)
        self.nonzeroreducer = ee.Reducer.countEvery()
        self.count_reducer = ee.Reducer.count()
        self.median_reducer = ee.Reducer.median()
        self.max_reducer = ee.Reducer.max()
        self.mean_reducer = ee.Reducer.mean()
        self.sd_reducer = ee.Reducer.stdDev()
        self.and_reducer = ee.Reducer.bitwiseAnd()
        self.meanstdreducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs= True)
        self.NORMALIZATION_KERNEL = ee.Kernel.square(radius=5000, units='meters', normalize=True, magnitude=1)

        self.circle_kernel = ee.Kernel.circle(3)
        
        # Period to use to grab images
        # 0 = Apr-Aug, 1 = Apr-June, 2 = July-Aug
        self.IMPERIOD = 0 
        self.PERIODDATA = [{'folder':'Landsat_April_August','months':[3,8]},
                      {'folder':'Landsat_April_June','months':[3,6]},
                      {'folder':'Landsat_July_August','months':[6,8]}]
        self.FOLDER = self.PERIODDATA[self.IMPERIOD]['folder']
        self.MONTHS = self.PERIODDATA[self.IMPERIOD]['months']

        
        #Data conditioning
        self.REGEX5 = ['.*(SR_B[123457])','.*(ST_B6)']
        self.REGEX8 = ['.*(SR_B[234567])','.*(ST_B10)']
        self.REGEX = []
        self.BGR = []
        self.LAYERS = ['BGR0', 'BGR1','BGR2', 'SWIR0', 'SWIR1','SWIR2','SR_TH']
        self.LAYERSDIFF = [lay+'_diff' for lay in self.LAYERS]
        
        self.MINREDUCER = ee.Reducer.min()
        self.MAXREDUCER = ee.Reducer.max()
        self.MEDIANREDUCER = ee.Reducer.median()
        self.BOUND = BOUND
        self.COVERGRID = BOUND.coveringGrid(proj=self.CRS_,scale=self.SCALE_*64*5)
        self.PAT = '(?P<year>\d{4})$'
        self.NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
        self.AGAREAS = self.get_nlcd_img(YEAR_SEL)
        self.make_inputs()
        
    def make_inputs(self):
        # NDVI STUFF
        def maskL8sr_and_ndvi(image) :
          # Get the pixel QA band.
          qa = image.select('QA_PIXEL')
          # Both flags should be set to zero, indicating clear conditions.
          # Bit 0 - Fill
          # Bit 1 - Dilated Cloud
          # Bit 2 - Cirrus
          # Bit 3 - Cloud
          # Bit 4 - Cloud Shadow 
          qaMask = qa.bitwiseAnd(int('0b11111', base=2)).eq(0)
          saturationMask = image.select('QA_RADSAT').eq(0)
          mask = qaMask.And(saturationMask)
          image = image.updateMask(mask)

          #For thermal bands
          image = image.select(['SR_B2', 'SR_B3','SR_B4', 'SR_B5', 'SR_B6','SR_B7','ST_B10']).rename(self.LAYERS)

          # Return the masked and scaled data, without the QA bands.
          #image = image.clip(self.BOUND)
          return image



        # NDVI STUFF
        def maskL5sr_and_ndvi(image) :
          qa = image.select('QA_PIXEL')
          # Bit 0 - Fill
          # Bit 1 - Dilated Cloud
          # Bit 2 - Unused
          # Bit 3 - Cloud
          # Bit 4 - Cloud Shadow 
          qaMask = qa.bitwiseAnd(int('0b11011', base=2)).eq(0)
          saturationMask = image.select('QA_RADSAT').eq(0)
          mask = qaMask.And(saturationMask)
          #mask = ee.Image(1)
          image = image.updateMask(mask)
          #Landsat 5: ndvi is between bands 4 and 3

          #For thermal bands
          image = image.addBands(srcImg=image.select('ST_B6').rename('SR_TH'), names=['SR_TH'])
          image = image.select(['SR_B1', 'SR_B2','SR_B3', 'SR_B4', 'SR_B5','SR_B7','ST_B6']).rename(self.LAYERS)

          # Return the masked and scaled data, without the QA bands.
          #image = image.clip(self.BOUND)
          return image



        def get_year_range(year_in):
          tz = 'US/Mountain'
          year_str = '{:d}'.format(year_in)
          year_val = ee.String(year_str)
          ##Tighten this band to get more of early or later
          year_begin = year_val.cat('-{:02d}-01'.format(self.MONTHS[0]))
          year_end = year_val.cat('-{:02d}-30'.format(self.MONTHS[1]))
          year_range = ee.DateRange(year_begin, year_end, tz)
          return year_range



        def get_landsatcollection(year_in):
          year_range = get_year_range(year_in)
          if year_in >= 2013:
            col_ndvi = self.landsat8.filterDate(year_range).map(maskL8sr_and_ndvi)
            self.REGEX = self.REGEX8
            self.BGR = ['SR_B2', 'SR_B3','SR_B4']
            self.SWIR = ['SR_B5', 'SR_B6','SR_B7']
          else:
            col_ndvi = self.landsat5.filterDate(year_range).map(maskL5sr_and_ndvi)
            self.REGEX = self.REGEX5
            self.BGR = ['SR_B1', 'SR_B2','SR_B3']
            self.SWIR = ['SR_B4', 'SR_B5','SR_B7']
          return col_ndvi
    
        def normalize_raster(rasterIn, var_name):
          rasterIn_ = rasterIn.updateMask(self.AGAREAS)
          mean_rast = rasterIn_.select(var_name).reduceNeighborhood(reducer=self.mean_reducer,
                                                           kernel=self.NORMALIZATION_KERNEL,
                                                               skipMasked = False).rename(var_name)
          sd_rast = rasterIn_.select(var_name).reduceNeighborhood(reducer=self.sd_reducer,
                                                           kernel=self.NORMALIZATION_KERNEL,
                                                               skipMasked = False).rename(var_name)
          normalized_raster = rasterIn.subtract(mean_rast).divide(sd_rast).rename(var_name)
          return normalized_raster

        def composite_by_mon(mon_begin):
          mon_begin = ee.Date(mon_begin)
          mon_end = mon_begin.advance(delta = 2.0, unit = 'month', timeZone = tz)
          mon_range = ee.DateRange(mon_begin, mon_end)
          layer_names = self.landsat_col.first().bandNames()
          these_imgsm = self.landsat_col.filterDate(mon_range).reduce(self.MINREDUCER).rename(layer_names)
          these_imgsM = self.landsat_col.filterDate(mon_range).reduce(self.MAXREDUCER).rename(layer_names)
          avg_imgs = self.landsat_col.filterDate(mon_range).reduce(self.MEDIANREDUCER).toInt32()

          Diff = these_imgsM.subtract(these_imgsm).rename(self.LAYERSDIFF).toInt32()
          to_output = ee.Image.cat([avg_imgs,Diff]).set('Period',mon_begin.format('YYYYMM'))
          return to_output



        def Normalize_Rasters(imgIN):
          imgIN = ee.Image(imgIN)
          imgIN_ = imgIN.updateMask(self.AGAREAS)
          bandNames = imgIN.bandNames()
          mean_rast = imgIN_.reduceRegions(collection=self.COVERGRID,reducer=self.mean_reducer,
                                        scale = self.SCALE_*5,
                                         crs = self.CRS_,
                                         tileScale = 16).reduceToImage(properties=[ee.String('').cat(bandNames.get(0)).cat('_mean')], reducer=ee.Reducer.first())
          sd_rast = imgIN_.reduceRegions(collection=self.COVERGRID,reducer=self.sd_reducer,
                                        scale = self.SCALE_*5,
                                         crs = self.CRS_,
                                         tileScale = 16).reduceToImage(properties=[ee.String('').cat(bandNames.get(0)).cat('_stdDev')], reducer=ee.Reducer.first())
          normalized_raster = imgIN.subtract(mean_rast).divide(sd_rast).rename(bandNames)
          normalized_raster = normalized_raster.unitScale(-3, 3).multiply(65536).toInt32()
          normalized_raster = normalized_raster.set('Test','OK')
          return normalized_raster
        def create_mean_bname(bname):
            bname = ee.String(bname)
            return bname.cat('_mean')
        def create_std_bname(bname):
            bname = ee.String(bname)
            return bname.cat('_stdDev')
        self.landsat_col = get_landsatcollection(self.YEAR_SEL)
        
        mon_starts = []
        tz = 'US/Mountain'
        self.BANDNAMES = []
        for mon in range(4,8):
            monbeg = ee.Date(ee.String(f'{self.YEAR_SEL}').cat('-%d-01'%mon), tz)
            #monend = monbeg.advance(delta = 1, unit = 'month', timeZone = tz)
            mon_starts.append(monbeg)
            self.BANDNAMES += [f'{mon-4}_{lay}_median' for lay in self.LAYERS] + [f'{mon-4}_{lay}_diff' for lay in self.LAYERS]
        self.BANDNAMES += [f'{lay}_stdDev' for lay in self.LAYERS]
        mon_starts = ee.List(mon_starts)
        print(mon_starts.getInfo())
        imgcoll = ee.ImageCollection(mon_starts.map(baseAlgorithm = composite_by_mon, dropNulls = True))
        self.imgBands = imgcoll.toBands()
        season_stdDev = self.landsat_col.filterDate(ee.DateRange(f'{self.YEAR_SEL}-04-01', f'{self.YEAR_SEL}-11-01')).reduce(self.sd_reducer).toInt32()
        img_bands_temp = ee.Image.cat([self.imgBands,season_stdDev])
        img_bands_tempAg = img_bands_temp.updateMask(self.AGAREAS)
        BANDNAMES = img_bands_temp.bandNames()
        mean_rast_regions = img_bands_tempAg.reduceRegions(collection=self.COVERGRID,reducer=self.mean_reducer,
                                        scale = self.SCALE_*5,
                                         crs = self.CRS_,
                                         tileScale = 16)
        mean_rast = ee.Image.cat([mean_rast_regions.reduceToImage(properties=[bname_], reducer=ee.Reducer.first()).rename(bname_) for bname_ in self.BANDNAMES])
        
        sd_rast_regions = img_bands_tempAg.reduceRegions(collection=self.COVERGRID,reducer=self.sd_reducer,
                                        scale = self.SCALE_*5,
                                         crs = self.CRS_,
                                         tileScale = 16)
        sd_rast = ee.Image.cat([sd_rast_regions.reduceToImage(properties=[bname_], reducer=ee.Reducer.first()).rename(bname_) for bname_ in self.BANDNAMES])
        normalized_raster = img_bands_temp.subtract(mean_rast).divide(sd_rast).rename(BANDNAMES)
        normalized_raster = normalized_raster.unitScale(-3, 3).multiply(65536).toInt32()
        self.inputbands = normalized_raster
        
            
        return 0




    

    def get_closest_year(self, year, year_list):
        year_out = 9999
        year_diff = 9999
        for year_ in year_list:
            matched = re.match(self.PAT, year_)
            if matched:
                cur_year = int(matched.group('year'))
                cur_year_diff = abs(year-cur_year)
                if cur_year_diff <= year_diff:
                    year_out = cur_year
                    year_diff = cur_year_diff + 0
        print(f'{year}, closest = {year_out}')
        return year_out

    

    # Import the NLCD collection.
    def get_nlcd_img(self, YEAR):
        # The collection contains images for multiple years and regions in the USA.
        Products = self.NLCD.aggregate_array('system:index').getInfo()
        print(Products)

        self.nlcdYear = self.get_closest_year(YEAR, Products)

        # Filter the collection to the 2016 product.
        nlcd_out = self.NLCD.filter(ee.Filter.eq('system:index', f'{self.nlcdYear}')).first()

        # Each product has multiple bands for describing aspects of land cover.
        print('Bands:', nlcd_out.bandNames().getInfo())

        # Select the land cover band.
        landcover82 = nlcd_out.select('landcover').eq(82)
        landcover81 = nlcd_out.select('landcover').eq(81)
        landcover = landcover82.Or(landcover81)
        return landcover



    

    
