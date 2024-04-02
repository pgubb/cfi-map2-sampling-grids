"""
Utilities used by example notebooks
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import math 
import geojson

from gadm import GADMDownloader
import geopandas as gpd
import pandas as pd

from geopy.distance import great_circle
from shapely.geometry import (
    shape,
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon
)

import ee
# Authenticate and initialize Earth Engine API
#ee.Authenticate()  # Only required once
ee.Initialize(
    opt_url='https://earthengine-highvolume.googleapis.com'
)

import datetime
from datetime import date

###################################
# RASTER AND VECTOR VIZ FUNCTIONS
###################################

def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_image_with_vectors(
    image: np.ndarray, boundaries: list, edge_colors: list, face_colors: list, line_widths: list, title: str = "", filename: str = "tempfile", factor: float = 1.0, plotannotation: Optional[str] = None, addadminlabels = False, labelcolor = 'white', clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images with additional vector."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        image = np.clip(image * factor, *clip_range)
        image.plot.imshow(ax = ax)
    else:
        image = image * factor
        image.plot.imshow(ax = ax)
    i = 0
    for boundary in boundaries:
        boundary.plot(ax =ax, edgecolor=edge_colors[i], facecolor = face_colors[i], linewidth=line_widths[i])
        i += 1
    if addadminlabels:
        for idx, row in boundaries[0].iterrows():
            plt.text(row.geometry.centroid.x, row.geometry.centroid.y, row['NAME_2'], fontsize=14, color = labelcolor)
    if plotannotation:
        # Position text at bottom right of plot 
        ax.text(0.99, 0.01, plotannotation, verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize = 20)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()

    fig.savefig(filename, dpi=300, bbox_inches='tight')


###########################
# GADM FUNCTIONS
###########################
    
def get_adminl1_names_fromGADM(country_name): 
    downloader = GADMDownloader(version="4.0")
    admin_gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=1)
    # Print a list of all unique admin level 1 names
    print('Unique admin level 1 names: ', admin_gdf['NAME_1'].unique(), "\n")

def get_adminl2_names_fromGADM(country_name, admin_l1_name): 
    downloader = GADMDownloader(version="4.0")
    admin_gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=2)
     # Keep only rows for admin level 1 in gdf
    admin_gdf = admin_gdf[admin_gdf['NAME_1'] == admin_l1_name]
    # Print a list of all unique admin level 1 names
    print('Unique admin level 2 names: ', admin_gdf['NAME_2'].unique(), "\n")

def find_admin_names_fromGADM(country_name): 
    get_adminl1_names_fromGADM(country_name)
    admin_l1_name = input('Enter the name of desired admin level 1 area (without ''): ')
    get_adminl2_names_fromGADM(country_name, admin_l1_name)
    admin_l2_name = []
    print("Enter the name of the desired admin level 2 area(s) (without '', press 'x' to finish):")  
    while True:  
        element = input()  
        if element == 'x':  
            break  
        admin_l2_name.append(element)  
    #admin_l2_name = input('Enter the name of the desired level 2 area(s): ')
    return(admin_l1_name, admin_l2_name)

def get_country_boundary_fromGADM(country_name):
    # Returns the outer boundary of the country defined by the contiguous subnational units of the GADM database
    downloader = GADMDownloader(version="4.0")
    country_gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=0)
    return country_gdf

def get_alladmin_fromGADM(country_name, admin_level= 2):
    # Returns the outer boundary of the admin area defined by the contiguous subnational units of the GADM database
    downloader = GADMDownloader(version="4.0")
    admin_gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=admin_level)
    return admin_gdf

def get_admin_l2_fromGADM(country_name, admin_l1_name, admin_l2_name = [], admin_level=2, join = True): 
    # Returns the outer boundary of the admin area defined by the contiguous subnational units of the GADM database defined in admin_name
    downloader = GADMDownloader(version="4.0")
    admin_gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=admin_level)

    # Keep only rows for admin level 1 in gdf
    admin_gdf = admin_gdf[admin_gdf['NAME_1'] == admin_l1_name]

    if (admin_level == 2):
        # Keep only rows  in admin_gdf where 'NAME 2' equals one of the values in admin_l2_name
        print('This admin level 1 area comprised of the following level 2 areas:\n', admin_gdf['NAME_2'].unique())
        admin_gdf = admin_gdf[admin_gdf['NAME_2'].isin(admin_l2_name)]
        # Add CRS to the admin_gdf
        admin_gdf.crs = 'epsg:4326'

    # Return a geometry object with the union of all the polygons in gdf 
    if (join): 
        admin_geom = admin_gdf.unary_union
        # Create a geo data frame from admin_geom, with additional columns NAME_1 with values set to country_name and NAME_2 with values set to 
        admin_gdf = gpd.GeoDataFrame(geometry=[admin_geom], crs='epsg:4326')

    # Calculate the area of the polygon in km2
    admin_area = admin_gdf.to_crs('epsg:3395').area / 10**6
    
    print('Total area of this polygon is {} km2'.format(round(admin_area, 2)))

    return admin_gdf

###########################
# BBOX SPLITTER FUNCTIONS
###########################

def split_bbox_into_blocks(bbox, block_size):
    min_lon, min_lat, max_lon, max_lat = bbox
    blocks = []
    block_id = 1
    
    # Calculate the number of rows and columns
    # Calculate the distance in kilometers between the min and max longitude
    
    distance_x = great_circle((min_lat, min_lon), (min_lat, max_lon)).kilometers
    distance_y = great_circle((min_lat, min_lon), (max_lat, min_lon)).kilometers
    print('East-west distance: {} km North-South distance {}'.format(distance_x, distance_y))
    num_cols = math.ceil(distance_x * 1000 / block_size)  # Convert block_size to meters
    num_rows = math.ceil(distance_y * 1000 / block_size)  # Convert block_size to meters

    print('East-west N blocks: {} North-south N blocks: {}'.format(num_rows, num_cols))

    # Calculate the latitude and longitude step sizes
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    print('Lon step (deg): {} Lat step (deg): {}'.format(lon_step, lat_step))

    # Iterate through rows and columns to create blocks
    for i in range(num_rows):
        for j in range(num_cols):
            block_min_lon = min_lon + j * lon_step
            block_max_lon = block_min_lon + lon_step
            block_min_lat = min_lat + i * lat_step
            block_max_lat = block_min_lat + lat_step
            
            # Create a polygon for the block
            block_polygon = Polygon([
                (block_min_lon, block_min_lat),
                (block_max_lon, block_min_lat),
                (block_max_lon, block_max_lat),
                (block_min_lon, block_max_lat),
                (block_min_lon, block_min_lat)
            ])
            
            # Add the block to the list of blocks with a unique ID
            block_feature = geojson.Feature(
                geometry=block_polygon.__geo_interface__,
                properties={"id": block_id}
            )
            blocks.append(block_feature)
            block_id += 1

    return geojson.FeatureCollection(blocks)

def generate_sampling_grid(bbox, boundary_gdf, block_size = 150):

    blkgrid_geoj = split_bbox_into_blocks(bbox, block_size)

    # CONVERTING GEOJSON TO GEOPANDAS DATA FRAME 
    blkgrid_gdf = gpd.GeoDataFrame.from_features(blkgrid_geoj['features'], crs='epsg:4326')

    if (boundary_gdf.empty == False): 
        # Print the total area of the boundary_gdf in km2
        boundary_area = boundary_gdf.to_crs('epsg:3395').area.sum() / 10**6
        print('Total area of boundary polygon: {} km2'.format(round(boundary_area, 2)))
        # Check if any part of the polygons in block_polys_gdf intersect with admin_gdf
        blkgrid_gdf['keep'] = blkgrid_gdf.geometry.intersects(boundary_gdf.unary_union)
        # Filter the polygons to keep only those that intersected with admin_gdf and drop the keep column
        blkgrid_gdf = blkgrid_gdf[blkgrid_gdf['keep']].drop(columns=['keep'])


    # Print the total area of the block polygons
    blkgrid_area = blkgrid_gdf.to_crs('epsg:3395').area.sum() / 10**6
    print('Total area of block polygons: {} km2'.format(round(blkgrid_area, 2)))
    # Print total number of block polygons in block_polys_in_admin
    print('Total number of blocks in sampling grid: ', len(blkgrid_gdf))

    return blkgrid_gdf

###########################
# EARTH ENGINE FUNCTIONS
###########################

"""
The following functions are used to build a cloud mask and build a cloudless mosaic from a collection of images
Source: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
"""

def get_s2_sr_cld_col(aoi, past_n_days, cloud_filter = 60):
    '''
    Define a function to filter the SR and s2cloudless collections according to area of interest and date parameters, 
    then join them on the system:index property. The result is a copy of the SR collection where each image has a new 's2cloudless' 
    property whose value is the corresponding s2cloudless image
    '''
    region = ee.Geometry.Rectangle(coords = aoi)
    end_date = ee.Date(str(date.today()))
    start_date = end_date.advance(past_n_days, 'day')

    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(region)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img, cld_prb_thresh=0.7):
    '''
    Define a function to add the s2cloudless probability layer and derived cloud mask as bands to an S2 SR image input.
    '''

    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cld_prb_thresh).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img, nir_drk_thresh=0.15, cld_prj_dist=1):
    '''
    Define a function to add dark pixels, cloud projection, and identified shadows as bands to an S2 SR image input. 
    Note that the image input needs to be the result of the above add_cloud_bands function because it relies on knowing which pixels are considered cloudy ('clouds' band).
    '''

    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(nir_drk_thresh*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, cld_prj_dist*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img, buffer=100):
    '''
    Final cloud shadow mask 
    '''

    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(buffer*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def add_cld_mask(img, buffer=100):
    '''
    Simplified cloud shadow mask 
    '''

    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld = img_cloud.select('clouds').gt(0).rename('cloudmask')

    # Add the final cloud-shadow mask to the image.
    return img_cloud.addBands(is_cld)

def apply_cld_shdw_mask(img):
    '''
    Cloud mask application function
    '''
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

def get_cloudless_collection(bbox, collection, cloudiness_threshold_pct=10, past_n_days = -365, clip = True):
    # Create earth engine geometry using the bounding box supplied in region
    region = ee.Geometry.Rectangle(coords = bbox)
    # Define the date range and cloudiness threshold
    today = ee.Date(str(date.today()))
    retrieval_window = today.advance(past_n_days, 'day')

    # Load images collection
    col = (ee.ImageCollection(collection)
                           .filterBounds(region)
                           .filterDate(retrieval_window, today)
                           .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloudiness_threshold_pct))
                           .sort('CLOUDY_PIXEL_PERCENTAGE'))

    # Print the number of images in the collection
    print('Number of images in collection: ', col.size().getInfo())

    mosaic = col.mosaic()

    if clip: 
        mosaic =  mosaic.clip(ee.Geometry.Rectangle(bbox))

    return mosaic

def get_cloudless_image(bbox, collection, cloudiness_threshold_pct=10, past_n_days = -365, clip = True):
    # Create earth engine geometry using the bounding box supplied in region
    region = ee.Geometry.Rectangle(coords = bbox)
    # Define the date range and cloudiness threshold
    today = ee.Date(str(date.today()))
    retrieval_window = today.advance(past_n_days, 'day')

    # Load images collection
    col = (ee.ImageCollection(collection)
                           .filterBounds(region)
                           .filterDate(retrieval_window, today)
                           .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloudiness_threshold_pct))
                           .sort('CLOUDY_PIXEL_PERCENTAGE'))

    # Print the number of images in the collection
    print('Number of images in collection: ', col.size().getInfo())
 
    # Take the least cloudy image
    least_cloudy = col.first()

    # Print the date of the s2_image
    print('Date of least cloudy S2 image: ', ee.Date(least_cloudy.get('system:time_start')).format('YYYY-MM-dd').getInfo())

    if not least_cloudy:
        return None

    # Visualize image using true color bands (this can be modified based on needs)
    #image_rgb = least_cloudy.visualize(bands=['B4', 'B3', 'B2'], min = 0, max = 4000)

    if clip: 
        least_cloudy = least_cloudy.clip(ee.Geometry.Rectangle(bbox))

    return least_cloudy


def get_landcover(bbox, past_n_days = -300, cloudiness_threshold_pct = 0.5, clip = True): 
     # Create earth engine geometry using the bounding box supplied in region
    region = ee.Geometry.Rectangle(coords = bbox)

    today = ee.Date(str(date.today()))
    retrieval_window = today.advance(past_n_days, 'day')

    col_filter = ee.Filter.And(
        ee.Filter.bounds(region),
        ee.Filter.date(retrieval_window, today), 
        ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloudiness_threshold_pct)
        )

    # Load images collection
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)
    s2_col = ee.ImageCollection('COPERNICUS/S2').filter(col_filter).sort('CLOUDY_PIXEL_PERCENTAGE')

    # Join corresponding DW and S2 images (by system:index).
    dw_s2_col = ee.Join.saveFirst('s2_img').apply(
        s2_col,
        dw_col,
        ee.Filter.equals(leftField='system:index', rightField='system:index'),
    )

    # Extract an example DW image and its source S2 image.
    s2_image = ee.Image(dw_s2_col.first())
    dw_image = ee.Image(s2_image.get('s2_img'))

    # Print the date of the s2_image
    print('Date of least cloudy LULC image: ', ee.Date(s2_image.get('system:time_start')).format('YYYY-MM-dd').getInfo())

    if clip: 
        s2_image = s2_image.clip(ee.Geometry.Rectangle(bbox))
        dw_image = dw_image.clip(ee.Geometry.Rectangle(bbox))

    return dw_image, s2_image

def get_landcover_mosaic(bbox, past_n_days = -300, clip = True): 
     # Create earth engine geometry using the bounding box supplied in region
    region = ee.Geometry.Rectangle(coords = bbox)

    today = ee.Date(str(date.today()))
    retrieval_window = today.advance(past_n_days, 'day')

    col_filter = ee.Filter.And(
        ee.Filter.bounds(region),
        ee.Filter.date(retrieval_window, today)
        )

    # Load images collection, mosaic
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)
    # select all bands except the label band

    dw_bands = dw_col.select(['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'snow_and_ice']).median()
    dw_label = dw_col.select(['label']).mode()

    if clip: 
        dw_bands = dw_bands.clip(ee.Geometry.Rectangle(bbox))
        dw_label = dw_label.clip(ee.Geometry.Rectangle(bbox))

    return dw_bands, dw_label

def get_built_areas(image, threshold = 0.5):
    built_area = image.select('built').gt(threshold).selfMask()
    return built_area

def get_viirs_composite(bbox, past_n_days = -300, clip = True):
    region = ee.Geometry.Rectangle(coords = bbox)

    today = ee.Date(str(date.today()))
    retrieval_window = today.advance(past_n_days, 'day')

    col_filter = ee.Filter.And(
        ee.Filter.bounds(region),
        ee.Filter.date(retrieval_window, today)
        )
    
    viirs_col = ee.ImageCollection('NOAA/VIIRS/001/VNP46A2').filter(col_filter)
    viirs_image = viirs_col.select('DNB_BRDF_Corrected_NTL').median()

    if clip: 
        viirs_image = viirs_image.clip(ee.Geometry.Rectangle(bbox))

    return viirs_image

def get_ee_feature(geom):
    x,y = geom.exterior.coords.xy
    coords = np.dstack((x,y)).tolist()
    g = ee.Geometry.Polygon(coords)
    return ee.Feature(g)


def get_adjacent_blocks(block_id: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function that takes a block_id and returns all adjacent blocks (as a Geodatframe) to that block
    (i.e., blocks that share an edge with the block) using a GeoDataFrame.
    """
    # Get the geometry of the block
    block_geom = gdf.loc[gdf['block_id'] == block_id, 'geometry'].values[0]
    
    # Select blocks that are adjacent by checking if they touch the target block's geometry
    # This ensures blocks share an edge (or point, in some definitions) with the target block
    adjacent_blocks = gdf[gdf['geometry'].touches(block_geom)]
    
    # Remove the target block from the list of adjacent blocks if it's included
    adjacent_blocks = adjacent_blocks[adjacent_blocks['block_id'] != block_id]
    
    return adjacent_blocks

def get_new_adjacent_blocks(edge_block_id: str, network_gdf: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to find new adjacent blocks to a given 'edge block' in the network,
    excluding blocks already included in the network.
    
    Parameters:
    - edge_block_id: str, the ID of the block on the outer edge of the network.
    - network_gdf: gpd.GeoDataFrame, the current GeoDataFrame representing the network of blocks.
    - gdf: gpd.GeoDataFrame, the original GeoDataFrame of all blocks.
    
    Returns:
    - gpd.GeoDataFrame of new adjacent blocks not already part of the network.
    """
    # Find all adjacent blocks to the edge block
    adjacent_blocks = get_adjacent_blocks(edge_block_id, gdf)
    
    # Get the block IDs already in the network
    network_block_ids = network_gdf['block_id'].unique()
    
    # Filter out blocks that are already in the network
    new_adjacent_blocks = adjacent_blocks[~adjacent_blocks['block_id'].isin(network_block_ids)]
    
    return new_adjacent_blocks

def expand_network(initial_block_id: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dynamically expands a network of blocks starting from an initial block ID, allowing
    the user to select edge blocks in each round until no more selections are made.
    
    Parameters:
    - initial_block_id: str - The ID of the first block in the network.
    - gdf: gpd.GeoDataFrame - GeoDataFrame of all blocks with their geometries.
    
    Returns:
    - network_gdf: gpd.GeoDataFrame - The final network of blocks as a GeoDataFrame,
      including a 'round' column indicating the round each block was added.
    """

    # Initialize network with the first block and set 'round' column to 0

    network_gdf = gdf[gdf['block_id'] == initial_block_id].copy()
    network_gdf['round'] = 0  # Initial block is added in round 0

    current_round = 1
    edge_blocks = [initial_block_id]  # Initialize edge blocks list with the first block

    while True:
        new_edge_blocks = []

        for block_id in edge_blocks:
            adjacent_blocks = get_new_adjacent_blocks(block_id, network_gdf, gdf)
            
            if not adjacent_blocks.empty:
                adjacent_blocks['round'] = current_round
                network_gdf = gpd.GeoDataFrame(pd.concat([network_gdf, adjacent_blocks], ignore_index=True))
                new_edge_blocks.extend(adjacent_blocks['block_id'].tolist())

        if not new_edge_blocks:
            print("No more new edge blocks to add. Network expansion complete.")
            break

        print("New edge blocks for round {}: ".format(current_round), new_edge_blocks)
        selected_blocks_input = input("Enter block IDs to add to the network, separated by commas, or press enter to finish: ")
        
        if not selected_blocks_input.strip():
            # Break the loop if input is empty, indicating the user wants to exit
            break

        selected_blocks = [block.strip() for block in selected_blocks_input.split(',') if block.strip()]
        edge_blocks = [block for block in selected_blocks if block in new_edge_blocks]

        if not edge_blocks:
            print("No valid new edge blocks selected. Exiting.")
            break

        current_round += 1

    print("Final network consists of block IDs: ", network_gdf['block_id'].tolist())
    print("Round information: \n", network_gdf[['block_id', 'round']])
    print("Preparing to return network_gdf")  # Debug print
    return network_gdf

# Define a function that plots a block and its adjacent blocks on the same plot
def plot_block_and_adjacent(block_id: str, gdf: gpd.GeoDataFrame) -> None:
    """
    Function that plots a block and its adjacent blocks on the same plot.
    """

    # Get the adjacent blocks
    adjacent_blocks = get_adjacent_blocks(block_id, gdf)
    
    # Plot the block and adjacent blocks
    fig, ax = plt.subplots(figsize=(10, 10))
  
    # Plot the block
    gdf[gdf['block_id'] == block_id].boundary.plot(ax=ax, color="green", linewidth = 1.75)
    # Plot the adjacent blocks
    adjacent_blocks.boundary.plot(ax=ax, color="orange", linewidth = 0.75)
    # Show the labels of the selected block and the adjacent blocks 
    for idx, row in gdf[gdf['block_id'] == block_id].iterrows():
        ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['block_id'], fontsize=8, ha='center')
    for idx, row in adjacent_blocks.iterrows():
        ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['block_id'], fontsize=8, ha='center')
    plt.title(f"Block {block_id} and its adjacent blocks")
    plt.show()