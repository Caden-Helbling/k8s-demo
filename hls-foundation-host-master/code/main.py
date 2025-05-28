import boto3
import gc
import geopandas as gpd
import json
import json
import os
import rasterio
import time
import torch


from app.chain import DialogParser
from app.lib.downloader import Downloader
from app.lib.infer import Infer
from app.lib.post_process import PostProcess

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from huggingface_hub import hf_hub_download

from multiprocessing import Pool, cpu_count

from rasterio.io import MemoryFile
from rasterio.merge import merge

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from shapely.geometry import shape

from skimage.morphology import disk, binary_closing
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL_CONFIGS = {
    'flood': {
        'config': 'sen1floods11_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-sen1floods11',
        'weight': 'sen1floods11_Prithvi_100M.pth',
        'collections': ['HLSS30'],
    },
    'burn_scars': {
        'config': 'burn_scars_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-burn-scar',
        'weight': 'burn_scars_Prithvi_100M.pth',
        'collections': ['HLSS30', 'HLSL30'],
    },
}

BUCKET_NAME = 'hls-foundation-predictions'


with open('../data/preloaded_events.json') as preloaded:
    PRELOADED_EVENTS = json.load(preloaded)


def update_config(config, model_path):
    with open(config, 'r') as config_file:
        config_details = config_file.read()
        updated_config = config_details.replace('<path to pretrained weights>', model_path)
        updated_config = updated_config.replace('[1, 2, 3, 8, 11, 12]', '[0, 1, 2, 3, 4, 5]')
    with open(config, 'w') as config_file:
        config_file.write(updated_config)


def load_model(model_name):
    repo = MODEL_CONFIGS[model_name]['repo']
    config = hf_hub_download(repo, filename=MODEL_CONFIGS[model_name]['config'])
    model_path = hf_hub_download(repo, filename=MODEL_CONFIGS[model_name]['weight'])
    update_config(config, model_path)
    infer = Infer(config, model_path)
    _ = infer.load_model()
    return infer


MODELS = {model_name: load_model(model_name) for model_name in MODEL_CONFIGS}


def download_files(infer_date, layer, bounding_box):
    downloader = Downloader(infer_date, layer)
    return downloader.download_tiles(bounding_box)


def save_cog(mosaic, profile, transform, filename):
    profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "dtype": 'float32',
            "count": 1,
        }
    )
    with rasterio.open(filename, 'w', **profile) as raster:
        raster.write(mosaic[0], 1)
    output_profile = cog_profiles.get('deflate')
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )
    with MemoryFile() as memory_file:
        cog_translate(
            filename,
            memory_file.name,
            output_profile,
            config=config,
            quiet=True,
            in_memory=True,
        )
        connection = boto3.client('s3')
        connection.upload_fileobj(memory_file, BUCKET_NAME, filename)

    return f"s3://{BUCKET_NAME}/{filename}"


def post_process(detections, transform):
    contours, shape = PostProcess.prepare_contours(detections)
    detections = PostProcess.extract_shapes(detections, contours, transform, shape)
    detections = PostProcess.remove_intersections(detections)
    return PostProcess.convert_to_geojson(detections)


def subset_geojson(geojson, bounding_box):
    geom = [shape(i['geometry']) for i in geojson]
    geom = gpd.GeoDataFrame({'geometry': geom})
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [bounding_box[0], bounding_box[1]],
                [bounding_box[2], bounding_box[1]],
                [bounding_box[2], bounding_box[3]],
                [bounding_box[0], bounding_box[3]],
                [bounding_box[0], bounding_box[1]],
            ]
        ],
    }
    bbox = shape(bbox)
    bbox = gpd.GeoDataFrame({'geometry': [bbox]})
    return json.loads(geom.overlay(bbox, how='intersection').to_json())


def batch(tiles, spacing=40):
    length = len(tiles)
    for tile in range(0, length, spacing):
        yield tiles[tile : min(tile + spacing, length)]


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {'Hello': 'World'}


@app.get('/models')
def list_models():
    response = jsonable_encoder(list(MODEL_CONFIGS.keys()))
    return JSONResponse({'models': response})


@app.post('/search')
async def search(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    parser = DialogParser()
    response = parser.parse(body['query'])
    return_json = {}
    if response.get('error'):
        return_json = {'error': response.get('error'), 'statusCode': 422}
    else:
        return_json = infer(
            response['event_type'], response['date'], response['bounding_box'], background_tasks
        )
        return_json['query'] = {
            'area': response['area'],
            'model_id': response['event_type'],
            'date': response['date'],
            'bounding_box': response['bounding_box'],
        }
    return JSONResponse(content=jsonable_encoder(return_json))


@app.get('/preloaded_events')
def preloaded_events():
    return JSONResponse({'events': PRELOADED_EVENTS})


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def infer_from_model(request: Request, background_tasks: BackgroundTasks):
    instances = await request.json()

    model_id = instances['model_id']
    infer_date = instances['date']
    bounding_box = instances['bounding_box']
    final_geojson = infer(model_id, infer_date, bounding_box, background_tasks)
    return JSONResponse(content=jsonable_encoder(final_geojson))


def infer(model_id, infer_date, bounding_box, background_tasks):
    if model_id not in MODELS:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    infer = MODELS[model_id]
    all_tiles = list()
    geojson_list = list()
    download_infos = list()
    geojson = {'type': 'FeatureCollection', 'features': []}

    for layer in MODEL_CONFIGS[model_id]['collections']:
        tiles = download_files(infer_date, layer, bounding_box)
        for tile in tiles:
            tile_name = tile
            if model_id == 'burn_scars':
                tile_name = tile_name.replace('.tif', '_scaled.tif')
            all_tiles.append(tile_name)

    start_time = time.time()
    mosaic = []
    s3_link = ''
    if all_tiles:
        try:
            torch.cuda.synchronize()
            results = list()
            for tiles in batch(all_tiles):
                results.extend(infer.infer(tiles))
            transforms = list()
            memory_files = list()
            del infer
            torch.cuda.empty_cache()
            for index, tile in enumerate(all_tiles):
                with rasterio.open(tile) as raster:
                    profile = raster.profile
                memfile = MemoryFile()
                profile.update({'count': 1, 'dtype': 'float32'})
                with memfile.open(**profile) as memoryfile:
                    memoryfile.write(results[index], 1)
                memory_files.append(memfile.open())

            mosaic, transform = merge(memory_files)
            mosaic[0] = binary_closing(mosaic[0], disk(6))
            [memfile.close() for memfile in memory_files]
            prediction_filename = f"{start_time}-predictions.tif"

            background_tasks.add_task(
                save_cog, mosaic, raster.meta.copy(), transform, prediction_filename
            )
            s3_link = f"s3://{BUCKET_NAME}/{prediction_filename}"
            geojson = post_process(mosaic[0], transform)

            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
            geojson = subset_geojson(geojson_list, bounding_box)
        except Exception as e:
            print('!!! infer error', infer_date, model_id, bounding_box, e)
            torch.cuda.empty_cache()
        print("!!! Infer Time:", time.time() - start_time)
    gc.collect()

    return {
        'predictions': geojson,
        's3_link': s3_link,
    }
