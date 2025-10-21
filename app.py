import time
import concurrent.futures
import geopandas as gpd
import numpy as np
from flask import Flask, request, jsonify
from shapely.geometry import Polygon
from shapely.affinity import translate, scale, rotate
from scipy.optimize import differential_evolution, minimize

class ShapeMatcher:
    def __init__(self, reference_shape: Polygon, test_shape: Polygon):
        self.reference_norm = reference_shape
        self.test_norm = self._preprocess(test_shape)

    def _preprocess(self, shape: Polygon) -> Polygon:
        centroid = shape.centroid
        centered_shape = translate(shape, xoff=-centroid.x, yoff=-centroid.y)
        if centered_shape.area == 0: return centered_shape
        scale_factor = 1.0 / np.sqrt(centered_shape.area)
        normalized = scale(centered_shape, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        return normalized.simplify(0.01)

    def _apply_transform(self, shape: Polygon, params: tuple) -> Polygon:
        s, theta, tx, ty = params
        transformed = scale(shape, xfact=s, yfact=s, origin=(0, 0))
        transformed = rotate(transformed, theta, origin=(0, 0))
        return translate(transformed, xoff=tx, yoff=ty)

    def _similarity_iou(self, shape_a: Polygon, shape_b: Polygon) -> float:
        intersection_area = shape_a.intersection(shape_b).area
        union_area = shape_a.union(shape_b).area
        return intersection_area / union_area if union_area != 0 else 0.0

    def _objective_function(self, params: tuple) -> float:
        test_transformed = self._apply_transform(self.test_norm, params)
        similarity = self._similarity_iou(self.reference_norm, test_transformed)
        return 1.0 - similarity

    def find_best_match(self) -> dict:
        bounds = [(0.5, 2.0), (-180, 180), (-1.0, 1.0), (-1.0, 1.0)]
        coarse_result = differential_evolution(
            self._objective_function, 
            bounds, 
            maxiter=30,
            popsize=12,
            tol=0.03,
            seed=42
        )
        fine_result = minimize(
            self._objective_function, 
            x0=coarse_result.x, 
            method='Powell', 
            options={'xtol': 1e-4, 'ftol': 1e-4}
        )
        return {"similarity_score": 1.0 - fine_result.fun}

def preprocess_country_shape(geometry: Polygon) -> Polygon:
    main_polygon = geometry
    if geometry.geom_type == 'MultiPolygon':
        main_polygon = max(geometry.geoms, key=lambda p: p.area)
    centroid = main_polygon.centroid
    centered = translate(main_polygon, xoff=-centroid.x, yoff=-centroid.y)
    if centered.area == 0: return centered
    scale_factor = 1.0 / np.sqrt(centered.area)
    normalized = scale(centered, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return normalized.simplify(0.01)

print("Loading country shapefile...")
world_df = gpd.read_file("countries/ne_10m_admin_0_countries.shp")
world_df = world_df[world_df['TYPE'] == 'Sovereign country']
world_df = world_df[world_df['NAME'] != 'Antarctica']
COUNTRY_NAME_COLUMN = 'NAME'
countries_df = world_df[[COUNTRY_NAME_COLUMN, 'geometry']].copy()
print("Projecting geometries...")
countries_df = countries_df.to_crs("+proj=moll")

print("Preprocessing all country shapes...")
start_time = time.time()
PREPROCESSED_COUNTRIES = [
    {"name": row[COUNTRY_NAME_COLUMN], "norm_geom": preprocess_country_shape(row['geometry'])}
    for _, row in countries_df.iterrows()
]
print(f"--- Preprocessing complete in {time.time() - start_time:.2f} seconds ---")

executor = concurrent.futures.ThreadPoolExecutor()

def worker_process(country_data: dict, user_polygon: Polygon) -> dict:
    country_name = country_data["name"]
    normalized_country_geom = country_data["norm_geom"]
    matcher = ShapeMatcher(reference_shape=normalized_country_geom, test_shape=user_polygon)
    result = matcher.find_best_match()
    return {"country": country_name, "score": result['similarity_score']}

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    start_time = time.time()
    data = request.get_json()
    if not data or 'shape_coords' not in data: return jsonify({"error": "Missing 'shape_coords'"}), 400
    try:
        user_poly = Polygon(data['shape_coords'])
        if not user_poly.is_valid: return jsonify({"error": "Invalid polygon."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not create polygon: {e}"}), 400
    futures = [executor.submit(worker_process, country_data, user_poly) for country_data in PREPROCESSED_COUNTRIES]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    sorted_results = sorted(results, key=lambda k: k['score'], reverse=True)
    print(f"\n--- Full request processed in {time.time() - start_time:.2f} seconds ---")
    return jsonify(sorted_results[:10])

@app.route('/countries', methods=['GET'])
def get_countries():
    country_names = [country['name'] for country in PREPROCESSED_COUNTRIES]
    return jsonify(sorted(country_names))

@app.route('/calculate_single', methods=['POST'])
def calculate_single():
    start_time = time.time()
    data = request.get_json()
    
    if not data or 'shape_coords' not in data or 'country_name' not in data:
        return jsonify({"error": "Request must include 'shape_coords' and 'country_name'"}), 400

    country_name_to_find = data['country_name']

    try:
        user_poly = Polygon(data['shape_coords'])
        if not user_poly.is_valid:
            return jsonify({"error": "Invalid polygon coordinates."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not create polygon: {e}"}), 400

    target_country_data = next((country for country in PREPROCESSED_COUNTRIES if country['name'].lower() == country_name_to_find.lower()), None)

    if not target_country_data:
        return jsonify({"error": f"Country '{country_name_to_find}' not found."}), 404

    print(f"Calculating for single country: {target_country_data['name']}...")
    result = worker_process(target_country_data, user_poly)
    
    print(f"--- Single country request processed in {time.time() - start_time:.2f} seconds ---")
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
