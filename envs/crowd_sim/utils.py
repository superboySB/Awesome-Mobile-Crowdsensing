import numpy as np

np.seterr(invalid='ignore')

# from datasets.KAIST.env_config import BaseEnvConfig
# from shapely.geometry import *

# tmp_config = BaseEnvConfig()


# def inPoly(polygon, x, y):
#     pt = (x, y)
#     line = LineString(polygon)
#     point = Point(pt)
#     polygon = Polygon(line)
#     return polygon.contains(point)


# def iscrosses(line1, line2):
#     if LineString(line1).crosses(LineString(line2)):
#         return True
#     return False


# def crossPoly(square, x1, y1, x2, y2):
#     our_line = LineString([[x1, y1], [x2, y2]])
#     line1 = LineString([square[0], square[2]])
#     line2 = LineString([square[1], square[3]])
#     if our_line.crosses(line1) or our_line.crosses(line2):
#         return True
#     else:
#         return False


# def judge_collision(new_robot_px, new_robot_py, old_robot_px, old_robot_py):
#     if tmp_config.env.no_fly_zone is None:
#         return False

#     for square in tmp_config.env.no_fly_zone:
#         if inPoly(square, new_robot_px, new_robot_py):
#             return True
#         if crossPoly(square, new_robot_px, new_robot_py, old_robot_px, old_robot_py):
#             return True
#     return False


def get_border(ur, lf):
    upper_left = [lf[0], ur[1]]
    upper_right = [ur[0], ur[1]]
    lower_right = [ur[0], lf[1]]
    lower_left = [lf[0], lf[1]]

    coordinates = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left
    ]

    geo_json = {"type": "FeatureCollection",
                "properties": {
                    "lower_left": lower_left,
                    "upper_right": upper_right
                },
                "features": []}

    grid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }

    geo_json["features"].append(grid_feature)

    return geo_json


def traj_to_timestamped_geojson(index, trajectory, car_num, drone_num, color):
    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index
    point_gdf["previous_time"] = point_gdf["time"].shift()

    features = []

    # for Point in GeoJSON type
    for _, row in point_gdf.iterrows():
        corrent_point_coordinates = [
            row["geometry"].xy[0][0],
            row["geometry"].xy[1][0]
        ]
        current_time = [row["time"].isoformat()]


        if point_gdf['id'].iloc[0] < 0 and point_gdf['id'].iloc[0] >= (-car_num):
            radius = 4  # 125(5 units)
            opacity = 0.05
            popup_html = f'<h4> (Car) Agent {car_num+drone_num-index-1}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>energy: {row["energy"]}J </p>'
        elif point_gdf['id'].iloc[0]<(-car_num):
            radius = 8  # 125(5 units)
            opacity = 0.05
            popup_html = f'<h4> (Drone) Agent {car_num+drone_num-index-1}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>energy: {row["energy"]}J </p>'
        else:
            radius = 2
            opacity = 1
            popup_html = f'<h4> Human {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>aoi: {int(row["aoi"])} </p>'

        # for Point in GeoJSON type  (Temporally Deprecated)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": corrent_point_coordinates,
                },
                "properties": {
                    "times": current_time,
                    'popup': popup_html,
                    "icon": 'circle',  # point
                    "iconstyle": {
                        'fillColor': color,
                        'fillOpacity': opacity,  # 透明度
                        'stroke': 'true',
                        'radius': radius,
                        'weight': 1,
                    },

                    "style": {  # line
                        "color": color,
                    },
                    "code": 11,

                },
            }
        )
    return features


# if __name__ == "__main__":
#     print(judge_collision(new_robot_px=6505, new_robot_py=5130,
#                           old_robot_px=6925, old_robot_py=5130))
