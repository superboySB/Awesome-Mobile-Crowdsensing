import numpy as np
import movingpandas
import pandas as pd
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


def traj_to_timestamped_geojson(index, trajectory: movingpandas.Trajectory, car_num, drone_num, color,
                                connect_line=False, fix_target=False, is_emergency=False):
    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index
    point_gdf["previous_time"] = point_gdf["time"].shift()
    point_gdf.loc[point_gdf["time"].iloc[0], "previous_geometry"] = point_gdf['geometry'].iloc[0]
    point_gdf.loc[point_gdf["time"].iloc[0], "previous_time"] = point_gdf['time'].iloc[0]
    # pd.Timedelta(days=0, hours=0, minutes=15))
    features = []
    # for Point in GeoJSON type
    for i, row in enumerate(point_gdf.itertuples(index=False)):
        current_point_coordinates = [row.geometry.xy[0][0], row.geometry.xy[1][0]]
        previous_point_coordinates = [row.previous_geometry.xy[0][0], row.previous_geometry.xy[1][0]]
        current_time = [row.time.isoformat()]
        previous_time = [row.previous_time.isoformat()]

        if 0 > row.id >= (-car_num):
            radius = 8  # 125(5 units)
            opacity = 0.05
            popup_html = f'<h4> (Car) Agent {car_num + drone_num - index - 1}</h4>' + \
                         f"<p style='font-size:14px;'>Timestamp: {i}</p>" + \
                         f"<p style='font-size:14px;'>reward: {row.reward:.4f} </p>" + \
                         f"<p style='font-size:14px;'>energy: {row.energy}J </p>"

            # f'<p>raw coord: {current_point_coordinates}</p>' + \
            # f'<p>grid coord: ({row.x},{row.y})</p>' + \
            # f'<p>dist coord: ({row.x_distance}m, {row.y_distance}m)</p>' + \
        elif row.id < (-car_num):
            radius = 6  # 125(5 units)
            opacity = 1
            popup_html = f'<h4> (Drone) Agent {car_num + drone_num - index - 1}</h4>' + \
                         f"<p style='font-size:14px;'>Timestamp: {i}</p>" + \
                         f"<p style='font-size:14px;'>reward: {row.reward:.4f} </p>" + \
                         f"<p style='font-size:14px;'>energy: {row.energy}J </p>"

            # f'<p>raw coord: {current_point_coordinates}</p>' + \
            # f'<p>grid coord: ({row.x},{row.y})</p>' + \
            # f'<p>dist coord: ({row.x_distance}m, {row.y_distance}m)</p>' + \
        else:
            if is_emergency:
                radius = 32
                opacity = min(1, row.creation_time / row.episode_length)
            else:
                radius = 6
                opacity = 1
            popup_html = f'<h4> PoI {int(row.id)}</h4>' + \
                         f"<p style='font-size:14px;'>grid coord: ({row.x},{row.y})</p>" + \
                         f"<p style='font-size:14px;'>Creation Time: {row.creation_time} </p>" + \
                         f"<p style='font-size:14px;'>Delay: {int(row.aoi)} </p>" + \
                         f"<p style='font-size:14px;'>Allocation: {row.allocation}</p>"

            # f"<p >raw coord: {current_point_coordinates}</p>" + \
            # f'<p>dist coord: ({row.x_distance}m, {row.y_distance}m)</p>' + \

        if connect_line:
            feature_dict = create_linestring_feature([previous_point_coordinates, current_point_coordinates],
                                                     [previous_time[0], current_time[0]],
                                                     color=color, caption=popup_html)
        else:
            feature_dict = create_point_feature(color, current_point_coordinates, current_time, opacity,
                                                popup_html, radius)
        features.append(feature_dict)
        if fix_target and not connect_line:
            break
    return features


def create_point_feature(color, current_point_coordinates, current_time, opacity, popup_html, radius):
    feature_dict = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": current_point_coordinates,
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
    return feature_dict


def create_linestring_feature(coordinates, dates, color, caption=None):
    feature_dict = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {
            "times": dates,
            "icon": 'circle',  # point
            "iconstyle": {
                'fillColor': color,
                'fillOpacity': 1,  # 透明度
                'stroke': 'true',
                'radius': 5,
                'weight': 1,
            },

            "style": {  # line
                "color": color,
            },
            "code": 11,
        },
    }
    if caption is not None:
        feature_dict["properties"]["popup"] = caption
    return feature_dict

# if __name__ == "__main__":
#     print(judge_collision(new_robot_px=6505, new_robot_py=5130,
#                           old_robot_px=6925, old_robot_py=5130))
def map_values_to_levels(arr: np.ndarray, num_levels=8):
    # Create an array of equally spaced levels from 0 to 1
    levels = np.linspace(0, 1, num_levels + 1)

    # Map each value in the input array to the nearest level
    mapped_values = np.interp(arr, levels[:-1], levels[1:])

    return mapped_values


def generate_hotspot_circle(size, radius):
    center = (size - 1) / 2  # Center of the array

    # Create an empty array
    hotspot_array = np.zeros((size, size))

    # Calculate the Gaussian-like values based on distance from the center
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance <= radius:
                # Use a Gaussian-like function to assign values
                hotspot_array[i, j] = np.exp(-distance ** 2 / (2 * (radius / 2) ** 2))

    return hotspot_array
