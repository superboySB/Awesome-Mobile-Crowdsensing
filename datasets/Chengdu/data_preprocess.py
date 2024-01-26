import os

import pandas as pd
import numpy as np
import time
from typing import Optional
from tqdm import tqdm


def construct_dataframe_for_time_larger_than_0(df, new_row):
    list_of_rows = []
    for time_index in range(int(new_row.time)):
        list_of_rows.append({
            'vehicle_id': new_row.vehicle_id,
            'time': time_index,
            'longitude': new_row.longitude,
            'latitude': new_row.latitude}
        )
    df = pd.concat([df, pd.DataFrame(list_of_rows)], axis=0, ignore_index=True)
    return df


def trans_form_of_lat(lng: float, lat: float):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
            np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
            np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
            np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def trans_form_of_lon(lng: float, lat: float):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
            np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
            np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 *
            np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def get_distance(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """ return the distance between two points in meters """
    lng1, lat1, lng2, lat2 = map(np.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    d_lon = lng2 - lng1
    d_lat = lat2 - lat1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance * 1000


def generate_rows(old_row, new_row, add_number, no_long_lat=False):
    if no_long_lat:
        add_longitude = 1
        add_latitude = 1
    else:
        add_longitude = (new_row.x - old_row.x) / add_number
        add_latitude = (new_row.y - old_row.y) / add_number
    list_of_rows = []
    for time_index in range(add_number):
        list_of_rows.append({
            'vehicle_id': old_row.vehicle_id,
            'time': int(old_row.time) + time_index + 1,
            'longitude': old_row.longitude,
            'latitude': old_row.latitude,
            'x': float(old_row.x) + (time_index + 1) * add_longitude,
            'y': float(old_row.y) + (time_index + 1) * add_latitude
        })
    return list_of_rows


def get_longitude_and_latitude_max(longitude_min, latitude_min, map_width) -> tuple:
    longitude_max = longitude_min
    latitude_max = latitude_min
    precision = 5 * 1e-1
    """
    += 1e-2 add 1467 meters
    += 1e-3 add 147 meters
    += 1e-4 add 15 meters
    += 1e-5 add 1 meter
    += 1e-6 add 0.25 meters
    """
    length = np.sqrt(2) * map_width
    while True:
        distance = get_distance(longitude_min, latitude_min, longitude_max, latitude_max)
        if np.fabs(distance - length) < precision:
            break
        if np.fabs(distance - length) > 2000.0:
            longitude_max += 1e-2
            latitude_max += 1e-2
        if 150.0 < np.fabs(distance - length) <= 2000.0:
            longitude_max += 1e-3
            latitude_max += 1e-3
        if 15.0 < np.fabs(distance - length) <= 150.0:
            longitude_max += 1e-4
            latitude_max += 1e-4
        if np.fabs(distance - length) > 1.0 and np.fabs(distance - length) <= 15.0:
            longitude_max += 1e-5
            latitude_max += 1e-5
        if np.fabs(distance - length) <= 1.0:
            longitude_max += 1e-6
            latitude_max += 1e-6
    return longitude_max, latitude_max


def gcj02_to_wgs84(lng: float, lat: float):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323

    d_lat = trans_form_of_lat(lng - 105.0, lat - 35.0)
    d_lng = trans_form_of_lon(lng - 105.0, lat - 35.0)

    rad_lat = lat / 180.0 * np.pi
    magic = np.sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = np.sqrt(magic)

    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * np.pi)
    d_lng = (d_lng * 180.0) / (a / sqrt_magic * np.cos(rad_lat) * np.pi)
    mg_lat = lat + d_lat
    mg_lng = lng + d_lng
    return [lng * 2 - mg_lng, lat * 2 - mg_lat]


class VehicleTrajectoriesProcessor(object):
    def __init__(
            self,
            file_name: str,
            longitude_min: float,
            latitude_min: float,
            map_width: float,
            time_start: str,
            time_end: str,
            out_file: str,
            output_analysis: Optional[bool] = False,
    ) -> None:
        """The constructor of the class."""
        """
        Args:
            file_name: the name of the file to be processed. 
                e.g., '/CSV/gps_20161116', source: Didi chuxing gaia open dataset initiative
            longitude_min: the minimum longitude of the bounding box. e.g., 104.04565967220308
            latitude_min: the minimum latitude of the bounding box. e.g., 30.654605745741608
            map_width: the width of the bounding box. e.g., 500 (meters)
            time_start: the start time. e.g., '2016-11-16 08:00:00'
            time_end: the end time. e.g., '2016-11-16 08:05:00'
            out_file: the name of the output file.  e.g., '/CSV/gps_20161116_processed.csv'
        """
        self._file_name = file_name
        self._longitude_min, self._latitude_min = gcj02_to_wgs84(longitude_min, latitude_min)
        self._map_width = map_width
        self._time_start = time_start
        self._time_end = time_end
        self._out_file = out_file
        self._output_analysis: bool = output_analysis
        time_style = "%Y-%m-%d %H:%M:%S"
        time_start_array = time.strptime(self._time_start, time_style)
        time_end_array = time.strptime(self._time_end, time_style)
        self._time_start_int = int(time.mktime(time_start_array))
        self._time_end_int = int(time.mktime(time_end_array))

        self._longitude_max, self._latitude_max = get_longitude_and_latitude_max(
            self._longitude_min, self._latitude_min, self._map_width)

        self.process(
            map_width=self._map_width,
            longitude_min=self._longitude_min,
            latitude_min=self._latitude_min,
            out_file=self._out_file
        )

    def process_row(self, df, old_row, new_row):
        if old_row.vehicle_id == new_row.vehicle_id:
            add_number = int(new_row.time) - int(old_row.time) - 1
            if add_number > 0:
                list_of_rows = generate_rows(old_row, new_row, add_number)
                df = pd.concat([df, pd.DataFrame(list_of_rows)], axis=0, ignore_index=True)
        else:
            if int(old_row.time) < self._time_end_int - self._time_start_int:
                list_of_rows = generate_rows(old_row, old_row,
                                             int(self._time_end_int - self._time_start_int) - int(old_row.time) - 1,
                                             no_long_lat=True)
                df = pd.concat([df, pd.DataFrame(list_of_rows)], axis=0, ignore_index=True)
            if int(new_row.time) > 0:
                df = construct_dataframe_for_time_larger_than_0(df, new_row)
        return df

    def process(
            self,
            map_width,
            longitude_min,
            latitude_min,
            out_file,
    ) -> None:

        # if files are already cached, skip reading
        if not (os.path.exists(out_file + "_without_fill" + ".csv") and os.path.exists(out_file + ".csv")):
            print("Reading data...")
            df = pd.read_csv(
                self._file_name,
                names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'],
                dtype={'vehicle_id': str, 'order_number': str, 'time': np.int32,
                       'longitude': np.float64, 'latitude': np.float64},
                header=0
            )
            # 经纬度定位
            df.drop(df.columns[[1]], axis=1, inplace=True)
            df.dropna(axis=0)

            longitude_max, latitude_max = get_longitude_and_latitude_max(longitude_min, latitude_min, map_width)
            print("Selecting data...")
            df = df[
                (df['longitude'] > longitude_min) &
                (df['longitude'] < longitude_max) &
                (df['latitude'] > latitude_min) &
                (df['latitude'] < latitude_max) &
                (df['time'] > self._time_start_int) &
                (df['time'] < self._time_end_int)]  # location

            # print number of rows
            print("Selected Completed. Number of rows: ", df.shape[0])
            print("Sorting data...")
            # sorted
            df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

            print("Discretize data...")
            vehicle_number = 0
            old_vehicle_id = None
            new_rows_dict = []
            for index in tqdm(range(len(df))):
                row = dict(df.iloc[index])
                vehicle_id = row['vehicle_id']
                if old_vehicle_id:
                    if vehicle_id == old_vehicle_id:
                        new_row = self.modify_data(latitude_min, longitude_min, row, vehicle_number)
                    else:
                        vehicle_number += 1
                        new_row = self.modify_data(latitude_min, longitude_min, row, vehicle_number)
                else:
                    new_row = self.modify_data(latitude_min, longitude_min, row, vehicle_number)
                old_vehicle_id = vehicle_id
                new_rows_dict.append(new_row)

            original_df = df
            df = pd.DataFrame(new_rows_dict)

            if self._output_analysis:
                df.to_csv(out_file + "_without_fill" + ".csv")

            print("Filling data...")
            old_row = None
            for row in tqdm(df.itertuples(index=False), total=len(df)):
                new_row = row
                if old_row:
                    df = self.process_row(df, old_row, new_row)
                    old_row = new_row
                else:
                    if int(new_row.time) > 0:
                        df = construct_dataframe_for_time_larger_than_0(df, new_row)
                    old_row = new_row
            df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
            df.to_csv(out_file + ".csv")

        if self._output_analysis:
            analyst = VehicleTrajectoriesAnalyst(
                trajectories_file_name=out_file + ".csv",
                trajectories_file_name_with_no_fill=out_file + "_without_fill" + ".csv",
                during_time=self._time_start_int - self._time_start_int,
            )
            analyst.output_characteristics()

    def modify_data(self, latitude_min, longitude_min, row, vehicle_number):
        row['vehicle_id'] = vehicle_number
        longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
        row['timestamp'] = row['time']
        row['time'] -= self._time_start_int
        x = get_distance(longitude_min, latitude_min, longitude, latitude_min)
        y = get_distance(longitude_min, latitude_min, longitude_min, latitude)
        for item in ['x', 'y', 'longitude', 'latitude']:
            row[item] = locals()[item]
        return row

    def get_out_file(self):
        return self._out_file

    def get_longitude_min(self) -> float:
        return self._longitude_min

    def get_longitude_max(self) -> float:
        return self._longitude_max

    def get_latitude_min(self) -> float:
        return self._latitude_min

    def get_latitude_max(self) -> float:
        return self._latitude_max


class VehicleTrajectoriesAnalyst(object):
    def __init__(
            self,
            trajectories_file_name: str,
            trajectories_file_name_with_no_fill: str,
            during_time: int,
    ) -> None:
        """Output the analysis of vehicular trajcetories, including
            average dwell time (s) of vehicles: ADT
            standard deviation of dwell time (s) of vehicles: ADT_STD
            average number of vehicles in each second: ANV
            standard deviation of number of vehicles in each second: ANV_STD
            average speed (m/s) of vehicles: ASV
            standard deviation of speed (m/s) of vehicles: ASV_STD

        Args:
            trajectories_file_name (str): file with processed trajectories
            trajectories_file_name_with_no_fill (str): file with processed trajectories without filling
            during_time (int): e.g., 300 s
        """
        self._trajectories_file_name = trajectories_file_name
        self._trajectories_file_name_with_no_fill = trajectories_file_name_with_no_fill
        self._during_time = during_time

    def output_characteristics(self):

        df = pd.read_csv(self._trajectories_file_name, names=['vehicle_id', 'time', 'x', 'y'], header=0)
        vehicle_ids = df['vehicle_id'].unique()
        number_of_vehicles_in_seconds = np.zeros(self._during_time, dtype=np.int32)
        vehicle_dwell_times = []
        print("Analysing trajectories...")
        for vehicle_id in tqdm(vehicle_ids):
            new_df = df[df['vehicle_id'] == vehicle_id]
            vehicle_dwell_time = 0.0
            for row in new_df.itertuples():
                distance = np.sqrt((row.x - 1500) ** 2 + (row.y - 1500) ** 2)
                if distance <= 1500:
                    vehicle_dwell_time += 1.0
                    number_of_vehicles_in_seconds[int(row.time)] += 1
            vehicle_dwell_times.append(vehicle_dwell_time)

        assert len(vehicle_dwell_times) == len(vehicle_ids)
        print("vehicle_number: ", len(vehicle_ids))

        adt = np.mean(vehicle_dwell_times)
        adt_std = np.std(vehicle_dwell_times)
        anv = np.mean(number_of_vehicles_in_seconds)
        anv_std = np.std(number_of_vehicles_in_seconds)
        print("Average dwell time (s):", adt)
        print("Standard deviation of dwell time (s):", adt_std)
        print("Average number of vehicles in each second:", anv)
        print("Standard deviation of number of vehicles in each second:", anv_std)

        vehicle_speeds = []
        df = pd.read_csv(self._trajectories_file_name_with_no_fill,
                         names=['vehicle_id', 'time', 'x', 'y'], header=0)
        vehicle_ids = df['vehicle_id'].unique()

        for vehicle_id in vehicle_ids:
            vehicle_speed = []
            new_df = df[df['vehicle_id'] == vehicle_id]
            last_time = -1.0
            last_x = 0.0
            last_y = 0.0
            for row in new_df.itertuples():
                if int(last_time) == -1:
                    last_time = row.time
                    last_x = row.x
                    last_y = row.y
                    continue
                distance = np.sqrt((row.x - last_x) ** 2 + (row.y - last_y) ** 2)
                speed = distance / (row.time - last_time)
                if not np.isnan(speed):
                    vehicle_speed.append(speed)
                last_time = time
                last_x = row.x
                last_y = row.y
            if vehicle_speed:
                average_vehicle_speed = np.mean(vehicle_speed)
                vehicle_speeds.append(average_vehicle_speed)

        asv = np.mean(vehicle_speeds)
        asv_std = np.std(vehicle_speeds)
        print("Average speed (m/s):", asv)
        print("Standard deviation of speed (m/s):", asv_std)


if __name__ == "__main__":
    """Vehicle Trajectories Processor related."""
    trajectories_file_name: str = os.path.join("/workspace", "saved_data", 'datasets',
                                               'Chengdu_taxi', 'gps_20161116')
    longitude_min: float = 104.04565967220308
    latitude_min: float = 30.654605745741608
    start_hour = 22
    start_minute = 0
    start_second = 0
    end_hour = 22
    end_minute = 5
    end_second = 0
    padded_start_hour = str(start_hour).zfill(2)
    padded_start_minute = str(start_minute).zfill(2)
    trajectories_time_start: str = ('2016-11-16' + ' ' + padded_start_hour + ':' +
                                    padded_start_minute + ':' + str(start_second).zfill(2))
    padded_end_hour = str(end_hour).zfill(2)
    padded_end_minute = str(end_minute).zfill(2)
    trajectories_time_end: str = '2016-11-16' + ' ' + padded_end_hour + ':' + \
                                 padded_end_minute + ':' + str(end_second).zfill(2)
    trajectories_out_file_name: str = os.path.join("/workspace", "saved_data", 'datasets',
                                                   'Chengdu_taxi',
                                                   f'trajectories_20161116_{padded_start_hour}'
                                                   f'{padded_start_minute}_{padded_end_hour}'
                                                   f'{padded_end_minute}')

    processor = VehicleTrajectoriesProcessor(
        file_name=trajectories_file_name,
        longitude_min=longitude_min,
        latitude_min=latitude_min,
        map_width=2000.0,
        time_start=trajectories_time_start,
        time_end=trajectories_time_end,
        out_file=trajectories_out_file_name,
        output_analysis=True,
    )
