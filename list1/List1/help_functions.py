import graph
import csv
from datetime import datetime


def time_to_priority(time_str):
    if time_str is None:
        return None
    hour, minute, second = map(int, time_str.split(":"))
    seconds_since_midnight = hour * 3600 + minute * 60 + second
    return seconds_since_midnight


def load_graph_from_csv(filename):
    new_graph: graph.Graph = graph.Graph()

    with open(filename, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            edge_id = row['id']
            company = row['company']
            line = row['line']
            departure_time = row['departure_time']
            arrival_time = row['arrival_time']
            start_stop = row['start_stop']
            end_stop = row['end_stop']
            start_lat = float(row['start_stop_lat'])
            start_lon = float(row['start_stop_lon'])
            end_lat = float(row['end_stop_lat'])
            end_lon = float(row['end_stop_lon'])

            new_graph.add_edge(start_stop, end_stop, edge_id, company, line, departure_time, arrival_time, start_lat,
                               start_lon, end_lat, end_lon)
        print(len(new_graph.edge_list))
    return new_graph


def convert_time_and_compare(source_time, target_time):
    target_time = datetime.strptime(target_time, "%H:%M:%S").time()
    source_time = datetime.strptime(source_time, "%H:%M:%S").time()
    return source_time <= target_time
