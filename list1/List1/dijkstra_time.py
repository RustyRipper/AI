from queue import PriorityQueue
import time


def time_to_priority(time_str):
    if time_str is None:
        return None
    hour, minute, second = map(int, time_str.split(":"))
    seconds_since_midnight = hour * 3600 + minute * 60 + second
    return seconds_since_midnight


def dijkstra_time(graph, start, goal, start_time):
    frontier = PriorityQueue()
    frontier.put((time_to_priority(start_time), start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = time_to_priority(start_time)

    how_many = 0
    while not frontier.empty():
        prio, current = frontier.get()

        if current == goal:
            break

        current_time = time.strftime("%H:%M:%S", time.gmtime(cost_so_far[current]))
        print(current_time)
        for next_one in graph.neighbors(current):
            how_many += 1
            new_cost = time_to_priority(graph.cost_time(current, next_one, current_time))
            if new_cost is not None and (next_one not in cost_so_far or new_cost < cost_so_far[next_one]):
                cost_so_far[next_one] = new_cost
                priority = new_cost
                frontier.put((priority, next_one))
                came_from[next_one] = current
    print(how_many)
    print('---')
    final_list = []
    new_key = goal
    while True:
        final_list.append(new_key)
        if new_key == start:
            final_list.reverse()
            break
        new_key = came_from[new_key]

    x = graph.get_edges_from_path_time(start_time, final_list)
    for edge in x:
        print("{:<10} {:<10} {:<30} {:<10} {:<20}".format(edge.line, edge.departure_time, edge.start_stop,
                                                          edge.arrival_time, edge.end_stop))