from queue import PriorityQueue
import time
from graph import *


def time_to_priority(time_str):
    if time_str is None:
        return None
    hour, minute, second = map(int, time_str.split(":"))
    seconds_since_midnight = hour * 3600 + minute * 60 + second
    return seconds_since_midnight


def heuristic(goal: Node, next_one: Node):
    return 100 * int(100 * (abs(goal.x - next_one.x) + abs(goal.y - next_one.y)))


def a_star_change(graph, start: str, goal: str, start_time):
    frontier = PriorityQueue()
    frontier.put((0, start))

    came_from = dict()
    cost_so_far = dict()
    came_from[start] = set()
    cost_so_far[start] = 0

    while not frontier.empty():
        prio, current = frontier.get()

        if current == goal:
            break

        for next_one_edge in graph.neighbors_lines(current, came_from):
            temp_list = []
            for x in list(came_from[current]):
                temp_list.append(x[0])
            if next_one_edge.line not in temp_list and current != start:
                new_cost_line = cost_so_far[current] + 1000
            else:
                new_cost_line = cost_so_far[current]

            if next_one_edge.end_stop not in cost_so_far or new_cost_line < cost_so_far[next_one_edge.end_stop]:
                cost_so_far[next_one_edge.end_stop] = new_cost_line

                priority = cost_so_far[next_one_edge.end_stop] + heuristic(graph.get_node_by_name(goal),
                                                                           graph.get_node_by_name(
                                                                               next_one_edge.end_stop))
                print(priority)
                frontier.put((priority, next_one_edge.end_stop))
                tup = (next_one_edge.line, next_one_edge.start_stop)
                if isinstance(tup, tuple) and len(tup) == 2:
                    came_from[next_one_edge.end_stop] = {tup}
                else:
                    pass
            elif new_cost_line == cost_so_far[next_one_edge.end_stop]:
                tup = (next_one_edge.line, next_one_edge.start_stop)
                if isinstance(tup, tuple) and len(tup) == 2:
                    came_from[next_one_edge.end_stop].add(tup)

    print('---')
    print(came_from[current])
    print(cost_so_far[current])
    final_list = []
    new_key = goal
    before = None
    line = None
    final_list_line = []
    while True:
        final_list.append(new_key)

        if new_key == start:
            final_list.reverse()
            break
        if new_key == goal:
            before = list(came_from[new_key])[0]
        else:
            temp_dict = dict()
            for x in list(came_from[new_key]):
                temp_dict[x[0]] = x[1]
            if before[0] in temp_dict:
                before = before[0], temp_dict[before[0]]
            else:
                before = temp_dict.popitem()

        new_key = before[1]
        final_list_line.append(before)
    print('xd')
    for y in final_list:
        print(y)
    for y in final_list_line:
        print(y)
    final_list_line.reverse()
    x = graph.get_edges_from_path_time_with_lines(start_time, final_list, final_list_line)
    for edge in x:
        print("{:<10} {:<10} {:<30} {:<10} {:<20}".format(edge.line, edge.departure_time, edge.start_stop,
                                                          edge.arrival_time, edge.end_stop))