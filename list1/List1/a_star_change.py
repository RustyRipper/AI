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
    return 10 * int(100 * (abs(goal.x - next_one.x) + abs(goal.y - next_one.y)))


def a_star_change(graph, start, goal, start_time):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    line_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    line_so_far[start] = None

    how_many = 0
    while not frontier.empty():
        prio, current = frontier.get()

        if current == goal:
            break

        current_time = time.strftime("%H:%M:%S", time.gmtime(cost_so_far[current]))
        print(current_time)

        for next_one_edge in graph.neighbors_lines(current):
            how_many += 1
            new_cost_line = 0
            if line_so_far[current] != next_one_edge.line and current != start:
                new_cost_line = 1000000
            else:
                line_so_far[current] = next_one_edge.line
            #new_cost = time_to_priority(graph.cost_time(current, next_one_edge.end_stop, current_time))
            #print(next_one_edge.start_stop, next_one_edge.end_stop, next_one_edge.line, next_one_edge.edge_id)
            if next_one_edge.end_stop not in cost_so_far or new_cost_line < cost_so_far[next_one_edge.end_stop]:
                cost_so_far[next_one_edge.end_stop] = new_cost_line
                line_so_far[next_one_edge.end_stop] = next_one_edge.line
                priority = new_cost_line + heuristic(graph.get_node_by_name(goal),
                                                graph.get_node_by_name(next_one_edge.end_stop))
                print(priority)
                frontier.put((priority, next_one_edge.end_stop))
                came_from[next_one_edge.end_stop] = current
    print(how_many)
    final_list = []
    new_key = goal
    while True:
        final_list.append(new_key)
        if new_key == start:
            final_list.reverse()
            return final_list
        new_key = came_from[new_key]
