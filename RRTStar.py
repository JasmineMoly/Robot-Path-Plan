import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None
        self.cost = 0


class RRTStar3D:
    def __init__(self, st, gl, ot, rand_area, step_size, max_iter, search_radius):
        self.start = Node(st)
        self.goal = Node(gl)
        self.obstacle_list = ot
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = []

    def generate_random_point(self):
        point = np.random.uniform(low=self.min_rand, high=self.max_rand, size=3)
        return point

    def nearest_node(self, point):
        distances = [np.linalg.norm(np.array(node.point) - np.array(point)) for node in self.node_list]
        return np.argmin(distances)

    def steer(self, from_node, to_node):
        direction = np.array(to_node.point) - np.array(from_node.point)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            new_node_point = to_node.point
        else:
            unit_direction = direction / distance
            new_node_point = from_node.point + self.step_size * unit_direction
        new_node = Node(new_node_point)
        new_node.parent = from_node
        return new_node

    def collision_free(self, point):
        for obstacle in self.obstacle_list:
            obstacle_center = np.array(obstacle[0])
            obstacle_radius = obstacle[1]
            distance = np.linalg.norm(obstacle_center - point)
            if distance <= obstacle_radius:
                return False
        return True

    def near_nodes(self, node):
        distances = [np.linalg.norm(np.array(node.point) - np.array(other_node.point)) for other_node in self.node_list]
        near_nodes_idx = [idx for idx, distance in enumerate(distances) if self.search_radius >= distance > 0]
        return near_nodes_idx

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return False
        costs = []
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_free(near_node.point) and np.linalg.norm(
                    np.array(new_node.point) - np.array(near_node.point)) <= self.step_size:
                cost = near_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                costs.append(cost)
            else:
                costs.append(float('inf'))
        min_cost_idx = near_nodes[np.argmin(costs)]
        new_node.cost = self.node_list[min_cost_idx].cost + np.linalg.norm(
            np.array(new_node.point) - np.array(self.node_list[min_cost_idx].point))
        new_node.parent = self.node_list[min_cost_idx]
        return True

    def rewire(self, new_node, near_nodes):
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_free(new_node.point) and np.linalg.norm(
                    np.array(new_node.point) - np.array(near_node.point)) <= self.step_size:
                new_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                if near_node.cost > new_cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    @staticmethod
    def trace_path(goal_node, goal_point):
        final_path = [goal_point]
        current_node = goal_node
        while current_node is not None:
            final_path.append(current_node.point)
            current_node = current_node.parent
        return np.array(final_path[::-1])

    def plan(self):
        self.node_list.append(self.start)
        for _ in range(self.max_iter):
            if np.random.rand() < 0.1:
                random_point = self.goal.point
            else:
                random_point = self.generate_random_point()
            nearest_node_idx = self.nearest_node(random_point)
            nearest_node = self.node_list[nearest_node_idx]
            new_node = self.steer(nearest_node, Node(random_point))
            if self.collision_free(new_node.point):
                near_nodes_idx = self.near_nodes(new_node)
                if self.choose_parent(new_node, near_nodes_idx):
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_nodes_idx)

        goal_node = None
        for node in self.node_list:
            if np.linalg.norm(np.array(node.point) - np.array(self.goal.point)) <= self.step_size:
                if self.collision_free(node.point):
                    goal_node = node
                    break

        if goal_node is None:
            return None
        print(goal_node.point)
        final_path = self.trace_path(goal_node, self.goal.point)
        return final_path

    def draw_path(self, final_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.start.point[0], self.start.point[1], self.start.point[2], c='green', marker='o', label='Start')
        ax.scatter(self.goal.point[0], self.goal.point[1], self.goal.point[2], c='red', marker='o', label='Goal')
        for obstacle in self.obstacle_list:
            # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            # x = obstacle[0][0] + obstacle[1] * np.cos(u) * np.sin(v)
            # y = obstacle[0][1] + obstacle[1] * np.sin(u) * np.sin(v)
            # z = obstacle[0][2] + obstacle[1] * np.cos(v)
            # ax.plot_surface(x, y, z, color='y', alpha=0.5)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = obstacle[0][0] + obstacle[1] * np.outer(np.cos(u), np.sin(v))
            y = obstacle[0][1] + obstacle[1] * np.outer(np.sin(u), np.sin(v))
            z = obstacle[0][2] + obstacle[1] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, rstride=4, cstride=4, color='y', linewidth=0, alpha=0.5)
        ax.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], c='blue', label='Path')
        for i in final_path:
            print(i)
        ax.set_xlim(self.min_rand, self.max_rand)
        ax.set_ylim(self.min_rand, self.max_rand)
        ax.set_zlim(self.min_rand, self.max_rand)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    start = [1, 1, 1]
    goal = [7, 10, 6]
    obstacle_list = [([6, 6, 5], 1), ([4, 5, 5], 2), ([4, 11, 2], 2), ([8, 6, 4], 2)]
    rrt_star = RRTStar3D(start, goal, obstacle_list, rand_area=[0, 15], step_size=0.7, max_iter=2000, search_radius=3)
    path = rrt_star.plan()

    if path is None:
        print("No valid path found!")
    else:
        print("Found!")
        rrt_star.draw_path(path)
