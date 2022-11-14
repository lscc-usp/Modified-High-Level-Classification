import numpy as np
import cv2

class Node(object):

    def __init__(self, value, isLeaf, left_1, left_2, right_1, right_2, pos, grid_size):

        self.value = value
        self.isLeaf = isLeaf
        self.left_1 = left_1
        self.left_2 = left_2
        self.right_1 = right_1
        self.right_2 = right_2

        self.pos = pos # The relative original coordinates of each piece
        self.grid_size = grid_size

class QuadTreeSolution(object):

    def __init__(self, image_grid, min_grid_size, variance_threshold):

        self.min_grid_size = min_grid_size
        self.image_grid = image_grid
        self.variance_threshold = variance_threshold
        self.histogram = {}  # Store the number of blocks at each level
        self.image_size = len(image_grid)
        self.root_node = self.node_construct(image_grid, pos=(0, 0), grid_size=self.image_size)

    def node_construct(self, grid, pos, grid_size):
        """
        Args:
            grid: numpy Array, (m,m)
            variance_threshold: the variance in a grid, the threshold for segmentation grid.
            pos : [(top left point),(right bottom point)]
            grid_size: len(self.img_size=len(img_grid))

        Returns:
            node: root Node
        """
        grid_mean = np.mean(grid)
        grid_variance = np.var(grid)
        value = (grid_mean, grid_variance)
        isleaf = True
        left_1, left_2, right_1, right_2 = None, None, None, None
        if grid_size < self.min_grid_size or grid_variance <= self.variance_threshold:
            if self.histogram.get(grid_size) == None: # If it is empty, add the value directly
                self.histogram[grid_size] = 1
            else:
                self.histogram[grid_size] += 1 # If not empty, directly add 1 to block level
            return Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)

        assert grid_size % 2 == 0 
        center_point = grid_size // 2  
        left_1_grid = grid[:center_point, :center_point]
        left_2_grid = grid[:center_point, center_point:]
        right_1_grid = grid[center_point:, :center_point]
        right_2_grid = grid[center_point:, center_point:]

        left_1 = self.node_construct(left_1_grid, pos=pos, grid_size=center_point)
        left_2 = self.node_construct(left_2_grid, pos=(pos[0] + center_point, pos[1]), grid_size=center_point)
        right_1 = self.node_construct(right_1_grid, pos=(pos[0], pos[1] + center_point), grid_size=center_point)
        right_2 = self.node_construct(right_2_grid, pos=(pos[0] + center_point, pos[1] + center_point),
                                      grid_size=center_point)
        # After all recursion is over, there will be no new leaves
        value = False
        isleaf = False
        node = Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)

        return node

    def bfs_for_segmentation(self):
        temp_node = self.root_node
        node_list = [temp_node]

        while len(node_list) > 0:
            temp_node = node_list.pop(0)
            # Leaves cannot be divided, nodes can be divided
            # If it is not a leaf, it is a node.
            # Then, add the child node and leaf to the nodelist, and repeat this operation.
            # until all elements in nodelist are leaves, then execute else, draw
            if temp_node.isLeaf is False: 
                node_list += [temp_node.left_1, temp_node.left_2, temp_node.right_1, temp_node.right_2]
            else:
                pos_x, pos_y, grid_size = temp_node.pos[0], temp_node.pos[1], temp_node.grid_size
                cv2.rectangle(self.image_grid, (pos_x, pos_y), (pos_x + grid_size, pos_y + grid_size), (0, 255, 255), 1)

        return self.image_grid

    def extract_img_features(self, vector_dim=7) -> list:
        # Blocks are arranged from largest to smallest
        feature_info = sorted([self.min_grid_size * (2 ** i) for i in range(vector_dim)], reverse=False)  # H -> L
        img_vector = [self.histogram.get(k, 0) for k in feature_info]
        return img_vector
