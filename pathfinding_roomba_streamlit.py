import streamlit as st
import numpy as np
from aima.search import Problem, astar_search
from PIL import Image, ImageDraw

# Hằng số
TILE_SIZE = 32
MAP_WIDTH = 40
MAP_HEIGHT = 23
SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE

# Load hình ảnh
roomba_img = Image.open("roomba.png").resize((TILE_SIZE, TILE_SIZE))
selection_img = Image.open("selection.png").resize((TILE_SIZE, TILE_SIZE))
map_img = Image.open("map.png").resize((SCREEN_WIDTH, SCREEN_HEIGHT))

# Ma trận mô tả bản đồ (0: vật cản, 1: đường đi)
matrix = [
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	[0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0],
	[0,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0],
	[0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0],
	[0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0],
	[0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# Lớp vấn đề ma trận (cho A* Search)
class GridProblem(Problem):
    def __init__(self, matrix, start, goal):
        super().__init__(start, goal)
        self.matrix = matrix

    def actions(self, state):
        x, y = state
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and self.matrix[ny][nx] == 1:
                actions.append((nx, ny))
        return actions

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, cost, state1, action, state2):
        return cost + 1

    def h(self, node):
        (x1, y1), (x2, y2) = node.state, self.goal
        return abs(x1 - x2) + abs(y1 - y2)

# Hàm tạo đường đi bằng A*
def create_path(matrix, start, goal):
    problem = GridProblem(matrix, start, goal)
    solution = astar_search(problem)
    if solution:
        return [node.state for node in solution.path()]
    else:
        return []

# Hàm vẽ bản đồ với vị trí Roomba và đường đi
def draw_map(matrix, roomba_pos, goal_pos, path):
    img = map_img.copy()
    draw = ImageDraw.Draw(img)

    # Vẽ đường đi nếu có
    if path:
        points = [(x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2) for x, y in path]
        draw.line(points, fill="blue", width=5)

    # Vẽ Roomba
    roomba_x, roomba_y = roomba_pos
    img.paste(roomba_img, (roomba_x * TILE_SIZE, roomba_y * TILE_SIZE), roomba_img)

    # Vẽ vị trí đích
    goal_x, goal_y = goal_pos
    img.paste(selection_img, (goal_x * TILE_SIZE, goal_y * TILE_SIZE), selection_img)

    return img

# Giao diện Streamlit
st.title("Roomba Pathfinding Web App")

# Lưu trạng thái Roomba, đường đi và vị trí đích trong session_state
if "roomba_pos" not in st.session_state:
    st.session_state.roomba_pos = (1, 1)  # Vị trí khởi đầu
if "path" not in st.session_state:
    st.session_state.path = []
if "goal_pos" not in st.session_state:
    st.session_state.goal_pos = (1, 1)  # Vị trí đích khởi đầu

# Hiển thị bản đồ
col1, col2 = st.columns(2)

with col1:
    st.image(draw_map(matrix, st.session_state.roomba_pos, st.session_state.goal_pos, st.session_state.path), use_column_width=True)

# Chọn mục tiêu
with col2:
    st.write("### Chọn mục tiêu")
    goal_x = st.slider("X", 0, MAP_WIDTH - 1, 1)
    goal_y = st.slider("Y", 0, MAP_HEIGHT - 1, 1)
    st.session_state.goal_pos = (goal_x, goal_y)

    # Nút tạo đường đi
    if st.button("Tạo đường đi"):
        start = st.session_state.roomba_pos
        path = create_path(matrix, start, st.session_state.goal_pos)
        if path:
            st.session_state.path = path
            st.success(f"Đã tìm thấy đường đi: {path}")
        else:
            st.error("Không tìm thấy đường đi!")

    # Nút di chuyển Roomba theo đường đi
    if st.button("Di chuyển Roomba"):
        if st.session_state.path:
            st.session_state.roomba_pos = st.session_state.path[-1]
            st.session_state.path = []  # Xóa đường đi sau khi di chuyển
        else:
            st.warning("Không có đường đi để di chuyển!")

# Cập nhật lại hình ảnh sau khi chọn mục tiêu hoặc di chuyển
st.image(draw_map(matrix, st.session_state.roomba_pos, st.session_state.goal_pos, st.session_state.path), use_column_width=True)
