import random
import math


def total_dist(path, mat):
    current_total_dist = 0
    for i in range(len(path) - 1):
        current_total_dist = current_total_dist + mat[path[i]][path[i + 1]]
    current_total_dist += mat[path[-1]][path[0]]  # 回到起點
    return current_total_dist


def SA(mat, init_path, init_temp, cooling_rate, num_iter):
    num_cities = len(mat)

    current_path = init_path
    current_path.insert(0, 0)  # 在最前面插入0，確保從城市0出發
    current_path.append(0)  # 在最後面再插入0，確保最後回到城市0
    current_distance = total_dist(current_path, mat)
    T = init_temp

    for i in range(num_iter):
        # print("iter:", i, "current_distance:", current_distance)
        # 降低溫度
        T = T*(1 - cooling_rate)
        # T=0 代表冷卻結束
        if (T == 0):
            break

        next_path = current_path.copy()
        # 隨機選擇兩個不同的城市（不包括城市0）進行交換
        city1, city2 = random.sample(range(1, num_cities), 2)
        next_path[city1], next_path[city2] = current_path[city2], current_path[city1]
        next_distance = total_dist(next_path, mat)

        # 計算能量變化
        delta_E = next_distance - current_distance

        # 如果新路徑更短
        if delta_E < 0:
            current_path = next_path
            current_distance = next_distance
        # 按照一定機率接受較差的路徑
        elif random.random() > math.exp(delta_E / T):
            current_path = next_path
            current_distance = next_distance

    # 更新最佳路徑
    best_path = current_path
    best_distance = current_distance

    return best_path, best_distance


if __name__ == "__main__":
    mat = [
        [0, 1, 9, 8, 40],  # A
        [1, 0, 2, 35, 50],  # B
        [9, 2, 0, 30, 10],  # C
        [8, 35, 30, 0, 5],  # D
        [40, 50, 10, 5, 0],  # E
    ]

    num_cities = len(mat)
    init_path = list(range(1, num_cities))  # 不包括城市0
    init_temp = 1000  # 初始温度
    cooling_rate = 0.001  # 冷却率
    num_iter = 1000  # 迭代次数

    best_path, best_dist = SA(
        mat, init_path, init_temp, cooling_rate, num_iter)

    print("最短路徑：", best_path)
    print("最短距離：", best_dist)

