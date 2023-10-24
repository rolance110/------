import math


def lexicographical(path):  # 字典序演算法
    path = next_permutation(path)
    return path


def next_permutation(path):
    n = len(path)
    k = n - 2
    while k >= 0 and path[k] >= path[k+1]:
        k -= 1
    l = n - 1
    while path[l] <= path[k]:
        l -= 1
    path[k], path[l] = path[l], path[k]
    path[k+1:] = path[k+1:][::-1]
    return path


def total_dist(path, mat):  # 計算總路徑長度

    path.insert(0, 0)  # 在最前面插入0，確保從城市0出發
    path.append(0)     # 在最後面再插入0，確保最後回到城市0

    current_total_dist = 0
    for i in range(len(path) - 1):
        current_total_dist = current_total_dist + mat[path[i]][path[i + 1]]
    current_total_dist += mat[path[-1]][path[0]]  # 回到起點
    return current_total_dist


def BF(mat):  # 窮舉法
    city_num = len(mat)
    best_path = list(range(city_num))
    best_distance = total_dist(best_path, mat)
    # 頭尾固定->窮舉(n-1)!種排列
    exhaustive_num = math.factorial(city_num-1)
    # 初始狀態 0->1->2->3->4->0
    path = list(range(1, city_num))

    for i in range(exhaustive_num-1):
        current_total_dist = total_dist(path, mat)
        # 刪掉path的頭尾0，避免被計算
        path = path[1:-1]
        # 執行字典序演算法
        path = lexicographical(path)
        # 更新最佳路徑
        if (current_total_dist < best_distance):
            best_distance = current_total_dist
            best_path = path

    return best_path, best_distance


if __name__ == "__main__":
    mat = [
        [0, 1, 9, 8, 40],
        [1, 0, 2, 35, 50],
        [9, 2, 0, 30, 10],
        [8, 35, 30, 0, 5],
        [40, 50, 10, 5, 0]
    ]

    best_path, best_distance = BF(mat)

    print("best Path:", best_path)
    print("best Distance:", best_distance)
